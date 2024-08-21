import os
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import jittor as jt
import time
# from torch.utils.data.distributed import DistributedSampler
# from torch.cuda.amp import autocast
from lib.train.trainers.misc import NativeScalerWithGradNormCount as NativeScaler

class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, accum_iter=1,
                 use_amp=False, shed_args=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler, shed_args)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        self.accum_iter = accum_iter
        if use_amp:
            print("Using amp")
            self.loss_scaler = NativeScaler()

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def clip_grad_norm_(parameters, max_norm, norm_type=2):
        """
        Clips gradient norm of an iterable of parameters.
        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.
        Args:
            parameters (Iterable[jt.Var]): an iterable of Tensors or Variables
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be 'inf' for
                infinity norm.
        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        
        if norm_type == float('inf'):
            total_norm = max(p.grad.norm(p=norm_type) for p in parameters if p.grad is not None)
        else:
            total_norm = 0
            for p in parameters:
                if p.grad is not None:
                    param_norm = p.grad.norm(p=norm_type)
                    total_norm += param_norm ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                if p.grad is not None:
                    p.grad *= clip_coef

        return total_norm

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        # torch.set_grad_enabled(loader.training)
        if loader.training:
            jt.flags.no_grad = False
        else:
            jt.flags.no_grad = True


        self._init_timing()

        self.optimizer.zero_grad()
        for data_iter_step, data in enumerate(loader, 1):
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            # else:
            #     with autocast():
            #         loss, stats = self.actor(data)

            loss /= self.accum_iter
            # backward pass and update weights
            if loader.training:
                # self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if (data_iter_step + 1) % self.accum_iter == 0:
                        if self.settings.grad_clip_norm > 0:
                            self.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                        self.optimizer.step()
                else:
                    self.loss_scaler(loss, self.optimizer, parameters=self.actor.net.parameters(),
                                     clip_grad=self.settings.grad_clip_norm,
                                     update_grad=(data_iter_step + 1) % self.accum_iter == 0)

            if (data_iter_step + 1) % self.accum_iter == 0:
                self.optimizer.zero_grad()
            # torch.cuda.synchronize()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(data_iter_step, loader, batch_size)

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                # if isinstance(loader.sampler, DistributedSampler):
                #     loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training and self.lr_scheduler is not None:
                try:
                    lr_list = self.lr_scheduler.get_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
