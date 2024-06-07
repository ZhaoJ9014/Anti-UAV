import jittor as jt
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking.un_fpn import unFPN_resnet50
from ltr.models.tracking.modal import  Modal_loss
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.dataset import AntiUav_fusion


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.device = 'cuda'
    settings.description = 'UnFPN with default settings.'
    settings.batch_size = 2
    settings.module_name = 'UnFPN'
    settings.num_workers = 0
    settings.multi_gpu = False
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 4.0
    settings.template_area_factor = 2.0
    settings.search_feature_sz = 32
    settings.template_feature_sz = 16
    settings.search_sz = settings.search_feature_sz * 8
    settings.temp_sz = settings.template_feature_sz * 8
    settings.center_jitter_factor = {'search': 3, 'template': 0}
    settings.scale_jitter_factor = {'search': 0.25, 'template': 0}
    settings.backbone_name = 'swin_base_patch4_window12_384'
    # Transformer
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048
    settings.featurefusion_layers = 2


    Antiuav_train_fusion = AntiUav_fusion(root='D:\study\\track\dataset\\Anti-UAV-RGBT\\train')
    # Antiuav_val_fusion = AntiUav_fusion(root='D:\study\\track\dataset\Anti-UAV-RGBT\\val')

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # Data processing to do on the training pairs
    data_processing_train = processing.TransTProcessing(search_area_factor=settings.search_area_factor,
                                                      template_area_factor = settings.template_area_factor,
                                                      search_sz=settings.search_sz,
                                                      temp_sz=settings.temp_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    # The sampler for training
    dataset_train = sampler.ModalSampler([Antiuav_train_fusion], [1],
                                samples_per_epoch=2000*settings.batch_size, max_gap=5, processing=data_processing_train)

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                              shuffle=True, drop_last=True, stack_dim=0)
    print(len(dataset_train))
    # Create network and actor
    model = unFPN_resnet50(settings)
    #
    # # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model, dim=0)
    #
    objective = Modal_loss(settings)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    #
    actor = actors.UnFPNActor(net=model, objective=objective)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]

    optimizer = jt.optim.AdamW(param_dicts, lr=1e-6, weight_decay=1e-4)
    lr_scheduler = jt.lr.Scheduler(optimizer, step_size=500, gamma=0.1)
    # Create trainer
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)
    # trainer._checkpoint_dir = 'D:\study\\track\\unsupervisedTracking\modal\modal\model_2_un'


    trainer.train(30, load_latest=False, fail_safe=True)

import ltr.admin.settings as ws_settings
settings = ws_settings.Settings()

run(settings)