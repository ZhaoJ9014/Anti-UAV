from . import BaseActor
import jittor as jt
import numpy as np


class ModaltActor(BaseActor):
    """ Actor for training the Modal"""
    def generate_loss(self, data, output, mode: str):
        targets =[]
        targets_origin = data['search_anno']
        for i in range(len(targets_origin)):
            h, w =data['search_images'][i][0].shape
            target_origin = targets_origin[i]
            target = {}
            target_origin = target_origin.reshape([1,-1])
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label_np = np.array([0])
            label = jt.array(label_np, dtype=jt.int64)

            target['labels'] = label
            targets.append(target)
        loss_dict = self.objective(output, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item()
                 }

        return losses, stats

    def __call__(self, data):
        """
        args:
            data - a dict {'RGB','IR'} . Each element contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time
        rgb = data['RGB']
        ir = data['IR']
        search = {
            'ir':ir['search_images'],
            'rgb':rgb['search_images']
        }
        template = {
            'ir': ir['template_images'],
            'rgb': rgb['template_images']
        }
        output = self.net(search, template)
        loss = []
        total_rgb = 0
        total_ir = 0
        stats = {}
        output = [output[-1]]
        for scale,scale_out in enumerate(output):
            rgb_output, ir_output = scale_out
            loss_rgb, stats_rgb = self.generate_loss(data=rgb,output=rgb_output,mode='rgb')
            # iou_rgb = stats_rgb['iou']

            loss_ir, stats_ir = self.generate_loss(data=ir,output=ir_output,mode='ir')
            iou_ir = stats_ir['iou']
            # if scale == 0:
            #     iou_scale0 = iou_ir
            #     stats['iou_IR'] = iou_scale0
            for key ,value in stats_rgb.items():
                rgb_key = 'RGB-' + key
                stats[rgb_key] = value
            for key, value in stats_ir.items():
                ir_key = 'IR-' + key + str(scale)
                stats[ir_key] = value
            total_rgb = loss_rgb
            total_ir = loss_ir
            # stats[str(scale)+'_'+'iouRGB'] = iou_rgb
            # stats[str(scale) + '_' + 'iouIR'] = iou_ir
            # stats[str(scale) + '_total'] = stats_rgb['Loss/giou'] + stats_ir['Loss/giou']
            # loss.append([loss_rgb,loss_ir])

        losses =  total_ir + total_rgb
        return losses, stats



class UnFPNActor(BaseActor):
    """ Actor for training the Modal"""
    def generate_loss(self, data, output, mode: str):
        targets =[]
        targets_origin = data['search_anno']
        for i in range(len(targets_origin)):
            h, w =data['search_images'][i][0].shape
            target_origin = targets_origin[i]
            target = {}
            target_origin = target_origin.reshape([1,-1])
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])

            label = jt.array(label, dtype=jt.int64)
            target['labels'] = label
            targets.append(target)
        loss_dict = self.objective(output, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item()
                 }

        return losses, stats

    def __call__(self, data):
        """
        args:
            data - a dict {'RGB','IR'} . Each element contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time
        rgb = data['RGB'].cuda()
        ir = data['IR'].cuda()
        search = {
            'ir':ir['search_images'],
            'rgb':rgb['search_images']
        }
        template = {
            'ir': ir['template_images'],
            'rgb': rgb['template_images']
        }
        output = self.net(search, template)
        loss = []
        total_rgb = 0
        total_ir = 0
        stats = {}
        output = [output]
        for scale,scale_out in enumerate(output):
            rgb_output, ir_output = scale_out
            loss_rgb, stats_rgb = self.generate_loss(data=rgb,output=rgb_output,mode='rgb')
            # iou_rgb = stats_rgb['iou']

            loss_ir, stats_ir = self.generate_loss(data=ir,output=ir_output,mode='ir')
            iou_ir = stats_ir['iou']
            # if scale == 0:
            #     iou_scale0 = iou_ir
            #     stats['iou_IR'] = iou_scale0
            for key ,value in stats_rgb.items():
                rgb_key = 'RGB-' + key
                stats[rgb_key] = value
            for key, value in stats_ir.items():
                ir_key = 'IR-' + key + str(scale)
                stats[ir_key] = value
            total_rgb = loss_rgb
            total_ir = loss_ir
            # stats[str(scale)+'_'+'iouRGB'] = iou_rgb
            # stats[str(scale) + '_' + 'iouIR'] = iou_ir
            # stats[str(scale) + '_total'] = stats_rgb['Loss/giou'] + stats_ir['Loss/giou']
            # loss.append([loss_rgb,loss_ir])

        losses = total_ir + total_rgb
        return losses, stats