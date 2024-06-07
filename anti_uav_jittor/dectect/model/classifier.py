import warnings


import jittor as jt
from jittor.nn import AdaptiveMaxPool2d
warnings.filterwarnings("ignore")
class RoIPool(jt.nn.Module):
    def __init__(self, output_size, spatial_scale):
        """
        初始化 RoIPool 模块.
        
        参数:
        - output_size: (tuple) 输出尺寸，格式为 (height, width)
        - spatial_scale: (float) 缩放因子，用于将输入的ROI坐标缩放到特征图尺度
        """
        super(RoIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, features, rois):
        """
        执行 RoI 池化操作.
        
        参数:
        - features: (Tensor) 输入的特征图，形状为 (N, C, H, W)
        - rois: (Tensor) ROI 坐标，形状为 (num_rois, 5)，每行格式为 (batch_index, x1, y1, x2, y2)
        
        返回:
        - output: (Tensor) 池化后的特征图，形状为 (num_rois, C, output_size[0], output_size[1])
        """
        num_rois = rois.size(0)
        output = []

        for i in range(num_rois):
            roi = rois[i]
            batch_index = int(roi[0])
            x1 = int(round(roi[1] * self.spatial_scale))
            y1 = int(round(roi[2] * self.spatial_scale))
            x2 = int(round(roi[3] * self.spatial_scale))
            y2 = int(round(roi[4] * self.spatial_scale))

            roi_feature = features[batch_index, :, y1:y2+1, x1:x2+1]

            adaptive_max_pool2d = AdaptiveMaxPool2d(self.output_size)

            pooled_feature = adaptive_max_pool2d(roi_feature)

            
            output.append(pooled_feature)

        return jt.concat(output, dim=0)

class VGG16RoIHead(jt.nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        # --------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        # --------------------------------------#
        self.cls_loc = jt.nn.Linear(4096, n_class * 4)
        # -----------------------------------#
        #   对ROIPooling后的的结果进行分类
        # -----------------------------------#
        self.score = jt.nn.Linear(4096, n_class)
        # -----------------------------------#
        #   权值初始化
        # -----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois = jt.flatten(rois, 0, 1)
        roi_indices = jt.flatten(roi_indices, 0, 1)

        rois_feature_map = jt.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

        indices_and_rois = jt.concat([roi_indices[:, None], rois_feature_map], dim=1)
        # -----------------------------------#
        #   利用建议框对公用特征层进行截取
        # -----------------------------------#
        pool = self.roi(x, indices_and_rois)
        # -----------------------------------#
        #   利用classifier网络进行特征提取
        # -----------------------------------#
        pool = pool.view(pool.size(0), -1)
        # --------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 4096]
        # --------------------------------------------------------------#
        fc7 = self.classifier(pool)

        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores


class Resnet50RoIHead(jt.nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        # --------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        # --------------------------------------#
        self.cls_loc = jt.nn.Linear(2048, n_class * 4)
        # -----------------------------------#
        #   对ROIPooling后的的结果进行分类
        # -----------------------------------#
        self.score = jt.nn.Linear(2048, n_class)
        # -----------------------------------#
        #   权值初始化
        # -----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois = jt.flatten(rois, 0, 1)
        roi_indices = jt.flatten(roi_indices, 0, 1)

        rois_feature_map = jt.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

        indices_and_rois = jt.concat([roi_indices[:, None], rois_feature_map], dim=1)
        # -----------------------------------#
        #   利用建议框对公用特征层进行截取
        # -----------------------------------#
        pool = self.roi(x, indices_and_rois)
        # -----------------------------------#
        #   利用classifier网络进行特征提取
        # -----------------------------------#
        fc7 = self.classifier(pool)
        # --------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        # --------------------------------------------------------------#
        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        # m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        jt.init.gauss_(m.weight).fmod_(2).mul_(stddev).add_(mean)
    else:
        jt.init.gauss_(m.weight, mean, stddev)
        # m.weight.data.normal_(mean, stddev)
        # m.bias.data.zero_()
        jt.init.constant_(m.bias, 0.0)

