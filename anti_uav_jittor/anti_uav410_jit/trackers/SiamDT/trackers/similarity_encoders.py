import jittor.nn as nn

from mmcv.cnn import normal_init

__all__ = ['RPN_Similarity_Learning', 'RCNN_Similarity_Learning']


class RPN_Similarity_Learning(nn.Module):

    def __init__(self,
                 roi_size=7,
                 channels=256,
                 featmap_num=5):
        super(RPN_Similarity_Learning, self).__init__()
        self.proj_query = nn.ModuleList([
            nn.Conv2d(channels, channels, roi_size, padding=0)
            for _ in range(featmap_num)])
        self.proj_out = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, padding=0)
            for _ in range(featmap_num)])

    def forward(self, template, feats_x):

        n_imgs = len(feats_x[0])
        for i in range(n_imgs):
            n_instances = len(template[i])
            for j in range(n_instances):
                query = template[i][j:j + 1]
                gallary = [f[i:i + 1] for f in feats_x]
                out_ij = [self.proj_query[k](query) * gallary[k]
                          for k in range(len(gallary))]
                out_ij = [p(o) for p, o in zip(self.proj_out, out_ij)]
                yield out_ij, i, j

    def init_weights(self):
        for m in self.proj_query:
            normal_init(m, std=0.01)
        for m in self.proj_out:
            normal_init(m, std=0.01)


class RCNN_Similarity_Learning(nn.Module):

    def __init__(self, channels=256):
        super(RCNN_Similarity_Learning, self).__init__()
        self.proj_z = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_x = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, z, x):
        # assume one image and one instance only
        assert len(z) == 1
        return self.proj_out(self.proj_x(x) * self.proj_z(z))

    def init_weights(self):
        normal_init(self.proj_z, std=0.01)
        normal_init(self.proj_x, std=0.01)
        normal_init(self.proj_out, std=0.01)
