# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import jittor as jt
class TrackingMatcher(jt.nn.Module):
    """This class computes an assignment between the ground-truth and the predictions of the network.
    The corresponding feature vectors within the ground-truth box are matched as positive samples.
    """

    def __init__(self):
        super().__init__()


    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Always tensor([0]) represents the foreground,
                           since single target tracking has only one foreground category
                 "boxes": Tensor of dim [1, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order),
                  and it is always 0, because single target tracking has only one target per image
            For each batch element, it holds:
                len(index_i) = len(index_j)
        """
        indices = []
        bs, num_queries = outputs["pred_logits"].shape[:2]
        for i in range(bs):
            cx, cy, w, h = targets[i]['boxes'][0]
            cx = cx.item(); cy = cy.item(); w = w.item(); h = h.item()
            xmin = cx-w/2; ymin = cy-h/2; xmax = cx+w/2; ymax = cy+h/2
            area = w*h
            len_feature = int(np.sqrt(num_queries))
            Xmin = int(np.ceil(xmin*len_feature))
            Ymin = int(np.ceil(ymin*len_feature))
            Xmax = int(np.ceil(xmax*len_feature))
            Ymax = int(np.ceil(ymax*len_feature))
            if Xmin == Xmax:
                Xmax = Xmax+1
            if Ymin == Ymax:
                Ymax = Ymax+1
            a = np.arange(0, num_queries, 1)
            b = a.reshape([len_feature, len_feature])
            c = b[Ymin:Ymax, Xmin:Xmax].flatten()
            d = np.zeros(len(c), dtype=int)
            indice = (c, d)
            indices.append(indice)
        return [(jt.Var(i, dtype=jt.int64), jt.Var(j, dtype=jt.int64)) for i, j in indices]

def build_matcher():
    return TrackingMatcher()
