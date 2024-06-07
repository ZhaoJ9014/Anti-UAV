
import jittor as jt

class MLP(jt.nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=128, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # Use a list comprehension instead of a generator expression
        self.layers = jt.nn.ModuleList([jt.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])])

    def execute(self, x):
        for i, layer in enumerate(self.layers):
            x = jt.nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
# class MLP(jt.nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = jt.nn.ModuleList(jt.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

#     def execute(self, x):
#         for i, layer in enumerate(self.layers):
#             x = jt.nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x

class PredictionNeck(jt.nn.Module):
    def __init__(self,hidden_dim,num_layers):
        super(PredictionNeck, self).__init__()
        hidden_dim = 512
        num_classes = 1
        self.class_embed_rgb = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed_rgb = MLP(hidden_dim, hidden_dim, 4, 3)
        self.class_embed_ir = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed_ir = MLP(hidden_dim, hidden_dim, 4, 3)

    def execute(self,hs):
        scales_out = []
        # hs = [hs[-1]]
        for feature in hs :
            outputs_class_rgb = self.class_embed_rgb(feature)  #1,2,512,2
            outputs_class_ir = self.class_embed_ir(feature)
            outputs_coord_rgb = self.bbox_embed_rgb(feature).sigmoid()
            outputs_coord_ir = self.bbox_embed_ir(feature).sigmoid()
            out = []
            out.append({'pred_logits': outputs_class_rgb, 'pred_boxes': outputs_coord_rgb})
            out.append({'pred_logits': outputs_class_ir, 'pred_boxes': outputs_coord_ir})
            scales_out.append(out)
        return scales_out

