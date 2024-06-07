import ltr_torch.admin.loading as ltr_loading

class NetWrapper:
    """Used for wrapping networks in pytracking.
    Network modules and functions can be accessed directly as if they were members of this class."""
    _rec_iter=0
    def __init__(self, net_path, use_gpu=True, initialize=False, **kwargs):
        self.net_path = net_path
        self.use_gpu = use_gpu
        self.net = None
        self.net_kwargs = kwargs
        if initialize:
            self.initialize()

    def __getattr__(self, name):
        if self._rec_iter > 0:
            self._rec_iter = 0
            return None
        self._rec_iter += 1
        try:
            ret_val = getattr(self.net, name)
        except Exception as e:
            self._rec_iter = 0
            raise e
        self._rec_iter = 0
        return ret_val

    def load_network(self):
        self.net, _ = ltr_loading.load_network(self.net_path, **self.net_kwargs)
        if self.use_gpu:
            self.cuda()
        self.eval()

    def initialize(self):
        self.load_network()


class NetWithBackbone(NetWrapper):
    """Wraps a network with a common backbone.
    Assumes the network have a 'extract_backbone_features(image)' function."""

    def __init__(self, net_path, use_gpu=True, initialize=False, **kwargs):
        super().__init__(net_path, use_gpu, initialize, **kwargs)

    def initialize(self):
        super().initialize()

    def template(self, z):
        self.net.template(z)

    def track(self, image):
        return self.net.track(image)