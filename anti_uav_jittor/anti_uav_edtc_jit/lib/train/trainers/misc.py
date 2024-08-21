import jittor as jt
# from torch._six import inf

class GradScaler:
    def __init__(self, init_scale=2**16):
        self.scale = init_scale

    def scale_gradients(self, optimizer, loss):
        # Scale the loss
        scaled_loss = loss * self.scale
        scaled_loss.backward()

        # Get the gradients in a list and check for inf/nan
        params = []
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)

        # Check if any gradients are infinite or NaN
        inf_or_nan = False
        for p in params:
            if jt.any(jt.isnan(p.grad)) or jt.any(jt.isinf(p.grad)):
                inf_or_nan = True
                break

        # If gradients are finite, update the parameters
        if not inf_or_nan:
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad /= self.scale
                        p.update(p.grad)
        else:
            # If gradients are infinite or NaN, skip the update
            print("Skipping update due to inf/nan gradients")

        # Adjust the scale
        if not inf_or_nan:
            self.scale *= 2
        else:
            self.scale /= 2

    def unscale_gradients(self, optimizer):
        # Unscale the gradients
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad /= self.scale

    def step(self, optimizer):
        # Perform the optimizer step
        optimizer.step()

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = self.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

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
    
    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> jt.Var:
    if isinstance(parameters, jt.Var):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return jt.var(0.)
    device = parameters[0].grad.device
    if norm_type == jt.Var(float('inf')):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = jt.norm(jt.stack([jt.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
