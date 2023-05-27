import torch

def optimization_config(cls):
    def register(self, name, tensor, lr=None, wd=None):
        """Register a tensor with configurable learning rate and weight decay"""
        
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim['lr'] = lr
            if wd is not None: optim['wd'] = wd
            setattr(getattr(self, name), '_optim', optim)