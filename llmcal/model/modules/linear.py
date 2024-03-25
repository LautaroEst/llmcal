
import lightning as L
from torch.nn import Linear as _Linear

class Linear(_Linear):

    def init_params(self, fabric: L.Fabric):
        pass

    def get_trainable_parameters(self):
        return self.parameters()