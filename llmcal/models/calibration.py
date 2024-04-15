
import lightning as L
from torch.optim import LBFGS

class CausalLMForClassificationPlusCalibration(L.LightningModule):
    
    def __init__(
        self, 
        base_model, 
        calibration_layer, 
        cache_dir: str = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        micro_batch_size: int = 1
    ):
        super().__init__()
        self.base_model = base_model
        self.calibration_layer = calibration_layer

        self.cache_dir = cache_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.micro_batch_size = micro_batch_size

    def forward(self, *args, **kwargs):
        outputs = self.base_model(*args, **kwargs)
        if self.calibration_layer is not None:
            outputs = self.calibration_layer(outputs["logits"])
        return outputs

    def configure_optimizers(self):
        if self.calibration_layer is None:
            return
        return LBFGS(
                [p for p in self.calibration_layer.parameters() if p.requires_grad],
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            ) 
            