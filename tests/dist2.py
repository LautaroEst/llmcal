
import torch
import lightning as L

def setup():

    base_model_config = {
        "accelerator": "cpu",
        "strategy": "auto",
        "devices": 2,
        "num_nodes": 1,
        "precision": 32,
        "plugins": None,
        "callbacks": None,
        "loggers": None,
    }

    fabric = L.Fabric(
        accelerator=base_model_config.get("accelerator", "cpu"),
        strategy=base_model_config.get("strategy", "auto"),
        devices=base_model_config.get("devices", 1),
        num_nodes=base_model_config.get("num_nodes", 1),
        precision=base_model_config.get("precision", 32),
        plugins=base_model_config.get("plugins", None),
        callbacks=base_model_config.get("callbacks", None),
        loggers=base_model_config.get("loggers", None),
    )

    fabric.launch(main, "arg1")


class MyModule(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.model(x)
    
def main(fabric, arg1):

    with fabric.init_module():
        model = MyModule()
    model = fabric.setup(model)
    state = {"arg1": 1}
    print(fabric.global_rank)

    print("Hello world", fabric.global_rank)
    fabric.save("state.ckpt", state)
    print(arg1, fabric.global_rank)

    x = model(torch.randn(2, 10))
    print(x, fabric.global_rank)
    fabric.barrier()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    setup()
