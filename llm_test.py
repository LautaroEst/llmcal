
from llmcal.model.modules.lit_gpt import LitGPT, LitGPTLanguageModel, LitGPTPromptClassifier, LitGPTSequenceClassification
from llmcal.model.modules.tokenizer import LitGPTTokenizer
import lightning as L
from lightning.fabric.plugins import BitsandbytesPrecision
from lit_gpt.utils import get_default_supported_precision
from lit_gpt.lora import mark_only_lora_as_trainable
import torch

prompt = 'Classify the sentiment of the sentence below in one of the following categories:\nA. Negative\nB. Positive\nQuestion: What is the sentiment of the sentence "sandler "?\nAnswer: B\nQuestion: What is the sentiment of the sentence "a climactic hero \'s "?\nAnswer: B\nQuestion: What is the sentiment of the sentence "pretty funny "?\nAnswer: B\nQuestion: What is the sentiment of the sentence "albeit a visually compelling one "?\nAnswer: B\nQuestion: What is the sentiment of the sentence "contrived to be as naturally charming as it needs to be "?\nAnswer:'


def init_fabric(model_args):

    # Configure precision and quantization
    precision = model_args.get("precision", get_default_supported_precision(training=False))
    quantize = model_args.get("quantize", None)
    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    # Configure Callbacks
    callbacks = None

    # Configure Loggers
    loggers = None

    # Init fabric
    fabric = L.Fabric(
        accelerator=model_args.get("accelerator", "auto"),
        strategy=model_args.get("strategy", "auto"),
        devices=model_args.get("devices", "auto"),
        num_nodes=model_args.get("num_nodes", 1),
        precision=precision,
        plugins=plugins,
        callbacks=callbacks,
        loggers=loggers,
    )
    fabric.launch()

    return fabric

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    # model_name = "meta-llama/Llama-2-7b-hf"

    fabric = init_fabric({"accelerator": "gpu", "devices": 1, "precision": "bf16-true"})
    with fabric.init_module(empty_init=True):
        # model = LitGPT(model_name)
        # model = LitGPTLanguageModel(model_name)
        # model = LitGPTPromptClassifier(model_name, embedding_pooling="last")
        model = LitGPTSequenceClassification(model_name, embedding_pooling="last", num_classes=3)
    
    for name, param in model.named_parameters():
        param.requires_grad_(True)
    # model.classifier.weight.requires_grad_(True)
    # print(model.classifier.weight)
    # mark_only_lora_as_trainable(model)
    model = fabric.setup_module(model)
    trainable_params = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.SGD(trainable_params, lr=0.01, momentum=0, weight_decay=1e-4)
    optimizer = fabric.setup_optimizers(optimizer)
    model.init_params(fabric)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    model.classifier.weight.data.normal_(mean=0.0, std=0.02)

    tokenizer = LitGPTTokenizer(model_name)

    inputs = {k: v.to(fabric.device) for k, v in tokenizer([prompt]).items()}
    targets = torch.tensor([0]).to(fabric.device)

    model.train()
    logits = model(prompt_ids=inputs["input_ids"], prompt_mask=inputs["attention_mask"])
    loss = torch.nn.functional.cross_entropy(logits, targets)
    print(model.classifier.weight.grad)
    fabric.backward(loss)
    print(loss)
    print(model.classifier.weight.grad)
    optimizer.step()
    optimizer.zero_grad()
    print(optimizer)

    # inputs = tokenizer([prompt])["input_ids"].to(fabric.device)
    # probs = model._forward_single_sample(inputs)["logits"][0, -1, :].softmax(0).cpu()
    
    # inputs = tokenizer([prompt])
    # logits = model(prompt_ids=inputs["input_ids"].to(fabric.device), prompt_mask=inputs["attention_mask"].to(fabric.device))["logits"]
    
    # inputs = tokenizer([prompt])
    # answers = [[tokenizer([answer])["input_ids"][0,1:].to(fabric.device) for answer in ["A", "B"]]]
    # logits = model(
    #     prompt_ids=inputs["input_ids"].to(fabric.device), 
    #     prompt_mask=inputs["attention_mask"].to(fabric.device),
    #     answers_ids=answers
    # )["logits"]

    
    
    # probs = logits[0, -1, :].softmax(0).cpu()
    # probs = logits[0,:].softmax(0).cpu()
    # sorted_probs, indices  = torch.topk(probs, 5)
    # print(indices)
    # print(sorted_probs)
    # print(tokenizer.tokenizer.decode(indices))



    


if __name__ == '__main__':
    main()
    # fabric = init_fabric({"accelerator": "gpu", "devices": 1, "precision": "bf16-true"})
    # with fabric.init_module():
    #     m = torch.nn.Linear(10, 10)
    #     m.requires_grad_(True)
    # with fabric.init_tensor():
    #     a = torch.randn(10, 10)
    # loss = m(a).sum()
    # fabric.backward(loss)
    # print(m.weight.grad)