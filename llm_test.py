
from llmcal.model.modules.lora import LitGPT, LitGPTLanguageModel, LitGPTPromptClassifier, LitGPTSequenceClassification
from llmcal.model.modules.tokenizer import LitGPTTokenizer
import lightning as L
from lightning.fabric.plugins import BitsandbytesPrecision
from lit_gpt.utils import get_default_supported_precision
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

    return fabric

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

    fabric = init_fabric({"accelerator": "gpu", "devices": 1, "precision": "bf16-true"})
    with fabric.init_module(empty_init=True):
        # model = LitGPT(model_name)
        # model = LitGPTLanguageModel(model_name)
        # model = LitGPTPromptClassifier(model_name, embedding_pooling="last")
        model = LitGPTSequenceClassification(model_name, embedding_pooling="last", num_classes=3)
    tokenizer = LitGPTTokenizer(model_name)
    model.init_params(fabric)
    model.eval()

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

    inputs = tokenizer([prompt])
    model.classifier.weight.data.normal_(mean=0.0, std=0.02)
    logits = model(prompt_ids=inputs["input_ids"].to(fabric.device), prompt_mask=inputs["attention_mask"].to(fabric.device))

    # probs = logits[0, -1, :].softmax(0).cpu()
    # probs = logits[0,:].softmax(0).cpu()
    # sorted_probs, indices  = torch.topk(probs, 5)
    # print(indices)
    # print(sorted_probs)
    # print(tokenizer.tokenizer.decode(indices))
    print(logits)



    


if __name__ == '__main__':
    main()