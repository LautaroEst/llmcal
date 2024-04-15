from typing import Literal
from .base_models.litgpt import LitGPT, LoRAGPT, LitGPTForClassification, LoRAGPTForClassification
from transformers import AutoModelForSequenceClassification
from litgpt.lora import mark_only_lora_as_trainable
from .causal_lm_for_classification import CausalLMForClassification
from .sequence_classification import SequenceClassificationModel
from .calibration import CausalLMForClassificationPlusCalibration
from .base_models.affine_calibration import AffineCalibrator


def _is_lit_model(checkpoint_dir):
    return True

def load_model_from_checkpoint(
    fabric, 
    checkpoint_dir,
    model_type: Literal["language_model", "sequence_classifier"] = "language_model",
    method: Literal["no_adaptation", "full_ft", "lora", "embeddings_finetuning", "affine_calibration"] = "no_adaptation",
    **kwargs
):
    # if model_type == "language_model":
    #     if _is_lit_model(checkpoint_dir):
    #         if method in ["no_adaptation", "full_ft", "embeddings_finetuning", "affine_calibration"]:
    #             base_model = LitGPT.from_pretrained(fabric, checkpoint_dir)
    #             for params in base_model.parameters():
    #                 params.requires_grad = method == "full_ft"
    #         elif method == "lora":
    #             lora_kwargs = {k: kwargs[k] for k in kwargs if k.startswith("lora_")}
    #             kwargs = {k: kwargs[k] for k in kwargs if not k.startswith("lora_")}
    #             base_model = LoRAGPT.from_pretrained(fabric, checkpoint_dir, **lora_kwargs)
    #             mark_only_lora_as_trainable(base_model)
    #         else:
    #             raise ValueError(f"Method {method} not recognized")
    #         model = CausalLMForClassification(base_model, **kwargs)
    #     else:
    #         raise NotImplementedError("Only LIT models are supported for language model")

    # elif model_type == "sequence_classification":
    #     if _is_lit_model(checkpoint_dir):
    #         if method in ["no_adaptation", "full_ft", "embeddings_finetuning", "affine_calibration"]:
    #             base_model = LitGPTForClassification.from_pretrained(fabric, checkpoint_dir)
    #             for params in base_model.parameters():
    #                 params.requires_grad = method == "full_ft"
    #         elif method == "lora":
    #             lora_kwargs = {k: kwargs[k] for k in kwargs if k.startswith("lora_") or k == "num_classes"}
    #             kwargs = {k: kwargs[k] for k in kwargs if not k.startswith("lora_")}
    #             if "lora_head" in lora_kwargs:
    #                 raise ValueError("lora_head should not be passed as a keyword argument when model_type is sequence_classifier")
    #             base_model = LoRAGPTForClassification.from_pretrained(fabric, checkpoint_dir, **lora_kwargs)
    #             mark_only_lora_as_trainable(base_model)
    #         else:
    #             raise ValueError(f"Method {method} not recognized")
    #     else:
    #         num_classes = kwargs.pop("num_classes", 2)
    #         with fabric.init_module():
    #             base_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, num_classes=num_classes)
    #     model = SequenceClassificationModel(base_model, **kwargs)
    # else:
    #     raise ValueError(f"Model type {model_type} not recognized")
    
    # return model

    is_lit_model = _is_lit_model(checkpoint_dir)

    if model_type == "language_model" and is_lit_model and method in ["no_adaptation", "affine_calibration"]:
        base_model = LitGPT.from_pretrained(fabric, checkpoint_dir)
        if method == "affine_calibration":
            for params in base_model.parameters():
                params.requires_grad = False
            calibration_layer = AffineCalibrator(**kwargs)
            calibration_layer.init_params(fabric)
        else:
            calibration_layer = None
        model = CausalLMForClassificationPlusCalibration(base_model, calibration_layer, **kwargs)
    
    elif model_type == "language_model" and is_lit_model and method in ["full_ft", "lora"]:
        if method == "lora":
            lora_kwargs = {k: kwargs[k] for k in kwargs if k.startswith("lora_")}
            kwargs = {k: kwargs[k] for k in kwargs if not k.startswith("lora_")}
            base_model = LoRAGPT.from_pretrained(fabric, checkpoint_dir, **lora_kwargs)
            mark_only_lora_as_trainable(base_model)
        else:
            base_model = LitGPT.from_pretrained(fabric, checkpoint_dir)
            for params in base_model.parameters():
                params.requires_grad = True
        model = FinetunableCausalLMForClassification(base_model, **kwargs)
    
    elif model_type == "language_model"and not is_lit_model:
        raise NotImplementedError("Only LIT models are supported for language model")

    elif model_type == "sequence_classifier" and is_lit_model and method in ["no_adaptation", "affine_calibration"]:
        base_model = LitGPTForClassification.from_pretrained(fabric, checkpoint_dir)
        for params in base_model.parameters():
            params.requires_grad = False
        calibration_layer = AffineCalibrator(**kwargs) if method == "affine_calibration" else None
        model = SequenceClassificationModelPlusCalibration(base_model, calibration_layer, **kwargs)

    elif model_type == "sequence_classifier" and is_lit_model and method in ["full_ft", "lora"]:
        if method == "lora":
            lora_kwargs = {k: kwargs[k] for k in kwargs if k.startswith("lora_") or k == "num_classes"}
            kwargs = {k: kwargs[k] for k in kwargs if not k.startswith("lora_")}
            base_model = LoRAGPTForClassification.from_pretrained(fabric, checkpoint_dir, **lora_kwargs)
            mark_only_lora_as_trainable(base_model)
        else:
            base_model = LitGPTForClassification.from_pretrained(fabric, checkpoint_dir)
            for params in base_model.parameters():
                params.requires_grad = True
        model = FinetunableSequenceClassificationModel(base_model, **kwargs)

    elif model_type == "sequence_classifier" and not is_lit_model and method == "full_ft":
        with fabric.init_module():
            base_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        for params in base_model.parameters():
            params.requires_grad = True
        model = FinetunableSequenceClassificationModel(base_model, **kwargs)

    else:
        raise ValueError(f"Model type {model_type} not recognized")    
    
    return model
    
        

        
        


    
    
    
    
    
    
    
    # if model_class in ["LitGPT", "LoRAGPT"]:
    #     model_class = getattr(litgpt, model_class)
    #     lm = model_class.from_pretrained(fabric, checkpoint_dir)
    #     model = litgpt.LanguageModelForClassification(lm, **kwargs)
    # elif model_class in ["BertForSequenceClassification"]:
    #     model_class = getattr(transformers, model_class)
    #     with fabric.init_module():
    #         model = model_class.from_pretrained(checkpoint_dir)
    # else:
    #     raise ValueError(f"Model class {model_class} not recognized")
    # return model