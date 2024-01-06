
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def main():
    model_name = "meta-llama/Llama-2-7b-hf"
    # model_name = "gpt2-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    encoder_sentences = [
        "This is a sentence.",
        "This is another sentence.",
        "This is a third sentence.",
    ]
    encoder_batch = tokenizer(encoder_sentences, return_tensors="pt", padding=True)
    encoder_output = model(use_cache=True, **encoder_batch)
    
    decoder_sentences = [
        " Bye Bye Bye",
        " Bye Bye Bye",
        " Bye Bye Bye",
    ]
    decoder_batch = tokenizer(decoder_sentences, return_tensors="pt", padding=True)
    
    batch_size = len(decoder_sentences)
    label_len = decoder_batch["attention_mask"].shape[1]
    sequence_lens = encoder_batch["attention_mask"].sum(dim=-1,keepdim=True)
    
    decoder_batch["position_ids"] = torch.arange(label_len, device=sequence_lens.device).repeat(batch_size,1) + sequence_lens
    decoder_batch["attention_mask"] = torch.cat(
        (
            encoder_batch["attention_mask"],
            torch.ones(
                (batch_size, label_len),
                dtype=torch.long,
                device=encoder_batch["attention_mask"].device
            )
        ),
        dim=1
    )
    decoder_batch["past_key_values"] = DynamicCache.from_legacy_cache(encoder_output["past_key_values"])
    output = model(use_cache=False, **decoder_batch)
    print(output.logits.shape)

    



if __name__ == "__main__":
    main()