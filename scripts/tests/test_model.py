
from llmcal import load_model_and_tokenizer

def main():
    # model_name = "gpt2-xl"
    model_name = "meta-llama/Llama-2-7b-hf"
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    sentences = [
        "The cat sat on the ",
        "The dog sat on the mat.",
        "The cat sat ",
        "The dog sat on the other side of the couch.",
        "The dog "
    ]

    labels = [" this is label 1", " label2", " this label3"]
    encoded_labels = [tokenizer(label, return_tensors="pt").input_ids for label in labels]
    print(encoded_labels)

    encoded_sentences = tokenizer(sentences, return_tensors="pt", padding=True)
    print(encoded_sentences)
    output = model(**encoded_sentences, encoded_labels=encoded_labels)
    print(output["logits"])




if __name__ == '__main__':
    main()