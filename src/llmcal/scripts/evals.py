
from datasets import load_dataset
import numpy as np
from scipy.special import log_softmax

def compute_nce(scores, labels):
    logprobs = log_softmax(scores, axis=1)
    ce = -logprobs[np.arange(len(labels)), labels].mean()
    priors = np.bincount(labels) / len(labels)
    ce_priors = -np.mean(np.log(priors[labels]))
    nce = ce / ce_priors
    return ce, nce

def compute_ner(scores, labels):
    preds = np.argmax(scores, axis=1)
    er = np.mean(preds != labels)
    max_label = np.bincount(labels).argmax()
    ner = er / np.mean(preds != max_label)
    return er, ner
    


def main():
    data = load_dataset("meta-llama/Llama-3.2-1B-evals", "Llama-3.2-1B-evals__mmlu__details", split="latest")
    # data = load_dataset("meta-llama/Llama-3.1-405B-evals", "Llama-3.1-405B-evals__mmlu__details", split="latest")
    classes = ["A", "B", "C", "D"]
    # keep columns "output_choice_negative_log_likelihood", "input_correct_responses"
    data = data.select_columns(["output_choice_negative_log_likelihoods", "input_correct_responses"]).to_pandas()
    data = data.rename(columns={"output_choice_negative_log_likelihoods": "score", "input_correct_responses": "label"})
    scores = np.vstack(data["score"].apply(lambda x: np.array(x["raw"])))
    labels = data["label"].apply(lambda x: classes.index(x[0].split(" ")[1])).astype(int).values.flatten()
    ce, nce = compute_nce(scores, labels)
    er, ner = compute_ner(scores, labels)
    goodness = nce * ner
    print(f"NCE: {nce}")
    print(f"CE: {ce}")
    print(f"NER: {ner}")
    print(f"ER: {er}")
    print(f"Goodness: {goodness}")

if __name__ == "__main__":
    main()