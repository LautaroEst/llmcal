
from datasets import load_from_disk
from scipy.special import softmax

def main():
    dataset = load_from_disk("experiments/refind_inst_0-shot_prompt/tinyllama/all/train").flatten().select_columns(["output.logits", "target"]).with_format("numpy")
    probs = softmax(dataset["output.logits"], axis=1)



if __name__ == "__main__":
    main()