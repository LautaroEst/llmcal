
import torch

def main():
    checkpoint = torch.load("experiments/sst2/n=100_rs=738/prefix_basic_sst2/lm_tinyllama_3T_bf16/lora_r=8/lit_model.pth")
    print(checkpoint)


if __name__ == "__main__":
    main()