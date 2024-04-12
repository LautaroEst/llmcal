
import torch

class DummyDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.prompts_ids = torch.tensor([
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
            [1, 1, 2, 3, 4],
            [6, 6, 7, 8, 9],
            [11, 11, 12, 13, 14],
            [16, 16, 17, 18, 19],
            [21, 21, 22, 23, 24],
            [2, 1, 2, 3, 4],
            [7, 6, 7, 8, 9],
            [12, 11, 12, 13, 14],
            [17, 16, 17, 18, 19],
            [22, 21, 22, 23, 24]
        ])
        self.prompt_mask = torch.tensor([
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1]
        ])
        self.answers_ids = [
            torch.tensor([25, 26, 27, 28, 29]),
            torch.tensor([30, 31, 32, 34]),
            torch.tensor([35, 38, 39]),
            torch.tensor([45, 46, 47, 48, 49])
        ]
        self.labels = [0, 3, 2, 1, 1, 4, 4, 2, 1, 1, 2, 1, 2, 1, 1]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "prompt_ids": self.prompts_ids[idx].unsqueeze(0),
            "prompt_mask": self.prompt_mask[idx].unsqueeze(0),
            "answers_ids": self.answers_ids,
            "labels": self.labels[idx]
        }