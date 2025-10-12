import torch
from abc import ABC
from .metrics import ECELoss
from .net import CondVAE
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def test(model, loader, device):
    preds = []
    labels = []
    model.eval()
    with torch.no_grad():
        for image, label in loader:
            preds.append(model(image.to(device)).cpu())
            labels.append(label.cpu())
    preds = torch.cat(preds, dim=0)  
    labels = torch.cat(labels) 
    acc = torch.argmax(preds, dim=-1).eq(labels).float().mean()
    ece_loss = ECELoss()
    ece = ece_loss(preds, labels)
    return acc.item(), ece.item()


class AdaptiveTemperatureScaling(ABC, torch.nn.Module):
    def __init__(self, vae_params, device="cpu") -> None:
        super().__init__()
        self.vae = CondVAE(**vae_params, device=device)

    def forward(self, logits, features, *args, **kwargs):
        temp = self.vae.sample_t(features)
        return torch.log_softmax(logits / temp, dim=-1)

    def fit(self, train_features, train_logits, train_labels, device="cpu", *args, **kwargs):
        # agg features

        val_dataset = TensorDataset(train_features, train_logits, train_labels)
        feat_loader = DataLoader(val_dataset, batch_size=128, drop_last=False, shuffle=True)

        # now train the vae
        optim = torch.optim.Adam(self.vae.parameters(), lr=5e-4)
        print("Training AdaTS")
        for epoch in tqdm(range(100)):
            loss = 0.0
            for feat, logits, labels in feat_loader:
                feat = feat.to(device)
                logits = logits.to(device)
                labels = labels.to(device)
                this_loss = self.vae.elbo(feat, labels)
                this_loss += self.vae.t_ce(feat, logits, labels)
                loss += this_loss
                this_loss.backward()
                optim.step()
                optim.zero_grad()




        
        