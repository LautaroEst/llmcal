
import torch

def norm_cross_entropy(logits, labels):
    ce = torch.nn.functional.cross_entropy(logits, labels)
    priors = torch.bincount(labels, minlength=logits.size(1)).float() / labels.size(0)
    ce_priors = -torch.mean(torch.log(priors[labels]))
    return ce / ce_priors