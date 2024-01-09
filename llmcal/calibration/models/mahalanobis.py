
import numpy as np
import torch
from torch import nn

from .base import BaseCalibrator
from ..optim import SGDMixin

class MahalanobisCalibrator(BaseCalibrator, SGDMixin):

    def __init__(self, num_features: int, num_classes: int, eps: float = 1e-3, random_state: int = None):
        super().__init__(num_features=num_features, num_classes=num_classes, random_state=random_state)
        self.eps = eps
        self.additional_arguments = {
            "eps": eps
        }

        # Parameters
        self.means = nn.Parameter(torch.ones(num_classes, num_features))


    def init_parameters(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels

        # Compute the mean and covariance for each class
        for c in range(self.num_classes):
            features_c = train_features[train_labels == c]
            self.means.data[c] = features_c.mean(dim=0)

    def compute_sigma(self):
        # Compute the covariance for each class
        sigma = torch.zeros(self.num_classes, self.num_features, self.num_features)
        for c in range(self.num_classes):
            features_c = self.train_features[self.train_labels == c]
            centered_features = features_c - self.means[c]
            sigma[c] = torch.matmul(centered_features.T, centered_features) / (features_c.shape[0] - 1) + self.eps * torch.eye(self.num_features, device=features.device)
        return sigma

    def forward(self, features):
        # Compute the Mahalanobis distance
        features_centered = features.unsqueeze(1) - self.means
        sigma = self.compute_sigma() # Compute the sigma with the training data
        inv_sigma = torch.cholesky_inverse(sigma)
        inv_sigma_features = torch.matmul(inv_sigma.unsqueeze(0), features_centered.unsqueeze(3)).squeeze(3)
        mahalanobis = torch.sum(features_centered * inv_sigma_features, dim=2)
        return -mahalanobis

        
class QDACalibrator(BaseCalibrator):
    
    def __init__(self, num_features: int, num_classes: int, eps: float = 1e-3, random_state: int = None):
        super().__init__(num_features=num_features, num_classes=num_classes, random_state=random_state)
        self.eps = eps
        self.additional_arguments = {
            "eps": eps
        }

        # Parameters
        self.means = nn.Parameter(torch.zeros(num_classes,num_features), requires_grad=False)
        self.covariances = nn.Parameter(torch.zeros(num_classes,num_features,num_features), requires_grad=False)
        self.priors = nn.Parameter(torch.zeros(num_classes), requires_grad=False)

    def forward(self, features):
        features_centered = features.unsqueeze(1) - self.means
        inv_cov = torch.cholesky_inverse(self.covariances)
        inv_cov_features = torch.matmul(inv_cov.unsqueeze(0), features_centered.unsqueeze(3)).squeeze(3)
        mahalanobis = torch.sum(features_centered * inv_cov_features, dim=2)
        logits = -(np.log((2 * np.pi)**self.num_classes) + torch.log(torch.linalg.det(self.covariances))) / 2 - mahalanobis + torch.log(self.priors)
        return logits

    def fit(
        self, 
        train_features, 
        train_labels, 
        validation_features=None, 
        validation_labels=None, 
        batch_size=32, 
        accelerator=None, 
        devices=None
    ):
        num_samples = train_features.shape[0]
        for class_idx in range(self.num_classes):
            class_features = train_features[train_labels == class_idx]
            self.means.data[class_idx] = torch.mean(class_features, dim=0)
            self.covariances.data[class_idx] = torch.cov(class_features.T)
            self.priors.data[class_idx] = torch.tensor(class_features.shape[0] / num_samples)
        return self


class LDACalibrator(BaseCalibrator):

    def __init__(self, num_features: int, num_classes: int, eps: float = 1e-3, random_state: int = None):
        super().__init__(num_features=num_features, num_classes=num_classes, random_state=random_state)
        self.eps = eps
        self.additional_arguments = {
            "eps": eps
        }

        # Parameters
        self.means = nn.Parameter(torch.zeros(num_classes,num_features), requires_grad=False)
        self.covariance = nn.Parameter(torch.eye(num_features), requires_grad=False)
        self.priors = nn.Parameter(torch.zeros(num_classes), requires_grad=False)
    
    def forward(self, features):
        logits = features @ self._A.T + self._b
        return logits

    def fit(
        self, 
        train_features, 
        train_labels, 
        validation_features=None, 
        validation_labels=None, 
        batch_size=32, 
        accelerator=None, 
        devices=None
    ):
        num_samples = train_features.shape[0]
        class_covariances = []
        for class_idx in range(self.num_classes):
            class_features = train_features[train_labels == class_idx]
            self.means.data[class_idx] = torch.mean(class_features, dim=0)
            class_covariances.append(torch.cov(class_features.T) * (class_features.shape[0] - 1))
            self.priors.data[class_idx] = torch.tensor(class_features.shape[0] / num_samples)
        self.covariance.data = torch.mean(torch.stack(class_covariances), dim=0) + self.eps * torch.eye(self.num_features)

        inv_covariance = torch.cholesky_inverse(self.covariance)
        self._A = 2 * self.means @ inv_covariance
        self._b = -torch.sum((self.means @ inv_covariance) * self.means, dim=1) + torch.log(self.priors)
        return self



if __name__ == "__main__":
    # model = MahalanobisCalibrator(2, 4)
    # model.init_parameters(
    #     torch.randn(100, 2),
    #     torch.randint(0, 4, (100,))
    # )

    # model = QDACalibrator(2, 4)
    model = LDACalibrator(2, 4)
    model.fit(
        torch.randn(100, 2),
        torch.randint(0, 4, (100,))
    )

    features = torch.randn(3, 2)
    print(features)
    out = model(features)
    print(out)
          
