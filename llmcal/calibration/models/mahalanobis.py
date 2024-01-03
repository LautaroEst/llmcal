
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaseCalibrator
from ..optim import SGDMixin


def square_mahalanobis_distance(X, mu, sigma):
    """
    Compute the square Mahalanobis distance between the matrix X of samples and a
    multivariate distribution with mean mu and covariance matrix sigma.

    Parameters
    ----------
    X : torch.Tensor(shape=(num_samples, num_features))
        Matrix of samples.
    mu : torch.Tensor(shape=(num_features,))
        Mean of the multivariate distribution.
    sigma : torch.Tensor(shape=(num_features, num_features))
        Covariance matrix of the multivariate distribution.

    Returns
    -------
    torch.Tensor(shape=(num_samples,))
        Square Mahalanobis distance between the samples and the multivariate distribution.
    """
    x_tilde = (X - mu)
    # inv_sigma = torch.cholesky_inverse(sigma)
    inv_sigma = torch.linalg.inv(sigma)
    return torch.sum(x_tilde * (x_tilde @ inv_sigma.T), dim=1)


class QDACalibrator(BaseCalibrator):
    
    def __init__(self, num_features, num_classes, random_state=None):

        super().__init__(num_features=num_features, num_classes=num_classes, random_state=random_state)
        self.means = nn.ParameterList(
            [nn.Parameter(torch.zeros(num_features)) for _ in range(num_classes)]
        )
        self.means.requires_grad_(False)
        
        self.covariances = nn.ParameterList(
            [nn.Parameter(torch.eye(num_features)) for _ in range(num_classes)]
        )
        self.covariances.requires_grad_(False)
        
        self.priors = nn.Parameter(torch.zeros(num_classes))
        self.priors.requires_grad_(False)

    def forward(self, features):
        logits = torch.stack([
            -(np.log((2 * np.pi)**self.num_classes) + torch.log(torch.linalg.det(cov))) / 2 - square_mahalanobis_distance(features,mu=mean,sigma=cov) + torch.log(prior) 
            for mean, cov, prior in zip(self.means, self.covariances, self.priors)
        ], dim=0).T
        return logits

    def fit(self, train_features, train_labels, validation_features=None, validation_labels=None, batch_size=32, accelerator=None, devices=None):
        num_samples = train_features.shape[0]
        for class_idx in range(self.num_classes):
            class_features = train_features[train_labels == class_idx]
            self.means[class_idx].data = torch.mean(class_features, dim=0)
            self.covariances[class_idx].data = torch.cov(class_features.T)
            self.priors.data[class_idx] = torch.tensor(class_features.shape[0] / num_samples)
        return self

class LDACalibrator(BaseCalibrator):

    def __init__(self, num_features, num_classes, random_state=None):
        
        super().__init__(num_features=num_features, num_classes=num_classes, random_state=random_state)
        self.means = nn.Parameter(torch.zeros(num_classes,num_features))
        self.means.requires_grad_(False)

        self.covariance = nn.Parameter(torch.eye(num_features))
        self.covariance.requires_grad_(False)

        self.priors = nn.Parameter(torch.zeros(num_classes))
        self.priors.requires_grad_(False)
    
    def forward(self, features):
        inv_covariance = self.inv_covariance
        A = 2 * self.means @ inv_covariance
        b = -torch.sum((self.means @ inv_covariance) * self.means, dim=1) + torch.log(self.priors)
        logits = features @ A.T + b
        return logits

    @property
    def inv_covariance(self):
        s, P = torch.linalg.eigh(self.covariance)
        return P * (1 / s) @ P.T

    def fit(self, train_features, train_labels, validation_features=None, validation_labels=None, batch_size=32, accelerator=None, devices=None):
        num_samples = train_features.shape[0]
        class_covariances = []
        for class_idx in range(self.num_classes):
            class_features = train_features[train_labels == class_idx]
            self.means.data[class_idx] = torch.mean(class_features, dim=0)
            class_covariances.append(torch.cov(class_features.T) * (class_features.shape[0] - 1))
            self.priors.data[class_idx] = torch.tensor(class_features.shape[0] / num_samples)
        self.covariance.data = torch.mean(torch.stack(class_covariances), dim=0)
        return self


class MahalanobisCalibrator(BaseCalibrator, SGDMixin):
    
    def __init__(self, num_features, num_classes, random_state=None):
        
        super().__init__(num_features=num_features, num_classes=num_classes, random_state=random_state)
        self.means = nn.ParameterList(
            [nn.Parameter(torch.randn(num_features,generator=self.generator)) for _ in range(num_classes)]
        )

    def compute_covariance(self, class_idx):
        mu = self.means[class_idx]
        features_centered = self.train_features.to(device=mu.device) - mu
        cov = features_centered.T @ features_centered / (self.train_features.shape[0] - 1)
        return cov

    def compute_distances_matrix(self, features):
        return torch.stack([
            square_mahalanobis_distance(
                features, 
                mu=self.means[class_idx], 
                sigma=self.compute_covariance(class_idx)
            ) for class_idx in range(self.num_classes)
        ], dim=1)

    def fit(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        **kwargs
    ):
        # Save training data
        self.train_features = train_features
        self.train_labels = train_labels

        # Init means
        for class_idx in range(self.num_classes):
            class_features = train_features[train_labels == class_idx]
            mean = []
            for batch in torch.split(class_features, kwargs["batch_size"]):
                mean.append(torch.sum(batch, dim=0))
            self.means[class_idx].data = torch.sum(torch.stack(mean), dim=0) / class_features.shape[0]
        super().fit(train_features, train_labels, **kwargs)
        return self

    # def compute_distances_matrix(self, features):
    #     distances = []
    #     for class_idx in range(self.num_classes):
    #         mu = self.means[class_idx]
    #         _, S, Vt = torch.linalg.svd(self.train_features - mu, full_matrices=False)
    #         x_centered = features - mu
    #         x_tilde = x_centered @ (Vt.T * (1 / S))
    #         distance = torch.sum(x_tilde * x_tilde, dim=1)
    #         distances.append(distance)
    #     return torch.stack(distances, dim=1)
        
    def loss(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        return loss
        
    def forward(self, features):
        distances = self.compute_distances_matrix(features)
        logits = -distances
        return logits
    
class MahalanobisCalibratorQR(BaseCalibrator, SGDMixin):
    
    def __init__(self, num_features, num_classes, random_state=None):
        
        super().__init__(num_features=num_features, num_classes=num_classes, random_state=random_state)
        self.means = nn.ParameterList(
            [nn.Parameter(torch.randn(num_features,generator=self.generator)) for _ in range(num_classes)]
        )
        self.A = nn.ParameterList(
            [nn.Parameter(torch.randn(num_features, num_features)) for _ in range(num_classes)]
        )
        self.S = nn.ParameterList(
            [nn.Parameter(torch.randn(num_features)+1) for _ in range(num_classes)]
        )

    def compute_distances_matrix(self, features):
        distances = []
        for class_idx in range(self.num_classes):
            Q, _ = torch.linalg.qr(self.A[class_idx])
            S = self.S[class_idx]
            x_tilde = (features - self.means[class_idx]) @ (Q / torch.abs(S))
            distance = torch.sum(x_tilde**2, dim=1)
            distances.append(distance)
        return torch.stack(distances, dim=1)

    def fit(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        **kwargs
    ):
        # Save training data
        self.train_features = train_features
        self.train_labels = train_labels

        # Init means
        for class_idx in range(self.num_classes):
            class_features = train_features[train_labels == class_idx]
            mean = []
            for batch in torch.split(class_features, kwargs["batch_size"]):
                mean.append(torch.sum(batch, dim=0))
            self.means[class_idx].data = torch.sum(torch.stack(mean), dim=0) / class_features.shape[0]
        super().fit(train_features, train_labels, **kwargs)
        return self

    # def compute_distances_matrix(self, features):
    #     distances = []
    #     for class_idx in range(self.num_classes):
    #         mu = self.means[class_idx]
    #         _, S, Vt = torch.linalg.svd(self.train_features - mu, full_matrices=False)
    #         x_centered = features - mu
    #         x_tilde = x_centered @ (Vt.T * (1 / S))
    #         distance = torch.sum(x_tilde * x_tilde, dim=1)
    #         distances.append(distance)
    #     return torch.stack(distances, dim=1)
        
    def loss(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        return loss
        
    def forward(self, features):
        distances = self.compute_distances_matrix(features)
        logits = -distances
        return logits
        
    
        