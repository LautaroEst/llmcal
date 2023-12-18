
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaseCalibrator


def square_mahalanobis_distance(X,mu,sigma):
    x_tilde = (X - mu)
    return torch.sum(x_tilde * (x_tilde @ torch.linalg.inv(sigma)), dim=1)


class QDACalibrator(BaseCalibrator):
    
    def __init__(self, num_features, num_classes, random_state=None):
        super().__init__(num_features=num_features, num_classes=num_classes)
        self.means = nn.ModuleList(
            [nn.Parameter(torch.zeros(num_features)) for _ in range(num_classes)]
        )
        self.covariances = nn.ModuleList(
            [nn.Parameter(torch.eye(num_features)) for _ in range(num_classes)]
        )
        self.priors = nn.Parameter(torch.zeros(num_classes))
        self.random_state = random_state

    def forward(self, features):
        unnorm_posteriors = torch.stack([
            -(torch.log((2 * np.pi)**self.num_classes) + torch.log(torch.linalg.det(cov))) / 2 - square_mahalanobis_distance(features,mu=mean,sigma=cov) + torch.log(prior) 
            for mean, cov, prior in zip(self.means, self.covariances, self.priors)
        ], dim=0).T
        return torch.log_softmax(unnorm_posteriors, dim=1)

    def fit(self, features, labels, **kwargs):
        num_samples = features.shape[0]
        for class_idx in range(self.num_classes):
            class_features = features[labels == class_idx].view(-1,1)
            self.means[class_idx] = nn.Parameter(torch.mean(class_features, dim=0))
            self.covariances[class_idx] = nn.Parameter(torch.cov(class_features.T))
            self.priors[class_idx] = nn.Parameter(class_features.shape[0] / num_samples)
        return self

class LDACalibrator(BaseCalibrator):

    def __init__(self, num_features, num_classes, random_state=None):
        super().__init__(num_features=num_features, num_classes=num_classes)
        self.means = nn.ModuleList(
            [nn.Parameter(torch.zeros(num_features)) for _ in range(num_classes)]
        )
        self.covariance = nn.Parameter(torch.eye(num_features))
        self.priors = nn.Parameter(torch.zeros(num_classes))
        self.random_state = random_state
    
    def forward(self, features):
        inv_covariance = self.inv_covariance
        A = 2 * self.means @ inv_covariance
        b = -torch.sum(self.means * inv_covariance @ self.means.T, dim=1) + torch.log(self.priors)
        unnorm_posteriors = features @ A.T + b
        return torch.log_softmax(unnorm_posteriors, dim=1)

    @property
    def inv_covariance(self):
        s, P = torch.linalg.eigh(self.covariance)
        return P * (1 / s) @ P.T

    def fit(self, features, labels, **kwargs):
        num_samples = features.shape[0]
        for class_idx in range(self.num_classes):
            class_features = features[labels == class_idx].view(-1,1)
            self.means[class_idx] = nn.Parameter(torch.mean(class_features, dim=0))
            self.priors[class_idx] = nn.Parameter(class_features.shape[0] / num_samples)
        self.covariance = nn.Parameter(torch.cov(features.T))
        return self


class DiscriminativeMahalanobisCalibrator(BaseCalibrator):
    
    def __init__(self, num_features, num_classes, random_state=None):
        super().__init__(num_features=num_features, num_classes=num_classes)
        self.means = nn.ModuleList(
            [nn.Parameter(torch.zeros(num_features)) for _ in range(num_classes)]
        )
        self.random_state = random_state

    def compute_covariance(self, class_features, class_mean):
        x_tilde = (class_features - class_mean)
        return x_tilde.T @ x_tilde / (class_features.shape[0] - 1)

    def compute_distances_matrix(self, features):
        return torch.stack([
            square_mahalanobis_distance(
                features, 
                mu=self.means[class_idx], 
                sigma=self.compute_covariance(features, self.means[class_idx])
            )
            for class_idx in range(self.num_classes)
        ], dim=0).T
    
    def loss(self, features, labels):
        distances = self.compute_distances_matrix(features)
        loss = F.cross_entropy(-distances, labels)
        return loss
        
    def forward(self, features):
        distances = self.compute_distances_matrix(features)
        return torch.log_softmax(-distances, dim=1)
    
        


            

    
        