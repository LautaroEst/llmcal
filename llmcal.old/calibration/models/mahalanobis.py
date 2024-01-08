
from logging import getLogger
import os
from typing import Literal, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as L
from torch.optim import Adam, SGD
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

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
        features_centered = self.train_features[self.train_labels == class_idx].to(device=mu.device) - mu
        cov = features_centered.T @ features_centered / (features_centered.shape[0] - 1)
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
    

class MahalanobisCalibratorSVD(BaseCalibrator, SGDMixin):
    
    def __init__(self, num_features, num_classes, random_state=None):
        
        super().__init__(num_features=num_features, num_classes=num_classes, random_state=random_state)
        self.means = nn.ParameterList(
            [nn.Parameter(torch.randn(num_features,generator=self.generator)) for _ in range(num_classes)]
        )

    def compute_distances_matrix(self, features):
        distances = []
        for class_idx in range(self.num_classes):
            U, S, Vt = torch.linalg.svd(features - self.means[class_idx], full_matrices=False)
            distance = torch.sum(U**2,dim=1)
            distances.append(distance)
        return torch.stack(distances, dim=1)
    
    def compute_inv_sigmas(self):
        sqrt_inv_sigmas = []
        for class_idx in range(self.num_classes):
            features_centered = self.train_features[self.train_labels == class_idx] - self.means[class_idx]
            _, S, Vt = torch.linalg.svd(features_centered, full_matrices=False)
            sqrt_inv_sigma = Vt.T / S
            sqrt_inv_sigmas.append(sqrt_inv_sigma * np.sqrt(features_centered.shape[0] - 1))
        return sqrt_inv_sigmas

    def fit(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        validation_features: torch.Tensor = None,
        validation_labels: torch.Tensor = None,
        optimizer: Literal["Adam", "SGD"] = "Adam",
        batch_size: Union[int,None] = None,
        max_epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 0,
        patience: int = 10,
        model_checkpoint_dir: str = None,
        **kwargs
    ):

        # Init means
        for class_idx in range(self.num_classes):
            class_features = train_features[train_labels == class_idx]
            mean = []
            for batch in torch.split(class_features, batch_size):
                mean.append(torch.sum(batch, dim=0))
            self.means[class_idx].data = torch.sum(torch.stack(mean), dim=0) / class_features.shape[0]

        logger = getLogger(__name__) # Initialize the logger

        # Check the inputs
        if (validation_features is None and validation_labels is not None) or (validation_features is not None and validation_labels is None):
            raise ValueError("Validation features and labels must be both None or both not None")
        elif validation_features is None or validation_labels is None:
            perform_validation = False
        else:
            perform_validation = True

        # Initialize the accelerator
        fabric = L.Fabric(**kwargs)
        fabric.launch()

        # Initialize the optimizer
        if optimizer == "Adam":
            optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "SGD":
            optimizer = SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")
        model, optimizer = fabric.setup(self, optimizer)

        # Initialize the dataloaders
        train_dataloader = self._prepare_dataloader(train_features, train_labels, fabric, batch_size=batch_size)
        if perform_validation:
            validation_dataloader = self._prepare_dataloader(validation_features, validation_labels, fabric, batch_size=batch_size)

        # Prepare history
        train_loss_history = []
        best_epoch_loss = train_loss = val_loss = float("inf")
        if perform_validation:
            val_loss_history = []

        # Start training
        p_counter = patience
        epochs_bar = tqdm(range(max_epochs), leave=False)
        for epoch in epochs_bar:

            if perform_validation:
                epochs_bar.set_description(f"Train Loss: {train_loss:.4f} / Validation Loss: {val_loss:.4f}")
            else:
                epochs_bar.set_description(f"Train Loss: {train_loss:.4f}")

            # Train
            model.train()
            train_loss = 0
            for batch_features, batch_labels in train_dataloader:
                optimizer.zero_grad()
                batch_logits = -self.compute_distances_matrix(batch_features)
                batch_loss = self.loss(batch_logits, batch_labels)
                train_loss += batch_loss.item() * batch_features.shape[0]
                fabric.backward(batch_loss)
                optimizer.step()
            train_loss /= len(train_dataloader.dataset)
            train_loss_history.append(train_loss)
            loss = train_loss

            # Validation
            if perform_validation:
                model.eval()
                self.sqrt_inv_sigmas = self.compute_inv_sigmas()
                val_loss = 0
                with torch.no_grad():
                    for batch_features, batch_labels in validation_dataloader:
                        batch_logits = model(batch_features)
                        val_loss += self.loss(batch_logits, batch_labels).item() * batch_features.shape[0]
                    val_loss /= len(validation_dataloader.dataset)
                val_loss_history.append(val_loss)
                loss = val_loss

            # Check for convergence
            if loss > best_epoch_loss:
                if p_counter == 0:
                    logger.info(f"Converged after {epoch} epochs")
                    break
                else:
                    p_counter -= 1
            else:
                p_counter = patience

            # Save the best model
            if loss < best_epoch_loss:
                torch.save(model.state_dict(), os.path.join(model_checkpoint_dir, "best_model_state_dict.pt"))
                best_epoch_loss = loss
            
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history if perform_validation else None
        if epoch == max_epochs - 1:
            logger.warning(f"WARNING: Calibration did not converge after {max_epochs} epochs")
        
        self.load_state_dict(torch.load(os.path.join(model_checkpoint_dir, "best_model_state_dict.pt")))
        self.sqrt_inv_sigmas = self.compute_inv_sigmas()
        return self
    
    def _prepare_dataloader(self, features, labels, fabric, batch_size=None):
        dataset = TensorDataset(features, labels)
        sampler = RandomSampler(
            dataset,
            replacement=False,
            num_samples=features.shape[0],
            generator=self.generator
        )

        if batch_size is None:
            batch_size = features.shape[0]
        dataloader = fabric.setup_dataloaders(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler
            ),
            use_distributed_sampler=True,
            move_to_device=True
        )
        return dataloader

    def loss(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        return loss
        
    def forward(self, features):
        distances = []
        for class_idx in range(self.num_classes):
            sqrt_inv_sigma = self.sqrt_inv_sigmas[class_idx]
            x_tilde = (features - self.means[class_idx]) @ sqrt_inv_sigma
            distance = torch.sum(x_tilde**2, dim=1)
            distances.append(distance)
        logits = -torch.stack(distances, dim=1)
        return logits

    
class MahalanobisCalibratorQR(BaseCalibrator, SGDMixin):
    
    def __init__(self, num_features, num_classes, random_state=None):
        
        super().__init__(num_features=num_features, num_classes=num_classes, random_state=random_state)
        self.means = nn.ParameterList(
            [nn.Parameter(torch.randn(num_features,generator=self.generator)) for _ in range(num_classes)]
        )
        self.A = nn.ParameterList(
            [nn.Parameter(torch.randn(num_features, num_features, generator=self.generator)) for _ in range(num_classes)]
        )
        self.S = nn.ParameterList(
            [nn.Parameter(torch.randn(num_features, generator=self.generator)+1) for _ in range(num_classes)]
        )

    @property
    def covariances(self):
        class _Covariances:
            def __init__(s):
                s.A = self.A
                s.S = self.S
            def __getindex__(s, idx):
                Q, _ = torch.linalg.qr(s.A[idx])
                S = s.S[idx]
                Q_tilde = Q / torch.abs(S)
                return Q_tilde @ Q_tilde.T
        return _Covariances()
        

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
        
    
        