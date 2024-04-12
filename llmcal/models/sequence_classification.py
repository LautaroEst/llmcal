

from typing import Literal

import torch
from torch import nn
import lightning as L


class SequenceClassificationModel(L.LightningModule):

    def __init__(
        self, 
        language_model,
    ):
        super().__init__()
        self.model = language_model
