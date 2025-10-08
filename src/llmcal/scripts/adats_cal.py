
from copy import deepcopy
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lbfgs import LBFGS
import torch.nn.functional as F
from typing import Literal
from sklearn.base import clone
from scipy.special import softmax
from ..src.adats.calibration import AdaptiveTemperatureScaling

from ..src.loggers import TBLogger, CSVLogger

warnings.filterwarnings("ignore", category=UserWarning, message=".*Experiment logs directory outputs*")


MAP_CALIBRATORS = {
    'z_16': {'z_dim': 16},

}


def main(
    output_dir: str = 'output',
    log_dir: str = 'output/logs',
    checkpoint_dir: str = 'output/checkpoints',
    train_logits: str = 'logits.csv',
    train_embeddings: str = 'embeddings.csv',
    train_labels: str = 'labels.csv',
    predict_logits: str = 'logits.csv',
    predict_embeddings: str = 'embeddings.csv',
    predict_labels: str = 'labels.csv',
    method: Literal["dirichlet_fixed_diag"] = "dirichlet_fixed_diag",
    seed: int = 0,
):
    torch.set_float32_matmul_precision("high")
    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)

    # Load train data
    train_embeddings = torch.from_numpy(pd.read_csv(train_embeddings, index_col=0, header=None).values).float()
    train_logits = torch.log_softmax(torch.from_numpy(pd.read_csv(train_logits, index_col=0, header=None).values).float(), dim=1)
    train_labels = torch.from_numpy(pd.read_csv(train_labels, index_col=0, header=None).values.flatten()).long()
    
    # Load predict data
    df_predict_embeddings = pd.read_csv(predict_embeddings, index_col=0, header=None)
    predict_embeddings = torch.from_numpy(df_predict_embeddings.values).float()
    df_predict_logits = pd.read_csv(predict_logits, index_col=0, header=None)
    predict_logits = torch.log_softmax(torch.from_numpy(df_predict_logits.values).float(), dim=1)
    df_predict_labels = pd.read_csv(predict_labels, index_col=0, header=None)
    predict_labels = torch.from_numpy(df_predict_labels.values.flatten()).long()

    # num_classes = train_logits.shape[1]
    # model = AffineCalibrator(method=method, num_classes=num_classes)
    vae_params = deepcopy(MAP_CALIBRATORS[method])
    vae_params["in_dim"] = train_embeddings.shape[1]
    vae_params["num_classes"] = train_logits.shape[1]
    model = AdaptiveTemperatureScaling(vae_params=vae_params)
    # state = fit(model, train_logits, train_labels, log_dir, tolerance, learning_rate, max_ls)
    model.fit(train_embeddings, train_logits, train_labels)

    # torch.save(state, checkpoint_dir / 'state.ckpt')
    # model.load_state_dict(state['best_model'])

    # Predict
    with torch.no_grad():
        cal_logits = model(predict_logits, predict_embeddings).cpu().numpy()

    # Save results
    pd.DataFrame(cal_logits, index=df_predict_logits.index).to_csv(output_dir / 'logits.csv', index=True, header=False)
    df_predict_labels.to_csv(output_dir / 'labels.csv', index=True, header=False)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)