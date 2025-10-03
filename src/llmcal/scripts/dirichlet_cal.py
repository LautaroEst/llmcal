
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
from ..src.dirichlet import DirichletCalibrator, FixedDiagonalDirichletCalibrator

from ..src.loggers import TBLogger, CSVLogger

warnings.filterwarnings("ignore", category=UserWarning, message=".*Experiment logs directory outputs*")


MAP_CALIBRATORS = {
    'dirichlet_fixed_diag': FixedDiagonalDirichletCalibrator(),

}


def main(
    output_dir: str = 'output',
    log_dir: str = 'output/logs',
    checkpoint_dir: str = 'output/checkpoints',
    train_logits: str = 'logits.csv',
    train_labels: str = 'labels.csv',
    predict_logits: str = 'logits.csv',
    predict_labels: str = 'labels.csv',
    method: Literal["dirichlet_fixed_diag"] = "dirichlet_fixed_diag",
    seed: int = 0,
):
    torch.set_float32_matmul_precision("high")
    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)

    # Load train data
    train_probas = softmax(pd.read_csv(train_logits, index_col=0, header=None).values, axis=1)
    train_labels = pd.read_csv(train_labels, index_col=0, header=None).values.flatten().astype(int)
    
    # Load predict data
    df_predict_logits = pd.read_csv(predict_logits, index_col=0, header=None)
    predict_probas = softmax(df_predict_logits.values, axis=1)
    df_predict_labels = pd.read_csv(predict_labels, index_col=0, header=None)
    predict_labels = df_predict_labels.values.flatten().astype(int)

    # num_classes = train_logits.shape[1]
    # model = AffineCalibrator(method=method, num_classes=num_classes)
    model = clone(MAP_CALIBRATORS[method])
    # state = fit(model, train_logits, train_labels, log_dir, tolerance, learning_rate, max_ls)
    model.fit(train_probas, train_labels)

    # torch.save(state, checkpoint_dir / 'state.ckpt')
    # model.load_state_dict(state['best_model'])

    # Predict
    cal_logits = np.log(model.predict_proba(predict_probas))

    # Save results
    pd.DataFrame(cal_logits, index=df_predict_logits.index).to_csv(output_dir / 'logits.csv', index=True, header=False)
    df_predict_labels.to_csv(output_dir / 'labels.csv', index=True, header=False)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)