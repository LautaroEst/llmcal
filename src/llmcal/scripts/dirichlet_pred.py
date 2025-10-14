
from pathlib import Path
import pickle

import numpy as np
from ..src.adats.calibration import AdaptiveTemperatureScaling

import pandas as pd
import torch

from .adats_cal import MAP_CALIBRATORS
from copy import deepcopy
from scipy.special import softmax

def main(
    checkpoint_path: str,
    method: str,
    predict_logits: str,
    predict_labels: str,
    output_dir: str = 'output',
):
    # Load predict data
    df_predict_logits = pd.read_csv(predict_logits, index_col=0, header=None)
    predict_probas = softmax(df_predict_logits.values, axis=1)
    df_predict_labels = pd.read_csv(predict_labels, index_col=0, header=None)
    predict_labels = df_predict_labels.values.flatten().astype(int)

    # Load model
    with open(checkpoint_path, 'rb') as f:
        model = pickle.load(f)

    # Predict
    cal_logits = np.log(model.predict_proba(predict_probas))

    # Save
    output_dir = Path(output_dir)
    pd.DataFrame(cal_logits, index=df_predict_logits.index).to_csv(output_dir / 'logits.csv', index=True, header=False)
    df_predict_labels.to_csv(output_dir / 'labels.csv', index=True, header=False)

    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)