
from pathlib import Path
from ..src.adats.calibration import AdaptiveTemperatureScaling

import pandas as pd
import torch

from .adats_cal import MAP_CALIBRATORS
from copy import deepcopy

def main(
    checkpoint_path: str,
    method: str,
    predict_logits: str,
    predict_labels: str,
    output_dir: str = 'output',
):
    # Load predict data
    df_predict_embeddings = pd.read_csv(predict_embeddings, index_col=0, header=None)
    predict_embeddings = torch.from_numpy(df_predict_embeddings.values).float()
    df_predict_logits = pd.read_csv(predict_logits, index_col=0, header=None)
    predict_logits = torch.log_softmax(torch.from_numpy(df_predict_logits.values).float(), dim=1)
    df_predict_labels = pd.read_csv(predict_labels, index_col=0, header=None)
    predict_labels = torch.from_numpy(df_predict_labels.values.flatten()).long()

    # Load model
    vae_params = deepcopy(MAP_CALIBRATORS[method.split('adats_')[-1]])
    model = AdaptiveTemperatureScaling(vae_params=vae_params)
    state_dict = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(state_dict['state_dict'])

    # Predict
    with torch.no_grad():
        cal_logits = model(predict_logits, predict_embeddings).cpu().numpy()

    # Save
    output_dir = Path(output_dir)
    pd.DataFrame(cal_logits, index=df_predict_labels.index).to_csv(output_dir / 'logits.csv', index=True, header=False)
    df_predict_labels.to_csv(output_dir / 'labels.csv', index=True, header=False)

    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)