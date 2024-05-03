from collections import defaultdict
import os
import torch
import re

def load_tensor_dataset(data_dir):
    datadict = defaultdict(dict)

    # Filter all files of the form {split}--{feature}--predict.pt
    for f in os.listdir(data_dir):
        m = re.match(r"(.*)--(.*)--predict.pt", f)
        if m is None:
            continue
        split, feature = m.groups()
        data = torch.load(os.path.join(data_dir, f))
        datadict[split][feature] = data
   
    return dict(datadict)

