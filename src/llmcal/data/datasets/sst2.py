import os
from datasets import load_dataset, load_from_disk
import numpy as np


def load_sst2():
    # data = load_dataset("nyu-mll/glue", "sst2")
    data = load_from_disk("data/sst2")
    datadict = {
        "train": data["train"],
        "test": data["validation"]
    }
    return datadict