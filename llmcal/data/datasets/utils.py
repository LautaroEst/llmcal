
import numpy as np

def sample_and_shuffle(dataset, num_samples, random_state):
    rs = np.random.RandomState(random_state)
    if num_samples is None:
        num_samples = len(dataset)
    idx = rs.choice(len(dataset), num_samples, replace=False).tolist()
    dataset = dataset.select(idx)
    return dataset

