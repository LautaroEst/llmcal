from datasets import load_dataset, concatenate_datasets
import numpy as np

VAL_HELD_OUT_SAMPLES = 12 * 77

def load_banking():
    data = load_dataset("PolyAI/banking77")
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["train"]), len(data["train"])+len(data["test"])))
    TEST_HELD_OUT_SAMPLES = len(data["test"])
    all_data = concatenate_datasets([data["train"], data["test"]])

    rs = np.random.RandomState(7348)
    idx = rs.permutation(len(all_data))
    test_heldout_idx = idx[:TEST_HELD_OUT_SAMPLES]
    val_heldout_idx = idx[TEST_HELD_OUT_SAMPLES:TEST_HELD_OUT_SAMPLES+VAL_HELD_OUT_SAMPLES]
    train_idx = idx[TEST_HELD_OUT_SAMPLES+VAL_HELD_OUT_SAMPLES:]

    datadict = {
        "train": all_data.select(train_idx),
        "validation_held_out": all_data.select(val_heldout_idx),
        "test_held_out": all_data.select(test_heldout_idx),
    }
    
    return datadict

