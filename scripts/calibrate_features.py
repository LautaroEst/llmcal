

import argparse
import json
import os
from logging import getLogger

import numpy as np
import torch

from llmcal.calibration import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_features', type=str, required=True)
    parser.add_argument('--train_labels', type=str, required=True)
    parser.add_argument('--eval_features', type=str, default=None)
    parser.add_argument('--eval_labels', type=str, default=None)
    parser.add_argument('--subsample_train', type=str, default=None)
    parser.add_argument('--subsample_eval', type=str, default=None)
    parser.add_argument('--validation_samples', type=int, default=0)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--random_state', type=int, default=0)
    
    # Args for calibration method
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument("--feature_map", type=str, default="None")
    parser.add_argument('--alpha', type=str, default="matrix") # affine
    parser.add_argument('--bias', action="store_true") # affine
    parser.add_argument('--loss', type=str, default="log-loss") # affine
    parser.add_argument('--eps', type=str, default=None) # mahalanobis

    # Args for training calibrator
    parser.add_argument('--accelerator', type=str, default="cpu")
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--max_ls', type=int, default=40)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--batch_size', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()
    method_id = get_method_id(args)
    output_dir = os.path.join(args.output_dir, method_id)

    # Init calibrator args
    init_calibrator_args = {
        "random_state": args.random_state
    }
    if args.method == "affine":
        init_calibrator_args["alpha"] = args.alpha
        init_calibrator_args["bias"] = args.bias
        init_calibrator_args["loss"] = args.loss
    elif args.method in ["qda", "lda", "mahalanobis", "mahalanobis_qr", "mahalanobis_svd"]:
        init_calibrator_args["eps"] = args.eps
    else:
        raise ValueError(f"Calibration method {args.method} not supported.")

    # Fit calibrator args
    fit_calibrator_args = {
        "batch_size": int(args.batch_size) if args.batch_size != "None" and args.batch_size is not None else None,
        "accelerator": args.accelerator,
        "devices": args.devices,
    }
    if args.method == "affine":
        fit_calibrator_args["model_checkpoint_dir"] = output_dir
        fit_calibrator_args["learning_rate"] = args.learning_rate
        fit_calibrator_args["max_ls"] = args.max_ls
        fit_calibrator_args["max_epochs"] = args.max_epochs
        fit_calibrator_args["tolerance"] = args.tolerance
        fit_calibrator_args["weight_decay"] = args.weight_decay
    elif args.method in ["qda", "lda"]:
        pass
    elif args.method in ["mahalanobis", "mahalanobis_qr", "mahalanobis_svd"]:
        fit_calibrator_args["model_checkpoint_dir"] = output_dir
        fit_calibrator_args["optimizer"] = args.optimizer
        fit_calibrator_args["max_epochs"] = args.max_epochs
        fit_calibrator_args["learning_rate"] = args.learning_rate
        fit_calibrator_args["weight_decay"] = args.weight_decay
        fit_calibrator_args["patience"] = args.patience
    else:
        raise ValueError(f"Calibration method {args.method} not supported.")

    return args, method_id, output_dir, init_calibrator_args, fit_calibrator_args


def get_method_id(args):
    if "logits.npy" in args.train_features:
        method_id = "logits" 
    elif "embeddings.npy" in args.train_features:
        method_id = "embeddings"
    else:
        raise ValueError("Feature type not supported.")

    if args.feature_map != "None":
        method_id += f"_ft_map={args.feature_map}"

    if args.method == "affine":
        method_id += f"_affine_alpha={args.alpha}_bias={args.bias}_loss={args.loss}"
    elif args.method == "prior_adaptation":
        method_id += f"_prior_adaptation_priors={args.priors}"
    elif args.method == "qda":
        method_id += f"_qda"
    elif args.method == "lda":
        method_id += f"_lda"
    elif args.method == "mahalanobis":
        method_id += f"_mahalanobis"
    elif args.method == "mahalanobis_qr":
        method_id += f"_mahalanobis_qr"
    elif args.method == "mahalanobis_svd":
        method_id += f"_mahalanobis_svd"
    else:
        raise ValueError(f"Calibration method {args.method} not supported.")
    return method_id

def create_splits(
    train_features,
    train_labels,
    subsample_train=None,
    validation_samples=0,
    eval_features=None,
    eval_labels=None,
    subsample_eval=None,
    random_state=0,
):
    rs = np.random.RandomState(random_state)
    logger = getLogger()

    # Create train set
    train_features = np.load(train_features)
    train_labels = train_features.argmax(axis=-1) if "logits.npy" in train_labels else np.load(train_labels)
    if subsample_train is not None:
        all_idx = np.arange(train_features.shape[0])
        if subsample_train > train_features.shape[0]:
            subsample_train = train_features.shape[0]
            logger.warning(f"Subsample train ({subsample_train}) is larger than the number of training samples ({train_features.shape[0]}). Setting subsample train to {train_features.shape[0]}.")
        train_idx = rs.choice(all_idx, size=subsample_train, replace=False)
        non_train_idx = np.setdiff1d(all_idx, train_idx)
        non_train_features = train_features[non_train_idx]
        non_train_labels = train_labels[non_train_idx]
        train_features = train_features[train_idx]
        train_labels = train_labels[train_idx]
    else:
        non_train_features = np.array([[]])
        non_train_labels = np.array([])

    # Create validation set
    if validation_samples < 0 or validation_samples > non_train_features.shape[0]:
        raise ValueError(f"Invalid number of validation samples: {validation_samples}. Must be in [0, {non_train_features.shape[0]}].")
    elif validation_samples == 0:
        validation_features = None
        validation_labels = None
    else:
        all_idx = np.arange(non_train_features.shape[0])
        validation_idx = rs.choice(all_idx, size=validation_samples, replace=False)
        validation_features = non_train_features[validation_idx]
        validation_labels = non_train_labels[validation_idx]

    # Create test set
    eval_features = np.load(eval_features) if eval_features is not None else None
    eval_labels = np.load(eval_labels) if eval_labels is not None else None
    if subsample_eval is not None:
        if subsample_eval > eval_features.shape[0]:
            subsample_eval = eval_features.shape[0]
            logger.warning(f"Subsample evalaution ({subsample_eval}) is larger than the number of evaluation samples ({eval_features.shape[0]}). Setting subsample evaluation to {eval_features.shape[0]}.")
        eval_idx = rs.choice(np.arange(eval_features.shape[0]), size=subsample_eval, replace=False)
        eval_features = eval_features[eval_idx]
        eval_labels = eval_labels[eval_idx]

    return {
        "train": {
            "features": train_features,
            "labels": train_labels,
        },
        "validation": {
            "features": validation_features,
            "labels": validation_labels,
        },
        "test": {
            "features": eval_features,
            "labels": eval_labels,
        },
    }


def train_calibrator(
    train_features,
    train_labels,
    validation_features=None,
    validation_labels=None,
    num_classes=2,
    method="affine",
    feature_map=None,
    init_calibrator_args={},
    fit_calibrator_args={},
):
    # Model
    train_samples, num_features = train_features.shape
    if method == "affine":
        model = AffineCalibratorWithFeatureMap(feature_map, num_features=num_features, num_classes=num_classes, **init_calibrator_args)
    elif method == "qda":
        if feature_map != "identity":
            raise ValueError(f"QDA calibration does not support feature maps.")
        model = QDACalibrator(num_features=num_features, num_classes=num_classes, **init_calibrator_args)
    elif method == "lda":
        if feature_map != "identity":
            raise ValueError(f"LDA calibration does not support feature maps.")
        model = LDACalibrator(num_features=num_features, num_classes=num_classes, **init_calibrator_args)
    elif method == "mahalanobis":
        if feature_map != "identity":
            raise ValueError(f"Mahalanobis calibration does not support feature maps.")
        model = MahalanobisCalibrator(num_features=num_features, num_classes=num_classes, **init_calibrator_args)
    elif method == "mahalanobis_qr":
        if feature_map != "identity":
            raise ValueError(f"Mahalanobis calibration does not support feature maps.")
        model = MahalanobisCalibratorQR(num_features=num_features, num_classes=num_classes, **init_calibrator_args)
    elif method == "mahalanobis_svd":
        if feature_map != "identity":
            raise ValueError(f"Mahalanobis calibration does not support feature maps.")
        model = MahalanobisCalibratorSVD(num_features=num_features, num_classes=num_classes, **init_calibrator_args)
    else:
        raise ValueError(f"Calibration method {method} not supported.")

    # Fit calibrator
    print(f"Training calibration model...")
    print(f"Model: {repr(model)}")
    print(f"Number of samples: {train_samples}")
    model.fit(
        train_features=torch.from_numpy(train_features),
        train_labels=torch.from_numpy(train_labels),
        validation_features=torch.from_numpy(validation_features) if validation_features is not None else None,
        validation_labels=torch.from_numpy(validation_labels) if validation_labels is not None else None,
        **fit_calibrator_args
    )
    print("Done.")
    print()
    return model


def main():

    # Read command args
    args, method_id, output_dir, init_calibrator_args, fit_calibrator_args = parse_args()
    os.makedirs(output_dir, exist_ok=True)

    # Create splits
    splits = create_splits(
        args.train_features,
        args.train_labels,
        int(args.subsample_train) if args.subsample_train != "None" and args.subsample_train is not None else None,
        args.validation_samples,
        args.eval_features,
        args.eval_labels,
        int(args.subsample_eval) if args.subsample_eval != "None" and args.subsample_eval is not None else None,
        args.random_state,
    )

    # Train calibrator
    model = train_calibrator(
        splits["train"]["features"],
        splits["train"]["labels"],
        splits["validation"]["features"],
        splits["validation"]["labels"],
        num_classes=args.num_classes,
        method=args.method,
        feature_map=args.feature_map,
        init_calibrator_args=init_calibrator_args,
        fit_calibrator_args=fit_calibrator_args,
    )

    # Predict on test set
    if splits["test"]["features"] is not None:
        calibrated_posteriors = model.calibrate(
            torch.from_numpy(splits["test"]["features"]),
            batch_size=fit_calibrator_args["batch_size"],
            accelerator=fit_calibrator_args["accelerator"],
            devices=fit_calibrator_args["devices"]
        )
        calibrated_posteriors = calibrated_posteriors.cpu().numpy()
        np.save(
            os.path.join(args.output_dir, method_id, "calibrated_posteriors.npy"),
            calibrated_posteriors
        )

    # Save args and history
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4, separators=(',', ': '))



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Exiting...")