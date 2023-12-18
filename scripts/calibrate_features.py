

import argparse
import json
import os

import numpy as np
import torch

from llmcal.calibration import (
    AffineCalibrator, 
    PriorsAdaptator, 
    QDACalibrator,
    LDACalibrator,
    DiscriminativeMahalanobisCalibrator,
    apply_feature_map
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_features', type=str, required=True)
    parser.add_argument('--train_labels', type=str, required=True)
    parser.add_argument('--eval_features', type=str, required=True)
    parser.add_argument('--eval_labels', type=str, required=True)
    parser.add_argument('--subsample_train', type=str, default=None)
    parser.add_argument('--subsample_eval', type=str, default=None)
    parser.add_argument('--num_classes', type=int, required=True)
    
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument("--feature_map", type=str, default="None")
    parser.add_argument("--use_pseudo_labels", action="store_true")
    # Args for affine
    parser.add_argument('--alpha', type=str, default="matrix")
    parser.add_argument('--bias', action="store_true")
    parser.add_argument('--loss', type=str, default="log-loss")
    # Args for prior adaptation
    parser.add_argument('--priors', type=str, default=None)
    # Args for mahalanobis
    pass

    parser.add_argument('--accelerator', type=str, default="cpu")
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--tolerance', type=float, default=1e-4)

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--random_state', type=int, default=0)
    args = parser.parse_args()

    args.subsample_train = None if args.subsample_train == "None" else int(args.subsample_train)
    args.subsample_eval = None if args.subsample_eval == "None" else int(args.subsample_eval)

    return args


def main():

    # Read command args
    args = parse_args()

    # Set random state
    rs = np.random.RandomState(args.random_state)

    # Load features and labels
    train_features = np.load(args.train_features)
    eval_features = np.load(args.eval_features)
    train_labels = train_features.argmax(axis=-1) if "logits.npy" in args.train_labels else np.load(args.train_labels)
    eval_labels = eval_features.argmax(axis=-1) if "logits.npy" in args.eval_labels else np.load(args.eval_labels)

    # Subsample
    if args.subsample_train is not None:
        train_idx = rs.choice(np.arange(len(train_features)), size=args.subsample_train, replace=False)
        train_features = train_features[train_idx]
        train_labels = train_labels[train_idx]
    if args.subsample_eval is not None:
        eval_idx = rs.choice(np.arange(len(eval_features)), size=args.subsample_eval, replace=False)
        eval_features = eval_features[eval_idx]
        eval_labels = eval_labels[eval_idx]
    
    # Calibrate
    calibration_args = {
        "random_state": args.random_state
    }
    if args.method == "affine":
        calibration_args["alpha"] = args.alpha
        calibration_args["bias"] = args.bias
        calibration_args["loss"] = args.loss
    elif args.method == "prior_adaptation":
        calibration_args["priors"] = args.priors
    else:
        raise ValueError(f"Calibration method {args.method} not supported.")
    
    calibrated_eval_posteriors = obtain_calibrated_posteriors(
        train_features, 
        train_labels, 
        eval_features, 
        args.num_classes, 
        args.method,
        feature_map=args.feature_map if args.feature_map != "None" else None,
        accelerator=args.accelerator,
        num_devices=args.num_devices,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tolerance=args.tolerance,
        **calibration_args
    )

    # Save calibrated posteriors
    method_id = get_method_id(args)
    os.makedirs(os.path.join(args.output_dir, method_id), exist_ok=True)
    np.save(
        os.path.join(args.output_dir, method_id, "calibrated_posteriors.npy"),
        calibrated_eval_posteriors
    )
    with open(os.path.join(args.output_dir, method_id, "args.json"), "w") as f:
        json.dump(vars(args), f)


def get_method_id(args):
    if args.method == "affine":
        method_id = f"affine_alpha={args.alpha}_bias={args.bias}_loss={args.loss}"
    elif args.method == "prior_adaptation":
        method_id = f"prior_adaptation_priors={args.priors}"
    elif args.method == "mahalanobis":
        method_id = f"mahalanobis"
    return method_id


def obtain_calibrated_posteriors(
    train_features, 
    train_labels, 
    eval_features, 
    num_classes, 
    method, 
    feature_map=None,
    accelerator="cpu",
    num_devices=1,
    optimizer=None,
    batch_size=128,
    num_epochs=100,
    lr=1e-3,
    weight_decay=0,
    tolerance=1e-4,
    **kwargs
):
    
    train_features = apply_feature_map(torch.from_numpy(train_features), feature_map)
    train_labels = torch.from_numpy(train_labels)
    eval_features = apply_feature_map(torch.from_numpy(eval_features), feature_map)
    num_features = train_features.shape[1]

    if method == "affine":
        model = AffineCalibrator(num_features, num_classes, **kwargs)
    elif method == "prior_adaptation":
        model = PriorsAdaptator(num_classes, **kwargs)
    elif method == "qda":
        model = QDACalibrator(num_features, num_classes, **kwargs)
    elif method == "lda":
        model = LDACalibrator(num_features, num_classes, **kwargs)
    elif method == "mahalanobis":
        model = DiscriminativeMahalanobisCalibrator(num_features, num_classes, **kwargs)
    else:
        raise ValueError(f"Calibration method {method} not supported.")
    
    print(f"Training {method} calibration model with {len(train_labels)} samples...")
    model.fit(
        train_features, 
        train_labels,
        accelerator=accelerator,
        num_devices=num_devices,
        optimizer=optimizer,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        tolerance=tolerance
    )
    calibrated_posteriors = model.calibrate(eval_features)
    calibrated_posteriors = calibrated_posteriors.numpy()
    return calibrated_posteriors


if __name__ == "__main__":
    main()