

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
    init_feature_map
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_features', type=str, required=True)
    parser.add_argument('--train_labels', type=str, required=True)
    parser.add_argument('--eval_features', type=str, required=True)
    parser.add_argument('--eval_labels', type=str, required=True)
    parser.add_argument('--subsample_train', type=str, default=None)
    parser.add_argument('--subsample_eval', type=str, default=None)
    parser.add_argument('--validation_samples', type=int, default=0)
    parser.add_argument('--num_classes', type=int, required=True)
    
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument("--feature_map", type=str, default="None")
    # Args for affine
    parser.add_argument('--alpha', type=str, default="matrix")
    parser.add_argument('--bias', action="store_true")
    parser.add_argument('--loss', type=str, default="log-loss")
    # Args for prior adaptation
    parser.add_argument('--priors', type=str, default=None)
    # Args for mahalanobis
    pass

    # Args for training calibrator
    parser.add_argument('--accelerator', type=str, default="cpu")
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--batch_size', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    parser.add_argument('--max_ls', type=int, default=40)

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
    train_features = np.load(args.train_features) # Train
    train_labels = train_features.argmax(axis=-1) if "logits.npy" in args.train_labels else np.load(args.train_labels)
    
    eval_features = np.load(args.eval_features) # Evaluation
    eval_labels = eval_features.argmax(axis=-1) if "logits.npy" in args.eval_labels else np.load(args.eval_labels)

    # Create train - validation split
    if args.validation_samples < 0 or args.validation_samples > len(train_features) - args.subsample_train:
        raise ValueError(f"Invalid number of validation samples: {args.validation_samples}. Must be in [0, {len(train_features) - args.subsample_train}].")
    elif args.validation_samples == 0:
        validation_features = train_features.copy()
        validation_labels = train_labels.copy()
    else:
        all_idx = np.arange(len(train_features))
        validation_idx = rs.choice(all_idx, size=args.validation_samples, replace=False)
        train_idx = np.setdiff1d(all_idx, validation_idx)
        validation_features = train_features[validation_idx]
        validation_labels = train_labels[validation_idx]
        train_features = train_features[train_idx]
        train_labels = train_labels[train_idx]

    # Subsample
    if args.subsample_train is not None:
        train_idx = rs.choice(np.arange(len(train_features)), size=args.subsample_train, replace=False)
        train_features = train_features[train_idx]
        train_labels = train_labels[train_idx]
    if args.subsample_eval is not None:
        eval_idx = rs.choice(np.arange(len(eval_features)), size=args.subsample_eval, replace=False)
        eval_features = eval_features[eval_idx]
        eval_labels = eval_labels[eval_idx]
    
    # Init calibrator args
    init_calibrator_args = {
        "random_state": args.random_state
    }
    if args.method == "affine":
        init_calibrator_args["alpha"] = args.alpha
        init_calibrator_args["bias"] = args.bias
        init_calibrator_args["loss"] = args.loss
    elif args.method == "prior_adaptation":
        init_calibrator_args["priors"] = args.priors
    elif args.method in ["qda", "lda", "mahalanobis"]:
        pass
    else:
        raise ValueError(f"Calibration method {args.method} not supported.")
    
    # Fit calibrator args
    if args.method == "affine":
        fit_calibrator_args = {
            "accelerator": args.accelerator,
            "num_devices": args.num_devices,
            "batch_size": int(args.batch_size) if args.batch_size != "None" and args.batch_size is not None else None,
            "max_ls": args.max_ls,
            "max_epochs": args.max_epochs,
            "tolerance": args.tolerance,
        }
    elif args.method == "prior_adaptation":
        fit_calibrator_args = {}
    elif args.method in ["qda", "lda"]:
        fit_calibrator_args = {}
    elif args.method == "mahalanobis":
        fit_calibrator_args = {
            "accelerator": args.accelerator,
            "num_devices": args.num_devices,
            "optimizer": args.optimizer,
            "batch_size": int(args.batch_size) if args.batch_size != "None" and args.batch_size is not None else None,
            "max_epochs": args.max_epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "tolerance": args.tolerance,
        }
    else:
        raise ValueError(f"Calibration method {args.method} not supported.")

    calibrated_eval_posteriors = obtain_calibrated_posteriors(
        train_features, 
        train_labels,
        validation_features,
        validation_labels,
        eval_features, 
        args.num_classes, 
        args.method,
        feature_map=args.feature_map if args.feature_map != "None" else None,
        init_calibrator_args=init_calibrator_args,
        fit_calibrator_args=fit_calibrator_args,
    )

    # Save calibrated posteriors
    method_id = get_method_id(args)
    os.makedirs(os.path.join(args.output_dir, method_id), exist_ok=True)
    np.save(
        os.path.join(args.output_dir, method_id, "calibrated_posteriors.npy"),
        calibrated_eval_posteriors
    )
    with open(os.path.join(args.output_dir, method_id, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4, separators=(',', ': '))


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
    else:
        raise ValueError(f"Calibration method {args.method} not supported.")
    return method_id


def obtain_calibrated_posteriors(
    train_features, 
    train_labels,
    validation_features,
    validation_labels,
    eval_features, 
    num_classes, 
    method, 
    feature_map=None,
    init_calibrator_args={},
    fit_calibrator_args={},
):
    
    # train_features = apply_feature_map(torch.from_numpy(train_features), feature_map)
    train_features = torch.from_numpy(train_features)
    train_labels = torch.from_numpy(train_labels)
    # validation_features = apply_feature_map(torch.from_numpy(validation_features), feature_map)
    validation_features = torch.from_numpy(validation_features)
    validation_labels = torch.from_numpy(validation_labels)
    # eval_features = apply_feature_map(torch.from_numpy(eval_features), feature_map)
    eval_features = torch.from_numpy(eval_features)
    
    # Feature map
    feature_map = init_feature_map(train_features.shape[1], feature_map)

    # Model
    if method == "affine":
        model = AffineCalibrator(feature_map.num_features, num_classes, **init_calibrator_args)
    elif method == "prior_adaptation":
        model = PriorsAdaptator(feature_map.num_features, **init_calibrator_args)
    elif method == "qda":
        model = QDACalibrator(feature_map.num_features, num_classes, **init_calibrator_args)
    elif method == "lda":
        model = LDACalibrator(feature_map.num_features, num_classes, **init_calibrator_args)
    elif method == "mahalanobis":
        model = DiscriminativeMahalanobisCalibrator(feature_map.num_features, num_classes, **init_calibrator_args)
    else:
        raise ValueError(f"Calibration method {method} not supported.")
    
    print(f"Training calibration model...")
    print(f"Model: {repr(model)}")
    print(f"Number of samples: {len(train_labels)}")
    model.fit(
        train_features, 
        train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        feature_map=feature_map,
        **fit_calibrator_args
    )
    print("Done.")
    print()
    calibrated_posteriors = model.calibrate(eval_features, batch_size=fit_calibrator_args["batch_size"])
    calibrated_posteriors = calibrated_posteriors.numpy()
    return calibrated_posteriors


if __name__ == "__main__":
    main()