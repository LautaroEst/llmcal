

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
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--batch_size', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
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
    logger = getLogger()

    # Set random state
    rs = np.random.RandomState(args.random_state)

    # Load features and labels
    train_features = np.load(args.train_features) # Train
    train_labels = train_features.argmax(axis=-1) if "logits.npy" in args.train_labels else np.load(args.train_labels)
    if args.subsample_train is not None:
        all_idx = np.arange(train_features.shape[0])
        if args.subsample_train > train_features.shape[0]:
            args.subsample_train = train_features.shape[0]
            logger.warning(f"Subsample train ({args.subsample_train}) is larger than the number of training samples ({train_features.shape[0]}).")
        train_idx = rs.choice(all_idx, size=args.subsample_train, replace=False)
        non_train_idx = np.setdiff1d(all_idx, train_idx)
        non_train_features = train_features[non_train_idx]
        non_train_labels = train_labels[non_train_idx]
        train_features = train_features[train_idx]
        train_labels = train_labels[train_idx]
    else:
        non_train_features = np.array([[]])
        non_train_labels = np.array([])

    eval_features = np.load(args.eval_features) # Evaluation
    eval_labels = np.load(args.eval_labels)
    if args.subsample_eval is not None:
        if args.subsample_eval > eval_features.shape[0]:
            args.subsample_eval = eval_features.shape[0]
            logger.warning(f"Subsample evalaution ({args.subsample_eval}) is larger than the number of evaluation samples ({eval_features.shape[0]}).")
        eval_idx = rs.choice(np.arange(eval_features.shape[0]), size=args.subsample_eval, replace=False)
        eval_features = eval_features[eval_idx]
        eval_labels = eval_labels[eval_idx]

    # Create train - validation split
    if args.validation_samples < 0 or args.validation_samples > non_train_features.shape[0]:
        raise ValueError(f"Invalid number of validation samples: {args.validation_samples}. Must be in [0, {non_train_features.shape[0]}].")
    elif args.validation_samples == 0:
        validation_features = None
        validation_labels = None
    else:
        all_idx = np.arange(non_train_features.shape[0])
        validation_idx = rs.choice(all_idx, size=args.validation_samples, replace=False)
        validation_features = non_train_features[validation_idx]
        validation_labels = non_train_labels[validation_idx]

    method_id = get_method_id(args)

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
    fit_calibrator_args = {
        "train_features": torch.from_numpy(train_features),
        "train_labels": torch.from_numpy(train_labels),
        "validation_features": torch.from_numpy(validation_features) if validation_features is not None else None,
        "validation_labels": torch.from_numpy(validation_labels) if validation_labels is not None else None,
        
        "batch_size": int(args.batch_size) if args.batch_size != "None" and args.batch_size is not None else None,
        "accelerator": args.accelerator,
        "devices": args.devices,
    }
    if args.method == "affine":
        fit_calibrator_args["model_checkpoint_dir"] = os.path.join(args.output_dir, method_id)
        fit_calibrator_args["learning_rate"] = args.learning_rate
        fit_calibrator_args["max_ls"] = args.max_ls
        fit_calibrator_args["max_epochs"] = args.max_epochs
        fit_calibrator_args["tolerance"] = args.tolerance
    elif args.method == "prior_adaptation":
        pass
    elif args.method in ["qda", "lda"]:
        pass
    elif args.method == "mahalanobis":
        fit_calibrator_args["model_checkpoint_dir"] = os.path.join(args.output_dir, method_id)
        fit_calibrator_args["optimizer"] = args.optimizer
        fit_calibrator_args["max_epochs"] = args.max_epochs
        fit_calibrator_args["learning_rate"] = args.learning_rate
        fit_calibrator_args["weight_decay"] = args.weight_decay
        fit_calibrator_args["patience"] = args.patience
    else:
        raise ValueError(f"Calibration method {args.method} not supported.")

    # Create output dir
    os.makedirs(os.path.join(args.output_dir, method_id), exist_ok=True)

    # Obtain calibrated posteriors
    calibrated_eval_posteriors, history = obtain_calibrated_posteriors(
        eval_features, 
        args.num_classes, 
        args.method,
        feature_map=args.feature_map,
        init_calibrator_args=init_calibrator_args,
        fit_calibrator_args=fit_calibrator_args,
    )

    # Save calibrated posteriors
    np.save(
        os.path.join(args.output_dir, method_id, "calibrated_posteriors.npy"),
        calibrated_eval_posteriors
    )

    # Save args and history
    with open(os.path.join(args.output_dir, method_id, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4, separators=(',', ': '))
    with open(os.path.join(args.output_dir, method_id, "history.json"), "w") as f:
        json.dump(history, f, indent=4, separators=(',', ': '))


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
    eval_features, 
    num_classes, 
    method, 
    feature_map=None,
    init_calibrator_args={},
    fit_calibrator_args={},
):
    
    # Convert to tensors
    eval_features = torch.from_numpy(eval_features)
    
    # Model
    train_samples, num_features = fit_calibrator_args['train_features'].shape
    if method == "affine":
        model = AffineCalibratorWithFeatureMap(feature_map, num_features=num_features, num_classes=num_classes, **init_calibrator_args)
    elif method == "prior_adaptation":
        model = PriorsAdaptator(num_features, **init_calibrator_args)
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
    else:
        raise ValueError(f"Calibration method {method} not supported.")

    # Fit calibrator    
    print(f"Training calibration model...")
    print(f"Model: {repr(model)}")
    print(f"Number of samples: {train_samples}")
    model.fit(**fit_calibrator_args)
    print("Done.")
    print()
    
    # Evaluate calibrator
    calibrated_posteriors = model.calibrate(
        eval_features, 
        batch_size=fit_calibrator_args["batch_size"], 
        accelerator=fit_calibrator_args["accelerator"], 
        devices=fit_calibrator_args["devices"]
    )
    calibrated_posteriors = calibrated_posteriors.cpu().numpy()
    
    history = {
        "train": model.train_loss_history if hasattr(model, "train_loss_history") else None,
        "validation": model.val_loss_history if hasattr(model, "val_loss_history") else None,
    }
    return calibrated_posteriors, history


if __name__ == "__main__":
    main()