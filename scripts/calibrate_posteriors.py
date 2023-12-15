

import argparse

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_logits', type=str, required=True)
    parser.add_argument('--train_labels', type=str, required=True)
    parser.add_argument('--eval_logits', type=str, required=True)
    parser.add_argument('--eval_labels', type=str, required=True)
    parser.add_argument('--subsample_train', type=int, default=None)
    parser.add_argument('--subsample_eval', type=int, default=None)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--random_state', type=int, default=0)
    args = parser.parse_args()

    return args


def main():

    # Read command args
    args = parse_args()

    # Set random state
    rs = np.random.RandomState(args.random_state)

    # Load logits and labels
    train_logits = np.load(args.train_logits)
    train_labels = np.load(args.train_labels)
    eval_logits = np.load(args.eval_logits)
    eval_labels = np.load(args.eval_labels)

    # Subsample
    if args.subsample_train is not None:
        train_idx = rs.choice(np.arange(len(train_logits)), size=args.subsample_train, replace=False)
        train_logits = train_logits[train_idx]
        train_labels = train_labels[train_idx]
    if args.subsample_eval is not None:
        eval_idx = rs.choice(np.arange(len(eval_logits)), size=args.subsample_eval, replace=False)
        eval_logits = eval_logits[eval_idx]
        eval_labels = eval_labels[eval_idx]
    
    # Calibrate
    results = calibrate_logits(train_logits, train_labels, eval_logits, eval_labels, args.method)


def calibrate_logits(train_logits, train_labels, eval_logits, eval_labels, method):

    num_samples, num_classes = train_logits.shape

    train_logits = torch.from_numpy(train_logits)
    train_labels = torch.from_numpy(train_labels)
    eval_logits = torch.from_numpy(eval_logits)
    
    if method == "affine_bias_only":
        model = AffineCalibrator(num_classes, "bias only", "log-loss")
        model.train_calibrator(train_logits, train_labels)
        cal_logits = model.calibrate(eval_logits)
    elif method == "UCPA-naive":
        model = UCPACalibrator(num_classes, max_iters=1, tolerance=1e-6)
        model.train_calibrator(train_logits)
        cal_logits = model.calibrate(eval_logits)
    elif method == "SUCPA-naive":
        model = UCPACalibrator(num_classes, max_iters=1, tolerance=1e-6)
        priors = torch.bincount(train_labels, minlength=num_classes) / num_samples
        model.train_calibrator(train_logits, priors)
        cal_logits = model.calibrate(eval_logits)
    elif method == "UCPA":
        model = UCPACalibrator(num_classes, max_iters=20, tolerance=1e-6)
        model.train_calibrator(train_logits)
        cal_logits = model.calibrate(eval_logits)
    elif method == "SUCPA":
        model = UCPACalibrator(num_classes, max_iters=20, tolerance=1e-6)
        priors = torch.bincount(train_labels, minlength=num_classes) / num_samples
        model.train_calibrator(train_logits, priors)
        cal_logits = model.calibrate(eval_logits)
    else:
        raise ValueError(f"Calibration method {method} not supported.")
    
    cal_logits = cal_logits.numpy()
    return cal_logits     