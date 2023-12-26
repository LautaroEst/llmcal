
import torch
import numpy as np

from llmcal.calibration import AffineCalibrator, init_feature_map, CalibratorWithFeatureMap
from psrcal import AffineCalLogLoss

def main():

    num_classes = 2
    
    model_test = CalibratorWithFeatureMap(
        init_feature_map(num_classes,None),
        AffineCalibrator(num_classes, num_classes, alpha="scalar", bias=True, loss="log-loss", random_state=None)
    )
    model_test.calibrator.alpha.data = torch.tensor(1.0)
    model_test.calibrator.bias.data = torch.zeros(num_classes)

    model_ref = AffineCalLogLoss(num_classes, bias=True, scale=True, priors=None)

    # scores = torch.log_softmax(torch.randn(100, num_classes), dim=-1)
    # labels = torch.randint(0, num_classes, (100,))
    
    scores = torch.log_softmax(torch.from_numpy(np.load("results/run_dataset_on_model/gpt2-xl/glue/cola/validation/0_shot/logits.npy")), dim=-1)
    labels = torch.from_numpy(np.load("results/run_dataset_on_model/gpt2-xl/glue/cola/validation/0_shot/labels.npy"))

    model_test.fit(scores, labels, batch_size=64, max_ls=40, max_epochs=100, tolerance=1e-4)
    model_ref.train(scores, labels)

    print("Test model:")
    print(model_test.calibrator.alpha)
    print(model_test.calibrator.bias)
    print()
    print("Reference model:")
    print(model_ref.temp)
    print(model_ref.bias)



if __name__ == '__main__':
    main()