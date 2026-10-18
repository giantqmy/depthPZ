"""Intensity control for the four-encoder RGB/NIR DoLP fusion experiment.

Replace RGB DoLP with the same three-channel RGB S0 used by the primary branch,
and NIR DoLP with the single-channel NIR S0. Keep the architecture, pretrained
initialization, seed, split, losses and optimizer identical to the DoLP trainer.
"""

from train_rgb_nir_lidar_finetune import RGBNIRLiDARDataset
from train_rgb_nir_lidar_dolp_finetune import main as run_fusion_training, make_model


class RGBNIRIntensityLiDARDataset(RGBNIRLiDARDataset):
    """Reuse rectified S0 tensors, retaining the auxiliary 3/1 channel counts."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        inputs = sample["inputs"]
        # These are the existing model's auxiliary input slot names. Their
        # contents are intensity images, recorded explicitly in run_config.
        # Clone to keep the two branches' input tensors independent.
        inputs["img_ref_dolp"] = inputs["img_ref"].clone()
        inputs["img_comp_dolp"] = inputs["img_comp"][:, :1].clone()
        return sample


def main():
    run_fusion_training(auxiliary_input="intensity", dataset_class=RGBNIRIntensityLiDARDataset)


if __name__ == "__main__":
    main()
