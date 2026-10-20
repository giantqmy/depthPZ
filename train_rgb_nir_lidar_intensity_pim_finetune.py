"""PIM intensity control for the four-encoder RGB/NIR stereo experiment.

Replace the auxiliary RGB/NIR DoLP inputs with the primary RGB S0 and NIR S0
images while keeping the PIM architecture, initialization, split, losses and
training loop identical to the DoLP PIM experiment.
"""

from train_rgb_nir_lidar_intensity_finetune import RGBNIRIntensityLiDARDataset
from train_rgb_nir_lidar_dolp_finetune import main as run_fusion_training


def main():
    run_fusion_training(
        auxiliary_input="intensity",
        dataset_class=RGBNIRIntensityLiDARDataset,
        fusion_type="pim",
    )


if __name__ == "__main__":
    main()
