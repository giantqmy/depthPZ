"""Train four-encoder RGB/NIR stereo with left/right wavelet PIM fusion.

Uses the same paired data, split, losses and training loop as the iAFF trainer.
PIM matches the core of PZDualPIMStage; a 512-to-256 projection adapts its two
outputs to each camera's stereo feature interface. Both PIMs start with new
weights. --from-scratch also initializes all encoders and stereo weights anew.
"""

from train_rgb_nir_lidar_dolp_finetune import (
    make_model as make_fusion_model,
    main as run_fusion_training,
)


def make_model(checkpoint, device, *, from_scratch=False):
    """Build a PIM model; checkpoint is a baseline stereo initialization only."""
    return make_fusion_model(checkpoint, device, from_scratch=from_scratch, fusion_type="pim")


def main():
    run_fusion_training(fusion_type="pim")


if __name__ == "__main__":
    main()
