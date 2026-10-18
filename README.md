# Vessel-Guided Curriculum Learning with Uncertainty Calibration for Long-Tailed Multi-Disease Retinal Fundus Classification

Multi-label retinal disease classification on the **RFMiD** dataset (46 disease labels). The pipeline combines vessel-prior guidance, curriculum learning, and calibrated uncertainty to handle the long-tailed label distribution typical of fundus datasets.

## Overview

| Pillar | What it does |
|---|---|
| **Vessel guidance** | A frozen HRM-UNet (pretrained on DRIVE) produces vessel prior maps, fused into a DenseNet-121 classifier through multi-scale, gamma-gated residual attention at three transition scales |
| **Curriculum learning** | Four-signal, percentile-ranked difficulty scoring with soft sigmoid sample weighting (floor 0.1) and deferred re-weighting (DRW) for the long tail |
| **Uncertainty calibration** | Temperature scaling, class-wise (vector) scaling, and MC-Dropout, evaluated with equal-mass ECE binning |
| **Long-tail handling** | Capped inverse-frequency `pos_weight`; tail mAP computed only on classes with enough test positives |

## Results

Ablation over 3 seeds (42, 1337, 2024), 25 evaluable classes (8 head / 9 mid / 8 tail; classes with fewer than 10 training positives are dropped).

| Condition | mAP | Macro AUC | ECE |
|---|---|---|---|
| A: Baseline | 0.549 | 0.934 | 0.014 |
| B: + Vessel guidance | **0.564** | 0.929 | 0.013 |
| C: + Curriculum | 0.529 | 0.883 | 0.035 |
| D: + Calibration (full) | 0.529 | 0.883 | **0.005** |

### Key findings

- **Vessel guidance** gives a consistent, modest improvement over the baseline.
- **Curriculum learning** is a reproducible negative result: it underperforms the baseline on all 3 seeds under both hard-exclusion and soft-weighting designs, although tail mAP did beat the baseline. A possible cause, not yet confirmed, is continuous rarity upweighting stacking with DRW and producing overconfident predictions on the sparsest classes.
- **Calibration:** temperature scaling worsened ECE in all 9 seed × condition runs, while class-wise scaling reduced ECE by 3–7×.
- **Caveat:** filtering to evaluable classes raised mAP by about 0.13 across all conditions. This is a measurement artifact of the changed class set, not a model improvement.

With only 3 seeds, these results show direction, not statistical significance.

## Repository structure

```
.
├── notebooks/
│   └── vessel_guided_curriculum_pipeline_v4.ipynb   # full pipeline (latest)
├── checkpoints/
│   └── hrm_unet_drive.pth                           # HRM-UNet weights (DRIVE)
├── data/                                            # RFMiD and DRIVE (not tracked)
└── README.md
```

Adjust paths to match your layout.

## Setup

The pipeline is built for **Google Colab (free tier, T4 GPU)**.

1. Open the notebook in Colab and select a GPU runtime.
2. Obtain the datasets:
   - [RFMiD](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid): classifier benchmark
   - [DRIVE](https://drive.grand-challenge.org/): vessel segmentation training
3. Place the `hrm_unet_drive.pth` checkpoint where the notebook expects it.
4. Run the notebook top to bottom.

## Implementation notes

- **Tiled vessel inference:** HRM-UNet (Mamba SS2D blocks) must use 64×64 patches with stride 32 and unfold/fold averaging, matching training-time sliding-window prediction.
- **Separate preprocessing:** the segmenter takes plain grayscale input with no CLAHE, kept apart from classifier preprocessing.
- **Auto-recovered hyperparameters:** segmenter hyperparameters are inferred from the checkpoint `state_dict` shapes.
- **Resumable caching:** vessel-map cache building writes sidecar JSON metadata and resumes with `mode="r+"`, so Colab disconnects don't lose progress.
- **AMP:** the loss is computed in fp32 for numerical stability.
- **Equal-mass ECE binning:** equal-width bins degenerate in sparse multi-label settings.
- **Long-tail baseline:** simple resampling is used as the base approach, as it has been reported to outperform more complex long-tail methods on RFMiD (HKGL).

## Design evolution

| Component | Earlier | Current |
|---|---|---|
| Backbone | ResNet50 | DenseNet-121 |
| Segmenter | Standard U-Net | HRM-UNet (Mamba SS2D, tiled inference) |
| Calibration | Dirichlet | Temperature + class-wise scaling + MC-Dropout |
| Curriculum | Hard exclusion | Soft sigmoid weighting (floor 0.1) |

## Roadmap

- [ ] Ablation without DRW to test the curriculum/DRW interaction hypothesis
- [ ] 5–10 seeds for significance testing
- [ ] Grad-CAM integration to visualize the effect of vessel guidance

## Citation

```bibtex
@misc{vessel_guided_curriculum,
  title  = {Vessel-Guided Curriculum Learning with Uncertainty Calibration
            for Long-Tailed Multi-Disease Retinal Fundus Classification},
  author = {K Dhiraj, M Vaishnavi and Dr Modigari Narendra},
  year   = {2026}
}
```

## License

Add a license (e.g. MIT) before publishing. Check the licenses of RFMiD, DRIVE, and the HRM-UNet code before redistributing any data or weights.
