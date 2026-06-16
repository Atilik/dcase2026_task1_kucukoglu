# DCASE 2026 Task 1 Submission — Kucukoglu (NYU)

This repository contains our custom implementation and submissions for **DCASE 2026 Task 1: Heterogeneous Audio Classification**. 

We built upon the baseline framework and extended it significantly to create 4 different ensemble systems combining hierarchical attention-based models (HATR), multimodal representations, and end-to-end fine-tuning.

## System Highlights

Our approach features multiple advanced extensions to the baseline:
- **Multi-Encoder Embeddings:** We extracted and utilized representations from **LAION-CLAP (HTSAT-tiny)**, **ConvNeXt-tiny (465mAP)**, and **Whisper-large-v3** to capture a diverse set of audio and semantic features.
- **End-to-End Fine-Tuning:** In addition to frozen embeddings, we implemented and successfully fine-tuned the CLAP HTSAT backbone end-to-end alongside our classifier.
- **Augmentation Strategies:** We implemented MixUp (alpha=0.2), temporal cutmix, class-weighted loss, hierarchical loss penalties, and cross-fold data augmentation.
- **Ensembling:** Our best submission (Submission 1) is a 5-model ensemble combining models trained on different combinations of the aforementioned encoders, augmentations, and fine-tuning strategies.

## Submissions

All 4 final generated submission files (output CSVs and metadata YAMLs) are located in the `results/` folder:

1. **Submission 1 (NYU_Ens1):** 5-model multi-encoder ensemble (mixup03, hloss07, balanced, convnext, and clap_ft). **Best system (81.13% hF).**
2. **Submission 2 (NYU_Ens2):** 5-model CLAP + ConvNeXt ensemble.
3. **Submission 3 (NYU_Single):** Single HATR model utilizing cross-fold noise swapping.
4. **Submission 4 (NYU_3Mod):** Three-modality single model (CLAP Audio + CLAP Text + Whisper Audio).

## Repository Structure

We have organized the repository into clean, functional folders:

- **`results/`**: Contains the final output CSVs and `.meta.yaml` files for all 4 submissions.
- **`model_outputs/`**: Contains all the PyTorch checkpoints, data splits, and pre-computed test logits for each experimental model run.
- **`scripts/`**: Contains all supplementary python files including:
  - Data building & embedding extraction (`build_dataset.py`, `extract_*.py`)
  - Analysis & plotting (`analyze_classes.py`, `check_test_distribution.py`)
- **`configs/`**: Contains all the YAML configuration files for the different models.
- **`Root directory`**: Core models and inference engines (`models.py`, `train_test.py`, `eval_inference.py`, `ensemble_evaluate.py`, `compute_class_metrics.py`).

*(The original baseline README has been preserved as `README_baseline.md` for reference.)*

## Notes on Running

Because we have cleanly organized the repository for final submission, the directories for `model_outputs` and `configs` have been moved. If you attempt to re-run the `eval_inference.py` or `compute_class_metrics.py` scripts locally, you will need to update the hardcoded paths in the Python code to point inside the new `model_outputs/` directory.

## Acknowledgements

This work extends the DCASE 2026 baseline. Original framework by Panagiota Anastasopoulou (Music Technology Group, UPF).
