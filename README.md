# Pneumonia detection from pediatric chest X-rays (AIE231 Neural Networks)

Numbers marked TBD need the final test run (M7, notebook 07).

## Problem
Binary classification of chest X-rays: NORMAL vs PNEUMONIA (positive class). Primary metric: recall on pneumonia, always read together with recall on normal, F1 and AUC.

## Data
Kaggle "Chest X-Ray Images (Pneumonia)" (paultimothymooney), CC BY 4.0, 5,856 images. 32 duplicate files removed (5,824 kept). Kaggle's own splits leak 170 patients across splits, so we re-split: patient-grouped, stratified 70/15/15 (train 4,031 / val 870 / test 923), seed 42. See data/README.md for download.

## Results (validation set unless stated)
| Model | Recall pneumonia | Recall normal | AUC |
|---|---|---|---|
| Majority class | 1.000 on pneumonia only, accuracy 0.730 | 0 | 0.5 |
| MLP baseline | 0.969 | 0.911 | 0.987 |
| Small CNN (from scratch) | 0.809 | 0.864 | 0.912 |
| VGG16 fine-tuned (block5) | 0.962 | 0.945 | 0.990 |
| Final VGG16, threshold 0.15, 3 seeds, TEST | TBD | TBD | TBD |
Ablations (2 seeds): pretraining and fine-tuning matter; augmentation, class weights and dropout show no clear gain beyond seed noise.

## Setup and run
pip install -r requirements.txt, then run notebooks 01 to 07 in order on a free Colab T4 GPU. Tests: `python -m pytest tests`. Demo: `python app/app.py` after M9 saves `models/vgg16_final.weights.h5`.

## Limitations
Course project, not a diagnostic tool. Pediatric data from one source, no external validation, one split, imbalance and a low threshold trade fewer missed cases for more false alarms.

## AI use
An AI assistant helped build the reference pipeline; team members must be able to explain every line. Details in the report appendix.

## Repository layout
- `src/` data.py (inventory, leak-free split, pipeline), models.py (CNN, VGG16), evaluate.py (metrics), train.py (experiment log), predict.py (inference)
- `notebooks/` 01 EDA, 02 preprocessing, 03 baselines, 04 models, 05 tuning, 06 ablations, 07 final test evaluation
- `configs/` one YAML per experiment; `experiments/` result CSVs; `tests/` pytest (`python -m pytest tests`); `app/app.py` Gradio demo; `report/` report and team division
- Team role: Eyad Hatem, Project Manager.
