# Pneumonia detection from pediatric chest X-rays

Course project for AIE231 Neural Networks, Alamein International University.

## What this project is
A neural-network system that looks at one chest X-ray of a child and predicts whether it shows pneumonia (NORMAL or PNEUMONIA). The intended user is a clinician or screening nurse who wants a second opinion that does not miss cases. A missed pneumonia case is the costly error, so the main metric is recall on pneumonia, always read together with recall on normal.

## Dataset
Kaggle "Chest X-Ray Images (Pneumonia)" (paultimothymooney), licence CC BY 4.0. The original data are by Kermany et al. (2018). It holds 5,856 pediatric chest X-rays from one hospital system, about 73% of them pneumonia.

## Approach
1. Understand the data: class balance, duplicates, patient overlap, image sizes and pixel statistics.
2. Prepare the data without leakage: remove duplicate files, build our own patient-grouped, stratified 70/15/15 train/validation/test split, handle class imbalance with class weights, and augment training images only.
3. Compare simple baselines with neural networks: a small CNN trained from scratch and a pretrained VGG16 fine-tuned on our data.
4. Tune and run ablations using the validation set only, then evaluate on the test set once at the end.
5. Analyse the errors and build a small inference demo.

## Team goal
Build a careful, honest project: every member can explain the whole pipeline, results are reported with their limits, and the test set is never used for tuning. Team role: Eyad Hatem, Project Manager.

## Where the project is now
This repository currently contains Milestones 1 and 2 only:
- Milestone 1: problem formulation and exploratory data analysis (`notebooks/01_eda.ipynb`).
- Milestone 2: data engineering and preprocessing (`notebooks/02_preprocessing.ipynb`, `src/data.py`).

The later milestones are being finished and will be added to this repository later.

## Limitations
This is a course project, not a diagnostic tool. The data come from one pediatric source, so results may not carry over to adults or other hospitals.
