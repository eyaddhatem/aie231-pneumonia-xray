# Milestone 2 explanation (oral exam, Data Engineer)
- Why not the official val folder: 16 images, too noisy to select models.
- Stratified = same class ratio in train/val/test. Grouped = a patient never crosses splits (leakage).
- Limitation: NORMAL images have no patient id.
- No scaler to fit: pixel scaling is inside the model, so inference reuses it.
- Class weights are computed from train only. w_c = N / (2 n_c). Rare class gets a bigger weight.
- Augmentation only on train; no flips because of anatomy.
- Test set used once at the end.
