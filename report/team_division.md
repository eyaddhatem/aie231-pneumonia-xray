# AIE231 Project: Work Division for 6 Members
Project: Chest X-Ray Pneumonia classification (Kaggle, CC BY 4.0). Roles follow spec section 4. Each member owns one work package (WP), is examined on it, also trains at least one neural-network experiment, commits under their OWN GitHub account on feature branches, and has a pull request reviewed by a different member. Everyone must be able to explain the whole pipeline.

| # | Role | Owns (files) | Must code | Must train / run | Report section |
|---|---|---|---|---|---|
| 1 | Tech Lead | repo skeleton, configs/, README, train.py entry point, final model selection | project skeleton, config file, training entry point, merges | Final model with 3 seeds, final results table | 1 Abstract, 2 Introduction, 9 Contributions |
| 2 | Data Engineer | src/data.py, data/ | loading, cleaning, split, preprocessing with leakage checks, augmentation | Imbalance and augmentation ablations (class weights vs none, augmentation vs none) | 4.1 Preprocessing |
| 3 | Baseline and Analysis Engineer | notebooks/01_eda.ipynb, baseline scripts | EDA notebook, majority + logistic/MLP baselines | Model A (custom CNN) vs baselines | 3 Dataset and EDA, 4.2 Baselines |
| 4 | Model Architect | src/models.py | Models B and C, hand parameter counts, GradientTape step | Architecture and regularization ablations (Dropout, activation, frozen vs fine-tuned) | 4.3 Architectures |
| 5 | Experimentation Engineer | experiments/, runner and logging, curve plots | experiment runner, CSV logger, learning-curve plots | Learning rate (3), batch size (3), optimizer (SGD / momentum / Adam) studies | 5 Experiments |
| 6 | Evaluation and Deployment Engineer | src/evaluate.py, src/predict.py, app/ | evaluate.py, predict.py, Gradio demo | Confusion matrix, ROC, failure cases for every model; inference time and model size | 6 Results and error analysis, 7 Deployment, 8 Limitations |

## What each member is examined on (study checklist)
1. **Tech Lead:** why recall on pneumonia is the primary metric; the loss and output layer; how the final model was picked on validation only; why mean +- std over 3 seeds; test set used once.
2. **Data Engineer:** why the official 16-image val folder is replaced by a stratified 70/15/15 split; why duplicates are removed and patient ids grouped; leakage rules; class weights vs oversampling; which two augmentations and why flips/rotations are limited for X-rays; pixel scaling.
3. **Baseline/EDA:** class counts and imbalance ratio; what the majority baseline scores and why accuracy misleads; logistic regression / MLP on pixels; findings-to-decision table; how Model A beats baselines.
4. **Model Architect:** output shape of every layer and parameter formulas (Dense = in x units + units; Conv2D = (F x F x C_in + 1) x filters) checked against model.summary(); why ReLU, sigmoid + BCE, kernel size, padding, pooling; Dropout; transfer learning, why freeze, which block to unfreeze; one tf.GradientTape step (forward, loss, gradients, update).
5. **Experimentation:** one-factor-at-a-time tuning from a fixed reference; the experiment log columns; why SGD vs momentum vs Adam differ; how to read train/val curves (overfitting vs underfitting); why tuning never touches the test set.
6. **Evaluation/Deployment:** accuracy, precision, recall, F1, macro-F1, confusion matrix, ROC-AUC; 20+ misclassified images grouped into failure categories; the most common confusion and a proposed fix; predict.py shares preprocessing with training; inference time, model size.

## Rules from the spec all six must follow
- Meaningful commits from each member's own account through the project (a one-commit upload loses code-quality marks).
- Contribution table in the report: member, work package, files, experiments run.
- State in the report where AI tools were used (spec section 8).
- Team mark (90%) can drop to 50% for a member who cannot explain their work or has no meaningful commits. Defense is 10%.
- Team of six: all six roles exist as above.

## Suggested timeline (spec section 5)
End of Week 9: topic, roles, repo. End of Week 10: EDA, split, baselines (WP 2, 3). End of Week 12: three models, parameter counts, first curves (WP 4, 5). End of Week 13: tuning, ablations, test evaluation, error analysis, report draft (WP 1, 5, 6). Week 14: tag v1.0, report PDF, slides, demo.
