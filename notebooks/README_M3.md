# Milestone 3 explanation (oral exam, Baseline Engineer)
- Majority baseline: always predict the most common class. High accuracy, zero recall on Normal. Shows accuracy misleads.
- Logistic regression = one neuron: sigmoid(w.x + b), log-loss. It is the simplest learned model.
- MLP with one hidden layer: Dense(64, ReLU), Dropout(0.3), Dense(1, sigmoid). Params: 4096*64+64 = 262,208 and 64+1 = 65, total 262,273.
- Scaler fit on train only (no leakage). 64x64 grey flattened pixels.
- Compare on validation, never on test. Baselines are the numbers every deep model must beat.
- Run log: experiments/experiment_log.csv (run id, date, author, model, hyperparameters, seed, time, params, metrics).
