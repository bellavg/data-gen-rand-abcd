import optuna
from optuna.storages import JournalFileStorage, JournalStorage

# 1. Connect to your existing study
db_path = "/scratch-shared/igardner1/aig_optuna_run/optuna_study.log"
storage = JournalStorage(JournalFileStorage(db_path))
study = optuna.load_study(study_name="aig_opt_hp_tuning", storage=storage)

trials_to_retry = []

for trial in study.trials:
    # Catch DataLoader crashes (FAIL) and Orphaned OS Kills (RUNNING)
    if trial.state in [optuna.trial.TrialState.FAIL, optuna.trial.TrialState.RUNNING]:
        trials_to_retry.append(trial)

    # Catch CUDA OOMs that were pruned instantly.
    # A natural pruning has intermediate values (scores from epochs).
    # An instant CUDA OOM has 0 intermediate values.
    elif trial.state == optuna.trial.TrialState.PRUNED:
        if len(trial.intermediate_values) == 0:
            trials_to_retry.append(trial)

print(f"Found {len(trials_to_retry)} trials that died due to OOM/Crashes.")

# 2. Push their parameters back into the queue
for trial in trials_to_retry:
    if len(trial.params) > 0:
        print(f"Re-enqueuing Trial {trial.number} parameters...")
        study.enqueue_trial(trial.params)

print("\nRecovery complete! You can now resume your bash script.")
print("Optuna will prioritize these enqueued trials first.")
