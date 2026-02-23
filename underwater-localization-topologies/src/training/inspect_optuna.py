import os
import optuna
import pandas as pd

storage = "sqlite:////home/fernando/tesis/underwater-localization-topologies/optuna_anp.db"

# 1) Ver qué estudios hay dentro de la DB
summaries = optuna.study.get_all_study_summaries(storage=storage)
for s in summaries:
    print(s.study_name, "n_trials=", s.n_trials, "best=", s.best_trial.value if s.best_trial else None)

# 2) Cargar tu estudio
study = optuna.load_study(study_name="anp_masked_v2", storage=storage)

print("\nSTUDY NAME:", study.study_name)
print("BEST VALUE:", study.best_value)
print("BEST PARAMS:", study.best_params)

# 3) Tabla de trials (incluye PRUNED/COMPLETE/FAIL)
df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/optuna", "optuna_anp_masked_v2_trials.csv")
df.to_csv(output_path, index=False)
print("\nTop-10 trials:")
print(df.sort_values("value").head(10)[["number","value","state"]])

# 4) Análisis agrupado por batch_size (o cualquier otro hyperparam)
study = optuna.load_study(study_name="anp_masked_v2", storage=storage)

df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
df = df[df["state"] == "COMPLETE"].copy()  # ignore PRUNED/FAIL

# ---- Grouped objective stats by batch size
parameter_name = "params_batch_size"
g = df.groupby(parameter_name)["value"]
print(f"\n=== Summary by {parameter_name} ===")
summary = pd.DataFrame({
    "n_trials": g.size(),
    "best": g.min(),
    "median": g.median(),
    "mean": g.mean(),
    "p10": g.quantile(0.10),
    "p90": g.quantile(0.90),
}).sort_values("best")

print(summary)
# save summary to CSV
summary_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/optuna", "optuna_anp_masked_v2_summary_by_batch_size.csv")
summary.to_csv(summary_output_path)

# ---- Top-N trials per batch size (to spot patterns in other params)
N = 5
cols = [
    "number", "value",
    "params_batch_size", "params_lr", "params_num_hidden", "params_weight_decay",
    "params_kl_warmup_epochs", "params_sensor_drop_mode", "params_sensor_drop_p",
    "params_mask_fill"
]
for bs, sub in df.sort_values("value").groupby("params_batch_size"):
    print(f"\n=== batch_size={bs} | top-{N} ===")
    print(sub[cols].head(N).to_string(index=False))