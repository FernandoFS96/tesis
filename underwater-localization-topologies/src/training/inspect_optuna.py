import optuna

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
df.to_csv("optuna_anp_masked_v1_trials.csv", index=False)
print("\nTop-10 trials:")
print(df.sort_values("value").head(10)[["number","value","state"]])