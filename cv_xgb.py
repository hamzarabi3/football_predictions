"""
Optuna example that demonstrates a pruner for XGBoost.cv.
In this example, we optimize the validation auc of cancer detection using XGBoost.
We optimize both the choice of booster model and their hyperparameters. Throughout
training of models, a pruner observes intermediate results and stop unpromising trials.
You can run this example as follows:
    $ python xgboost_cv_integration.py
"""

#%%
import optuna
import os
import pandas as pd
import xgboost as xgb
from pickle import load, dump
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
import joblib 
from xgboost.training import train
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

filterwarnings("ignore")


data_folder = "data"


features_csv = os.path.join(data_folder, "features.csv")
targets_csv = os.path.join(data_folder, "targets.csv")

#%%


def objective(trial):
    features = pd.read_csv(features_csv, index_col=0)
    targets = pd.read_csv(targets_csv, index_col=0)
    features.drop(['index','season'],axis=1,inplace=True)
    le= LabelEncoder()
    targets=le.fit_transform(targets)
    joblib.dump(le,os.path.join('models','encoder.joblib'))
    #train_X, test_X, train_Y, test_Y = train_test_split(features, targets, test_size=0.3)

    dtrain = xgb.DMatrix(features, label=targets,enable_categorical=True)

    param = {
        "verbosity": 0,
        "objective": "multi:softprob",
        "eval_metric": "auc",
        'num_class': 3,
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")
    history = xgb.cv(param, dtrain, num_boost_round=100, callbacks=[pruning_callback])

    mean_auc = history["test-auc-mean"].values[-1]
    return mean_auc


if __name__ == "__main__":
    
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(pruner=pruner, direction="maximize")
    study.optimize(objective, n_trials=100)

    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")

    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")