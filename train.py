import os
import pandas as pd
import xgboost as xgb
from pickle import load, dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
import joblib 
from xgboost.training import train

filterwarnings("ignore")


data_folder = "data"
features_csv = os.path.join(data_folder, "features.csv")
targets_csv = os.path.join(data_folder, "targets.csv")


features = pd.read_csv(features_csv, index_col=0)
try:
    features.drop(["Unnamed: 0", "index"], axis=1, inplace=True)
except KeyError:
    pass
targets = pd.read_csv(targets_csv, index_col=0)
from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()
targets=le.fit_transform(targets)
joblib.dump(le,os.path.join('models','label_encoder.joblib'))
print(features.shape)
print(targets.shape)

train_X, test_X, train_Y, test_Y = train_test_split(features, targets, test_size=0.3)
# test_X,val_X, test_Y,val_Y=train_test_split(test_X,test_Y,test_size=0.5)
dtrain = xgb.DMatrix(train_X, label=train_Y, enable_categorical=True)
dtest = xgb.DMatrix(test_X, label=test_Y, enable_categorical=True)
model_dir = os.path.join("models", "xgb_models")
print(train_X.columns)
random_state = 96
# for y in targets.columns:
#     # if gpu is available uncomment the following line
#     # xc=xgb.XGBClassifier(n_estimators=1000,tree_method='gpu_hist',random_state=random_state)
#     # if GPU is not available keep the following line
#     xc = xgb.XGBClassifier(n_estimators=1000, random_state=random_state)
#     xc.fit(train_X, train_Y[y])
#     a = int(xc.score(test_X, test_Y[y]) * 100)
#     save_file = os.path.join(model_dir, f"{y}_xgb_model.pickle")
#     with open(save_file, "wb") as f:
#         dump(xc, f)

#     print(f"saved model for {y}, accuracy :{a}%")


param = {
    'n_estimators':600,
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset

num_round = 20  # the number of training iterations

xgb.train(param,dtrain,num_round)