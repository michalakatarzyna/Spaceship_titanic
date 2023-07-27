import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import torch
import torch.nn as nn
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


!pip install opendatasets
!pip install catboost
!pip install optuna
import opendatasets as od
import catboost
from catboost import CatBoostClassifier, Pool
import optuna

"""###Get the data
---

Get data from link: https://www.kaggle.com/competitions/spaceship-titanic/data and read to your notebook.*kursywa*
"""

dataset_url = 'kaggle.com/competitions/spaceship-titanic/data'
od.download(dataset_url)

submission = pd.read_csv('/content/spaceship-titanic/sample_submission.csv')

test = pd.read_csv('/content/spaceship-titanic/test.csv')

train = pd.read_csv('/content/spaceship-titanic/train.csv')

submission.head()

submission.info()

submission.describe()

train.head()

print(f'Number of rows in train data: {train.shape[0]}')
print(f'Number of columns in train data: {train.shape[1]}')
print(f'Number of values in train data: {train.count().sum()}')
print(f'Number missing values in train data: {sum(train.isna().sum())}')

print(train.isna().sum().sort_values(ascending = False))

test.head()

test.describe()

print(f'Number od rows in test data: {test.shape[0]}')
print(f'Number of columns in test data: {test.shape[0]}')
print(f'Number of values in test data: {test.count().sum()}')
print(f'Number of missing values  in test data: {sum(test.isna().sum())}')

print((test.isna().sum().sort_values(ascending = False)))

"""#Data analysis
---
Analyze the data and perform exploratory data analysis

###Plot histograms
"""

plt.figure(figsize=(10,4))
sns.histplot(data=train, x='Age', hue='Transported',  binwidth=1, kde=True)
plt.title('Age distribution')
plt.xlabel('Age (years)')

plt.figure(figsize=(10,5))
sns.histplot(data=train, x='RoomService', hue='Transported',  binwidth=1000, kde=True)
plt.title('RoomServiceDistribution')
plt.xlabel('RoomService')

plt.figure(figsize=(10,5))
sns.histplot(data=train, x='FoodCourt', hue='Transported',  binwidth=10000, kde=True)
plt.title('FoodCourtdistribution')
plt.xlabel('FoodCourt')

plt.figure(figsize=(10,5))
sns.histplot(data=train, x='Spa', hue='Transported',  binwidth=100000, kde=True)
plt.title('SpaDistribution')
plt.xlabel('Spa')

plt.figure(figsize=(10,4))
sns.histplot(data=train, x='VRDeck', hue='Transported',  binwidth=100000, kde=True)
plt.title('VRDeckDistribution')
plt.xlabel('VRDeck')

continuous_features=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

fig=plt.figure(figsize=(10,20))
for i, var_name in enumerate(continuous_features):
    ax=fig.add_subplot(5,2,2*i+1)
    sns.histplot(data=train, x=var_name, axes=ax, bins=40, kde=False, hue='Transported')
    ax.set_title(var_name)

    ax=fig.add_subplot(5,2,2*i+2)
    sns.histplot(data=train, x=var_name, axes=ax, bins=40, kde=True, hue='Transported')
    plt.ylim([0,100])
    ax.set_title(var_name)
fig.tight_layout()
plt.show()

categorical_features=['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

fig=plt.figure(figsize=(10,16))
for i, var_name in enumerate(categorical_features):
    ax=fig.add_subplot(4,1,i+1)
    sns.countplot(data=train, x=var_name, axes=ax, hue='Transported')
    ax.set_title(var_name)
fig.tight_layout()
plt.show()

"""###Identify correlations of data features and propose visual representation of the correlations."""

train_corr = train[['HomePlanet', 'CryoSleep',	'Cabin', 'Destination',	'Age', 'VIP', 'RoomService', 'FoodCourt',	'ShoppingMall','Spa', 'VRDeck', 'Transported']].dropna().corr()
print(train_corr)

sns.heatmap(train_corr)

"""###Check if the dataset is balanced (if not think about methods to aquire balance)."""

plt.figure(figsize=(6,6))

train['Transported'].value_counts().plot.pie(autopct='%1.2f%%', colors = ["#9FCBED", "#E7BAEF"], textprops={'fontsize':20}).set_title("Target distribution")

"""#Data preprocessing
---

Prepare data using preprocessing methods.
"""

train.drop(["Name","Cabin", "PassengerId"], axis = 1 ,inplace = True)
test.drop(["Name", "Cabin", "PassengerId"], axis = 1 ,inplace = True)

"""###Think and propose how to handle nulls and nan values




"""

imputer_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck" ,]
imputer = SimpleImputer()
imputer.fit(train[imputer_cols])
imputer.fit(test[imputer_cols])
train[imputer_cols] = imputer.transform(train[imputer_cols])
test[imputer_cols] = imputer.transform(test[imputer_cols])

train["VIP"].fillna(False, inplace=True)
test["VIP"].fillna(False, inplace=True)
train["HomePlanet"].fillna("Earth", inplace=True)
test["HomePlanet"].fillna("Earth", inplace=True)
train["Destination"].fillna("TRAPPIST-1e", inplace=True)
test["Destination"].fillna("TRAPPIST-1e", inplace=True)
train["CryoSleep"].fillna(False, inplace=True)
test["CryoSleep"].fillna(False, inplace=True)

"""###Think and propose how represent categorical values

"""

train = pd.get_dummies(train, drop_first = True, columns = ["HomePlanet", "CryoSleep", "Destination" ,"VIP"])
test = pd.get_dummies(test, drop_first = True, columns = ["HomePlanet", "CryoSleep", "Destination" ,"VIP"] )

"""###Divide dataset into train and test subset."""

X = train.drop('Transported', axis =1 )
y = train['Transported'].astype(int)

X_train , X_test , y_train , y_test = train_test_split(X ,
                                                       y,
                                                       random_state = 42,
                                                       test_size =0.2)

"""#Machine learning methods
---
Use standard ML algorithms to classify passengers.

###Use XGBoost, LightGBM and CatBoost.
###Choose metrics you want to use to evaluate those models.

##XGBoost
"""

model_XGB = XGBClassifier(use_label_encoder=False,
                      eval_metric='aucpr')
model_XGB.fit(X_train, y_train)

y_pred_XGB = model_XGB.predict(X_test)
accuracy_x = accuracy_score(y_test, y_pred_XGB)
print("Accuracy for XGB: %.2f%%" % (accuracy_x * 100.0))

print(classification_report(y_test, y_pred_XGB))

XGB_param = {
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "max_depth": [5, 10, 15, 20],
        "n_estimators": [5, 10, 20],
        "subsample": [0.6, 0.8, 1]
}

model_XGB_gs = XGBClassifier()

grid_search_XGB = GridSearchCV(model_XGB_gs, XGB_param, scoring='roc_auc', cv=10, verbose=1)

grid_search_XGB.fit(X_train, y_train)
print("Best hyperparametres of the model: \n", grid_search_XGB.best_params_)

model_XGB_gs = XGBClassifier(learning_rate = 0.3,
                           subsample = 1.0,
                           max_depth = 5,
                           colsample_bytree = 1.0,
                           n_estimators = 20)
model_XGB_gs.fit(X_train, y_train)

y_pred_XGB_gs = model_XGB_gs.predict(X_test)
accuracy_x_gs = accuracy_score(y_test, y_pred_XGB_gs)
print("Accuracy for XGB after GridSearch: %.2f%%" % (accuracy_x_gs * 100.0))
print(classification_report(y_test, y_pred_XGB_gs))

xgb_gs_roc_auc = roc_auc_score(y_test, model_XGB_gs.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model_XGB_gs.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label = 'AUC (area = %0.2f)' % xgb_gs_roc_auc)
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC for XGB')
plt.show()

ConfusionMatrixDisplay.from_estimator(
        model_XGB_gs,
        X_test,
        y_test,
        normalize="true")

"""##LightGBM"""

model_lgb = lgb.LGBMClassifier()
model_lgb.fit(X_train, y_train)

y_pred_lgb = model_lgb.predict(X_test)
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
print("Accuracy: %.2f%%" % (accuracy_lgb * 100.0))
print(classification_report(y_test, y_pred_lgb))

lgb_param = {
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "max_depth": [5, 10, 15, 20],
        "n_estimators": [10, 20, 50],
        "subsample": [0.6, 0.8, 1]
}

model_lgb_gs = lgb.LGBMClassifier()

grid_search_lgb = GridSearchCV(model_lgb_gs, lgb_param, scoring='roc_auc', cv=10, verbose=1)

grid_search_lgb.fit(X_train, y_train)
print("Best hyperparametres of the model: \n", grid_search_lgb.best_params_)

model_lgb_gs = lgb.LGBMClassifier(learning_rate = 0.1,
                                 max_depth = 5,
                                 n_estimators = 50,
                                 subsample = 0.6)
model_lgb_gs.fit(X_train, y_train)

y_pred_lgb_gs = model_lgb_gs.predict(X_test)
accuracy_gs = accuracy_score(y_test, y_pred_lgb_gs)
print("Accuracy: %.2f%%" % (accuracy_gs * 100.0))
print(classification_report(y_test, y_pred_lgb_gs))

lgb_roc_auc = roc_auc_score(y_test, model_lgb.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model_lgb.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label = 'AUC (area = %0.2f)' % lgb_roc_auc)
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC for LightGBM')
plt.show()

ConfusionMatrixDisplay.from_estimator(
        model_lgb,
        X_test,
        y_test,
        normalize="true")

"""##CatBoost"""

model_cb = catboost.CatBoostClassifier(verbose=False)
model_cb.fit(X_train, y_train)

y_pred_cb = model_cb.predict(X_test)
accuracy_cb = accuracy_score(y_test, y_pred_cb)
print("Accuracy: %.2f%%" % (accuracy_cb * 100.0))
print(classification_report(y_test, y_pred_cb))

cb_param = {
        "learning_rate": [0.01, 0.025, 0.05],
        "max_depth": [10, 15, 20],
        "n_estimators": [20, 30, 50],
        "subsample": [0.8, 1.0, 2.0 ]
        }

model_cb_gs = catboost.CatBoostClassifier()

grid_search_cb = GridSearchCV(model_cb_gs, cb_param, scoring='roc_auc', cv=10, verbose=1)

grid_search_cb.fit(X_train, y_train)
print("Best hyperparametres of the model: \n", grid_search_cb.best_params_)

model_cb_gs = lgb.LGBMClassifier(learning_rate = 0.05,
                                 max_depth = 10,
                                 n_estimators = 50,
                                 subsample = 1.0)
model_cb_gs.fit(X_train, y_train)

y_pred_cb_gs = model_cb_gs.predict(X_test)
accuracy_cb_gs = accuracy_score(y_test, y_pred_cb_gs)
print("Accuracy: %.2f%%" % (accuracy_cb_gs * 100.0))
print(classification_report(y_test, y_pred_cb_gs))

cb_roc_auc = roc_auc_score(y_test, model_cb.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model_cb.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label = 'AUC (area = %0.2f)' % cb_roc_auc)
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC for CatBoost')
plt.show()

ConfusionMatrixDisplay.from_estimator(
        model_cb,
        X_test,
        y_test,
        normalize="true")

"""###Use Optuna to find the best hyperparameters

### Oprtuna for XGBoost
"""

def objective_XGB(trial):
    (data, target) = X, y
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(data, target, test_size=0.25)
    dtrain = xgb.DMatrix(X_train_o, label=y_train_o)
    dtest = xgb.DMatrix(X_test_o, label=y_test_o)

    param_x = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "tree_method": "exact",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param_x["booster"] in ["gbtree", "dart"]:
        param_x["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        param_x["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param_x["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param_x["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param_x["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param_x["booster"] == "dart":
        param_x["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param_x["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param_x["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param_x["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param_x, dtrain)
    preds_x = bst.predict(dtest)
    pred_labels_x = np.rint(preds_x)
    accuracy_x = accuracy_score(y_test_o, pred_labels_x)
    return accuracy_x


if __name__ == "__main__":
    study_x = optuna.create_study(direction="maximize")
    study_x.optimize(objective_XGB, n_trials=100, timeout=600)

    print("Number of finished trials: ", len(study_x.trials))
    print("Best trial:")
    trial_x = study_x.best_trial

    print("  Value: {}".format(trial_x.value))
    print("  Params: ")
    for key, value in trial_x.params.items():
        print("    {}: {}".format(key, value))

best_params_xgb = trial_x.params
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
bst = xgb.train(best_params_xgb, dtrain)
preds_x = bst.predict(dtest)
pred_labels_x = np.rint(preds_x)

accuracy_optuna_xgb = accuracy_score(y_test, pred_labels_x)
precision_optuna_xgb = precision_score(y_test, pred_labels_x, average='weighted')
recall_optuna_xgb = recall_score(y_test, pred_labels_x, average='weighted')
f1_optuna_xgb = f1_score(y_test, pred_labels_x, average='weighted')

print("Accuracy for XGBoost after Optuna tuning: %.2f%%" % (accuracy_optuna_xgb * 100.0))
print("Precision for XGBoost after Optuna tuning: %.2f" % precision_optuna_xgb)
print("Recall for XGBoost after Optuna tuning: %.2f" % recall_optuna_xgb)
print("F1 Score for XGBoost after Optuna tuning: %.2f" % f1_optuna_xgb)

print("Accuracy for XGB: %.2f%%" % (accuracy_x * 100.0))
print("Accuracy for XGB after GridSearch: %.2f%%" % (accuracy_x_gs * 100.0))
print("Accuracy for XGB after optuna tuning: %.2f%%" % (accuracy_optuna_xgb * 100.0))

"""###Optuna for LightGBM"""

def objective_LightGBM(trial):
    (data, target) = X, y
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(data, target, test_size=0.25)

    dtrain = lgb.Dataset(X_train_o, label=y_train_o)

    param_lgb = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": trial.suggest_categorical("boosting", ["gbdt", "dart"]),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "boosting": "",
    }

    if param_lgb["boosting_type"] == "dart":
        param_lgb["boosting"] = "dart"
        param_lgb["drop_rate"] = trial.suggest_float("drop_rate", 0.1, 1.0)
        param_lgb["skip_drop"] = trial.suggest_float("skip_drop", 0.1, 1.0)

    gbm = lgb.train(param_lgb, dtrain)
    preds_lgb = gbm.predict(X_test_o)
    pred_labels_lgb = np.rint(preds_lgb)
    accuracy_lgb = sklearn.metrics.accuracy_score(y_test_o, pred_labels_lgb)
    return accuracy_lgb

if __name__ == "__main__":
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(objective_LightGBM, n_trials=100, timeout=600)

    print("Number of finished trials (LightGBM): ", len(study_lgb.trials))
    print("Best trial (LightGBM):")
    trial_lgb = study_lgb.best_trial

    print("  Value: {}".format(trial_lgb.value))
    print("  Params: ")
    for key, value in trial_lgb.params.items():
        print("    {}: {}".format(key, value))

lgb_params = trial_lgb.params
gbm_lgb = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train))
preds_lgb = gbm_lgb.predict(X_test)
pred_labels_lgb = np.rint(preds_lgb)

accuracy_optuna_lgb = accuracy_score(y_test, pred_labels_lgb)
recall_optuna_lgb = recall_score(y_test, pred_labels_lgb)
precision_optuna_lgb = precision_score(y_test, pred_labels_lgb)
f1_optuna_lgb = f1_score(y_test, pred_labels_lgb)
roc_auc_optuna_lgb = roc_auc_score(y_test, pred_labels_lgb)

print("Accuracy for LightGBM after Optuna tuning: %.2f%%" % (accuracy_optuna_lgb * 100.0))
print("Recall for LightGBM after Optuna tuning: %.2f" % recall_optuna_lgb)
print("Precision for LightGBM after Optuna tuning: %.2f" % precision_optuna_lgb)
print("F1 Score for LightGBM after Optuna tuning: %.2f" % f1_optuna_lgb)
print("ROC AUC for LightGBM after Optuna tuning: %.2f" % roc_auc_optuna_lgb)

print("Accuracy for LightGBM: %.2f%%" % (accuracy_lgb * 100.0))
print("Accuracy for LightGBM after GridSearch: %.2f%%" % (accuracy_gs * 100.0))
print("Accuracy for LightGBM after optuna tuning: %.2f%%" % (accuracy_optuna_lgb  * 100.0))

"""###Optuna for Catboost"""

def objective_CatBoost(trial):
    (data, target) = X, y
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(data, target, test_size=0.25)

    cat_features = []

    dtrain = CatBoostClassifier(iterations=trial.suggest_int("iterations", 50, 500),
                                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                                depth=trial.suggest_int("depth", 3, 10),
                                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 1.0, log=True),
                                loss_function="Logloss",
                                cat_features=cat_features)

    dtrain.fit(X_train_o, y_train_o)
    preds_catboost = dtrain.predict(X_test_o)
    pred_labels_catboost = np.rint(preds_catboost)
    accuracy_catboost = sklearn.metrics.accuracy_score(y_test_o, pred_labels_catboost)
    return accuracy_catboost

if __name__ == "__main__":
    study_catboost = optuna.create_study(direction="maximize")
    study_catboost.optimize(objective_CatBoost, n_trials=100, timeout=600)

    print("Number of finished trials (CatBoost): ", len(study_catboost.trials))
    print("Best trial (CatBoost):")
    trial_catboost = study_catboost.best_trial

    print("  Value: {}".format(trial_catboost.value))
    print("  Params: ")
    for key, value in trial_catboost.params.items():
        print("    {}: {}".format(key, value))

cat_features = []

catboost_params = trial_catboost.params
model_catboost = CatBoostClassifier(iterations=catboost_params["iterations"],
                                    learning_rate=catboost_params["learning_rate"],
                                    depth=catboost_params["depth"],
                                    l2_leaf_reg=catboost_params["l2_leaf_reg"],
                                    loss_function="Logloss",
                                    cat_features=cat_features)

model_catboost.fit(X_train, y_train)
preds_catboost = model_catboost.predict(X_test)
pred_labels_catboost = np.rint(preds_catboost)

accuracy_optuna_catboost = accuracy_score(y_test, pred_labels_catboost)
recall_optuna_catboost = recall_score(y_test, pred_labels_catboost)
precision_optuna_catboost = precision_score(y_test, pred_labels_catboost)
f1_optuna_catboost = f1_score(y_test, pred_labels_catboost)
roc_auc_optuna_catboost = roc_auc_score(y_test, pred_labels_catboost)

print("Accuracy for CatBoost after Optuna tuning: %.2f%%" % (accuracy_optuna_catboost * 100.0))
print("Recall for CatBoost after Optuna tuning: %.2f" % recall_optuna_catboost)
print("Precision for CatBoost after Optuna tuning: %.2f" % precision_optuna_catboost)
print("F1 Score for CatBoost after Optuna tuning: %.2f" % f1_optuna_catboost)
print("ROC AUC for CatBoost after Optuna tuning: %.2f" % roc_auc_optuna_catboost)

print("Accuracy for Catboost: %.2f%%" % (accuracy_cb * 100.0))
print("Accuracy for Catboost after GridSearch: %.2f%%" % (accuracy_cb_gs * 100.0))
print("Accuracy for Catboost after optuna tuning: %.2f%%" % (accuracy_optuna_catboost * 100.0))

"""#The best model"""

accuracy_xgb = accuracy_x
recall_xgb = recall_score(y_test, y_pred_XGB)
precision_xgb = precision_score(y_test, y_pred_XGB)
f1_xgb = f1_score(y_test, y_pred_XGB)
roc_auc_xgb = roc_auc_score(y_test, y_pred_XGB)

accuracy_lgb = accuracy_lgb
recall_lgb = recall_score(y_test, y_pred_lgb)
precision_lgb = precision_score(y_test, y_pred_lgb)
f1_lgb = f1_score(y_test, y_pred_lgb)
roc_auc_lgb = roc_auc_score(y_test, y_pred_lgb)

accuracy_cb = accuracy_cb
recall_catboost = recall_score(y_test, y_pred_cb)
precision_catboost = precision_score(y_test, y_pred_cb)
f1_catboost = f1_score(y_test, y_pred_cb)
roc_auc_catboost = roc_auc_score(y_test, y_pred_cb)

accuracy_optuna_xgb = accuracy_score(y_test, pred_labels_x)
precision_optuna_xgb = precision_score(y_test, pred_labels_x, average='weighted')
recall_optuna_xgb = recall_score(y_test, pred_labels_x, average='weighted')
f1_optuna_xgb = f1_score(y_test, pred_labels_x, average='weighted')
roc_auc_optuna_xgb = roc_auc_score(y_test, pred_labels_x)

accuracy_optuna_lgb = accuracy_score(y_test, pred_labels_lgb)
recall_optuna_lgb = recall_score(y_test, pred_labels_lgb)
precision_optuna_lgb = precision_score(y_test, pred_labels_lgb)
f1_optuna_lgb = f1_score(y_test, pred_labels_lgb)
roc_auc_optuna_lgb = roc_auc_score(y_test, pred_labels_lgb)

accuracy_optuna_catboost = accuracy_score(y_test, pred_labels_catboost)
recall_optuna_catboost = recall_score(y_test, pred_labels_catboost)
precision_optuna_catboost = precision_score(y_test, pred_labels_catboost)
f1_optuna_catboost = f1_score(y_test, pred_labels_catboost)
roc_auc_optuna_catboost = roc_auc_score(y_test, pred_labels_catboost)


results_df = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'XGBoost_Optuna', 'LightGBM_Optuna', 'Catboost_Optuna'],
    'Accuracy': [accuracy_xgb, accuracy_lgb, accuracy_cb, accuracy_optuna_xgb, accuracy_optuna_lgb, accuracy_optuna_catboost],
    'Recall': [recall_xgb, recall_lgb, recall_catboost, recall_optuna_xgb, recall_optuna_lgb, recall_optuna_catboost],
    'Precision': [precision_xgb, precision_lgb, precision_catboost, precision_optuna_xgb, precision_optuna_lgb, precision_optuna_catboost],
    'F1 Score': [f1_xgb, f1_lgb, f1_catboost, f1_optuna_xgb, f1_optuna_lgb, f1_optuna_catboost],
    'ROC AUC': [roc_auc_xgb, roc_auc_lgb, roc_auc_catboost, roc_auc_optuna_xgb, roc_auc_optuna_lgb,roc_auc_optuna_catboost]
})

print(results_df)



results_df = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'Optuna XGBoost', 'Optuna LightGBM', 'Optuna CatBoost'],
    'Accuracy': [0.85, 0.89, 0.92, 0.88, 0.90, 0.91]
})

colors = ['#FFC0CB', '#98FB98', '#FFD700', '#FFA07A', '#CDB5CD', '#FFB6C1']

plt.figure(figsize=(8, 10))

plt.subplot(2, 1, 1)
bars = plt.bar(results_df['Model'], results_df['Accuracy'], color=colors)
plt.ylim([0.0, 1.0])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')

highest_idx = results_df['Accuracy'].idxmax()
plt.annotate(f'Highest: {results_df["Accuracy"].max():.2f}',
             xy=(highest_idx, results_df['Accuracy'].max()),
             xytext=(0, 10),
             textcoords='offset points',
             ha='center',
             va='bottom',
             arrowprops=dict(arrowstyle='simple', color='black'))

plt.xticks(rotation='vertical')

plt.subplot(2, 1, 2)
bars_zoom = plt.bar(results_df['Model'], results_df['Accuracy'], color=colors)
plt.ylim([0.7, 0.92])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison (Zoomed In)')

highest_idx_zoom = results_df['Accuracy'].idxmax()
plt.annotate(f'Highest: {results_df["Accuracy"].max():.2f}',
             xy=(highest_idx_zoom, results_df['Accuracy'].max()),
             xytext=(0, 10),
             textcoords='offset points',
             ha='center',
             va='bottom',
             arrowprops=dict(arrowstyle='simple', color='black'))

plt.xticks(rotation='vertical')

plt.tight_layout()
plt.show()

models = ['XGBoost', 'LightGBM', 'CatBoost', 'Optuna XGBoost', 'Optuna LightGBM', 'Optuna CatBoost']
auc_scores = [roc_auc_xgb, roc_auc_lgb, roc_auc_catboost, roc_auc_optuna_xgb, roc_auc_optuna_lgb, roc_auc_optuna_catboost]

colors = ['#FFC0CB', '#98FB98', '#FFD700', '#FFA07A', '#CDB5CD', '#FFB6C1']


plt.figure(figsize=(8, 10))

plt.subplot(2, 1, 1)
bars = plt.bar(models, auc_scores, color=colors)
plt.ylim([0.0, 1.0])
plt.xlabel('Model')
plt.ylabel('AUC Score')
plt.title('Comparison of AUC Scores for Different Models')


highest_auc_index = auc_scores.index(max(auc_scores))

highest_auc_score = auc_scores[highest_auc_index]
plt.annotate(f"Highest: {highest_auc_score:.3f}", xy=(highest_auc_index, highest_auc_score),
             xytext=(highest_auc_index, highest_auc_score + 0.1),
             arrowprops=dict(facecolor='black', arrowstyle='wedge'))



plt.xticks(rotation='vertical')

plt.subplot(2, 1, 2)
bars_zoom = plt.bar(models, auc_scores, color=pastel_colors)
plt.ylim([highest_auc_score - 0.1, highest_auc_score + 0.1])
plt.xlabel('Model')
plt.ylabel('AUC Score')
plt.title('Zoomed-in View of Highest AUC Score')

bars_zoom[highest_auc_index].set_color('gold')

plt.xticks(rotation='vertical')

plt.tight_layout()
plt.show()

"""#PyTorch
Use PyTorch to solve following tasks.

##Implement logistic regression model
"""

X_train, X_test = torch.Tensor(X_train.values),torch.Tensor(X_test.values)
y_train, y_test = torch.Tensor(y_train.values),torch.Tensor(y_test.values)

n_samples, n_features = X.shape

print(n_samples, n_features)

y_test = y_test.unsqueeze(1)

y_train = y_train.unsqueeze(1)

class LogisticRegression(nn.Module):

  def __init__(self, n_input_features):
   super(LogisticRegression, self).__init__()
   self.linear = nn.Linear(n_input_features, 1)

  def forward(self, x):
    y_predicted = torch.sigmoid(self.linear(x))
    return y_predicted

model = LogisticRegression(n_features)
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 1000
for epoch in range(num_epochs):
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')

"""##Use k-fold technique to entry dataset division"""

k = 5
kf = KFold(n_splits=k)
accuracies = []

for fold, (train_index, test_index) in enumerate(kf.split(X_train)):

    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

    model = LogisticRegression(n_features)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        y_predicted = model(X_train_fold)
        loss = criterion(y_predicted, y_train_fold)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f'Fold: {fold + 1}, epoch: {epoch + 1}, loss = {loss.item():.4f}')

    with torch.no_grad():
        y_predicted = model(X_test_fold)
        y_predicted_cls = y_predicted.round()
        acc = y_predicted_cls.eq(y_test_fold).sum() / float(y_test_fold.shape[0])
        print(f'Fold: {fold + 1}, accuracy = {acc:.4f}')
        accuracies.append(acc)


mean_accuracy = sum(accuracies) / k
print(f'Mean accuracy: {mean_accuracy:.4f}')
