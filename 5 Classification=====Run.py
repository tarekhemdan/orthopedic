# Hyperopt library as an alternative regression optimizer
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time
from hyperopt import fmin, tpe, hp, Trials
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('column_2C_weka.csv')

# Extract the 'status' column
status = df['status']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'status' column
status_encoded = label_encoder.fit_transform(status)

# Update the 'status' column in the DataFrame
df['status'] = status_encoded

# Preprocess the dataset
X = df.drop('status', axis=1)
y = df['status']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to pandas DataFrames
X_train = pd.DataFrame(X, columns=df.columns[:-1])
y_train = pd.DataFrame(y, columns=['status'])

# Define the objective function for Hyperopt
def objective(params):
    n_estimators = int(params['n_estimators'])
    max_depth = int(params['max_depth'])
    min_samples_split = int(params['min_samples_split'])
    min_samples_leaf = int(params['min_samples_leaf'])

    # Create the classifier with the suggested parameters
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 random_state=42)

    # Perform cross-validation with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        clf.fit(X_train_fold, y_train_fold.values.ravel())
        y_val_pred = clf.predict(X_val_fold)
        cv_scores.append(accuracy_score(y_val_fold, y_val_pred))

    # Calculate the average accuracy
    accuracy_avg = sum(cv_scores) / len(cv_scores)

    return 1 - accuracy_avg  # Optimize for 1 - accuracy

# Define the search space for Hyperopt
space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1)
}

# Optimize using Hyperopt with 100 iterations
trials = Trials()
start_time = time.time()
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)
end_time = time.time()

# Get the best parameters and retrain the model
best_params = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']),
    'min_samples_split': int(best['min_samples_split']),
    'min_samples_leaf': int(best['min_samples_leaf'])
}
best_clf = RandomForestClassifier(**best_params, random_state=42)
best_clf.fit(X_train.values, y_train.values.ravel())

# Predict on the training set using the best classifier
y_pred = best_clf.predict(X_train.values)

# Calculate classification metrics
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred, average='weighted')
recall = recall_score(y_train, y_pred, average='weighted')
f1 = f1_score(y_train, y_pred, average='weighted')

# Calculate ROC curve and AUC score
y_scores = best_clf.predict_proba(X_train.values)
fpr, tpr, thresholds = roc_curve(y_train, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Hyperopt')
plt.legend(loc="lower right")
plt.show()

# Print classification metrics and best parameters
print("Hyperopt Classification for status: ")
print("Best Parameters:", best_params)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC:", roc_auc)

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)

############################################################################################
# scikit-optimize library as an alternative regression optimizer
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time
from skopt import forest_minimize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('column_2C_weka.csv')

# Extract the 'status' column
status = df['status']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'status' column
status_encoded = label_encoder.fit_transform(status)

# Update the 'status' column in the DataFrame
df['status'] = status_encoded

# Preprocess the dataset
X = df.drop('status', axis=1)
y = df['status']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to pandas DataFrames
X_train = pd.DataFrame(X, columns=df.columns[:-1])
y_train = pd.DataFrame(y, columns=['status'])

# Define the objective function for scikit-optimize
def objective(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])

    # Create the classifier with the suggested parameters
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 random_state=42)

    # Perform cross-validation with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        clf.fit(X_train_fold, y_train_fold.values.ravel())
        y_val_pred = clf.predict(X_val_fold)
        cv_scores.append(accuracy_score(y_val_fold, y_val_pred))

    # Calculate the average accuracy
    accuracy_avg = sum(cv_scores) / len(cv_scores)

    return 1 - accuracy_avg  # Optimize for 1 - accuracy

# Define the search space for scikit-optimize
space = [
    (100, 1000),  # n_estimators
    (5, 30),  # max_depth
    (2, 20),  # min_samples_split
    (1, 10)  # min_samples_leaf
]

# Optimize using scikit-optimize with 100 iterations
start_time = time.time()
res = forest_minimize(objective, space, n_calls=100, random_state=42)
end_time = time.time()

# Get the best parameters and retrain the model
best_params = {
    'n_estimators': int(res.x[0]),
    'max_depth': int(res.x[1]),
    'min_samples_split': int(res.x[2]),
    'min_samples_leaf': int(res.x[3])
}
best_clf = RandomForestClassifier(**best_params, random_state=42)
best_clf.fit(X_train.values, y_train.values.ravel())

# Predict on the training set using the best classifier
y_pred = best_clf.predict(X_train.values)

# Calculate classification metrics
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred, average='weighted')
recall = recall_score(y_train, y_pred, average='weighted')
f1 = f1_score(y_train, y_pred, average='weighted')

# Calculate ROC curve and AUC score
y_scores = best_clf.predict_proba(X_train.values)
fpr, tpr, thresholds = roc_curve(y_train, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for scikit-optimize')
plt.legend(loc="lower right")
plt.show()

# Print classification metrics and best parameters
print("Hyperopt Classification for status: ")
print("Best Parameters:", best_params)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC:", roc_auc)

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)

###################################################################################
# pip install optunity
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time
import optunity
import optunity.metrics as metrics
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('column_2C_weka.csv')

# Extract the 'status' column
status = df['status']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'status' column
status_encoded = label_encoder.fit_transform(status)

# Update the 'status' column in the DataFrame
df['status'] = status_encoded

# Preprocess the dataset
X = df.drop('status', axis=1)
y = df['status']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to pandas DataFrames
X_train = pd.DataFrame(X, columns=df.columns[:-1])
y_train = pd.DataFrame(y, columns=['status'])


# Define the objective function for Optunity
def objective(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    # Create the classifier with the suggested parameters
    clf = RandomForestClassifier(n_estimators=int(n_estimators),
                                 max_depth=int(max_depth),
                                 min_samples_split=int(min_samples_split),
                                 min_samples_leaf=int(min_samples_leaf),
                                 random_state=42)

    # Perform cross-validation with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        clf.fit(X_train_fold, y_train_fold.values.ravel())
        y_val_pred = clf.predict(X_val_fold)
        cv_scores.append(metrics.accuracy(y_val_fold, y_val_pred))

    # Calculate the average accuracy
    accuracy_avg = sum(cv_scores) / len(cv_scores)

    return accuracy_avg  # Optimize for accuracy


# Optimize using Optunity with 100 iterations
start_time = time.time()
optimal_pars, _, _ = optunity.maximize(objective, num_evals=100, n_estimators=[100, 1000], max_depth=[5, 30],
                                       min_samples_split=[2, 20], min_samples_leaf=[1, 10])
end_time = time.time()

# Get the best parameters and retrain the model
best_params = {
    'n_estimators': int(optimal_pars['n_estimators']),
    'max_depth': int(optimal_pars['max_depth']),
    'min_samples_split': int(optimal_pars['min_samples_split']),
    'min_samples_leaf': int(optimal_pars['min_samples_leaf'])
}
best_clf = RandomForestClassifier(**best_params, random_state=42)
best_clf.fit(X_train.values, y_train.values.ravel())

# Predict on the training set using the best classifier
y_pred = best_clf.predict(X_train.values)

# Calculate classification metrics
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)

# Calculate ROC curve and AUC score
y_scores = best_clf.predict_proba(X_train.values)
fpr, tpr, thresholds = roc_curve(y_train, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for optunity')
plt.legend(loc="lower right")
plt.show()

# Print classification metrics and best parameters
print("Hyperopt Classification for status: ")
print("Best Parameters:", best_params)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC:", roc_auc)

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)

#####################################################################################
# pip install GPyOpt

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time
import GPyOpt
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('column_2C_weka.csv')

# Extract the 'status' column
status = df['status']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'status' column
status_encoded = label_encoder.fit_transform(status)

# Update the 'status' column in the DataFrame
df['status'] = status_encoded

# Preprocess the dataset
X = df.drop('status', axis=1)
y = df['status']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to pandas DataFrames
X_train = pd.DataFrame(X, columns=df.columns[:-1])
y_train = pd.DataFrame(y, columns=['status'])


# Define the objective function for GPyOpt
def objective(params):
    n_estimators = int(params[0, 0])
    max_depth = int(params[0, 1])
    min_samples_split = int(params[0, 2])
    min_samples_leaf = int(params[0, 3])

    # Create the classifier with the suggested parameters
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 random_state=42)

    # Perform cross-validation with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        clf.fit(X_train_fold, y_train_fold.values.ravel())
        y_val_pred = clf.predict(X_val_fold)
        cv_scores.append(accuracy_score(y_val_fold, y_val_pred))

    # Calculate the average accuracy
    accuracy_avg = sum(cv_scores) / len(cv_scores)

    return 1 - accuracy_avg  # Optimize for 1 - accuracy


# Define the search space for GPyOpt
space = [
    {'name': 'n_estimators', 'type': 'discrete', 'domain': (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)},
    {'name': 'max_depth', 'type': 'discrete', 'domain': (5, 6, 7, 8, 9, 10, 15, 20, 25, 30)},
    {'name': 'min_samples_split', 'type': 'discrete', 'domain': (2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20)},
    {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)}
]

# Optimize using GPyOpt with 100 iterations
opt = GPyOpt.methods.BayesianOptimization(f=objective, domain=space, acquisition_type='EI', exact_feval=True)
start_time = time.time()
opt.run_optimization(max_iter=100)
end_time = time.time()

# Get the best parameters and retrain the model
best_params = {
    'n_estimators': int(opt.x_opt[0]),
    'max_depth': int(opt.x_opt[1]),
    'min_samples_split': int(opt.x_opt[2]),
    'min_samples_leaf': int(opt.x_opt[3])
}
best_clf = RandomForestClassifier(**best_params, random_state=42)
best_clf.fit(X_train.values, y_train.values.ravel())

# Predict on the training set using the best classifier
y_pred = best_clf.predict(X_train.values)

# Calculate classification metrics
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)

# Calculate ROC curve and AUC score
y_scores = best_clf.predict_proba(X_train.values)
fpr, tpr, thresholds = roc_curve(y_train, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for GPyOpt')
plt.legend(loc="lower right")
plt.show()

# Print classification metrics and best parameters
print("Hyperopt Classification for status: ")
print("Best Parameters:", best_params)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC:", roc_auc)

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)

##########################################################################################
#Optuna
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('column_2C_weka.csv')

# Extract the 'status' column
status = df['status']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'status' column
status_encoded = label_encoder.fit_transform(status)

# Update the 'status' column in the DataFrame
df['status'] = status_encoded

# Preprocess the dataset
X = df.drop('status', axis=1)
y = df['status']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to pandas DataFrames
X_train = pd.DataFrame(X, columns=df.columns[:-1])
y_train = pd.DataFrame(y, columns=['status'])

# Define the objective function for Optuna
def objective(trial):
    # Define the parameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Create the classifier with the suggested parameters
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 random_state=42)

    # Perform cross-validation with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        clf.fit(X_train_fold, y_train_fold.values.ravel())
        y_val_pred = clf.predict(X_val_fold)
        cv_scores.append(accuracy_score(y_val_fold, y_val_pred))

    # Calculate the average accuracy
    accuracy_avg = sum(cv_scores) / len(cv_scores)

    return 1 - accuracy_avg  # Optimize for 1 - accuracy

# Optimize using Optuna
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
start_time = time.time()
study.optimize(objective, n_trials=100)
end_time = time.time()

# Get the best parameters and retrain the model
best_params = study.best_params
best_clf = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                  max_depth=best_params['max_depth'],
                                  min_samples_split=best_params['min_samples_split'],
                                  min_samples_leaf=best_params['min_samples_leaf'],
                                  random_state=42)
best_clf.fit(X_train.values, y_train.values.ravel())

# Predict on the training set using the best classifier
y_pred = best_clf.predict(X_train.values)

# Calculate classification metrics
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)

# Calculate ROC curve and AUC score
y_scores = best_clf.predict_proba(X_train.values)
fpr, tpr, thresholds = roc_curve(y_train, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Optuna')
plt.legend(loc="lower right")
plt.show()

# Print classification metrics and best parameters
print("Optuna Classification for status: ")
print("Best Parameters:", best_params)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC:", roc_auc)

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)