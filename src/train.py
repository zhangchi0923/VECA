import argparse
import numpy as np
import joblib
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from utils.preprocess import get_train_data, feature_preprocess, diag
from utils.logger import get_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m',
    '--model',
    type=str,
    choices=['svr', 'mlp', 'gbrt', 'lasso'],
    help='Specify which ML model to train and evaluate.'
)
parser.add_argument(
    "--info_dir",
    type=str,
    default='./data/data_info.xlsx',
    help="Specify file path of data info excel of training data.",
)
parser.add_argument(
    '--log_dir',
    type=str,
    default='./src/train_logs/',
    help='Specify directory path of log files.'
)
parser.add_argument(
    '--model_dir',
    type=str,
    default='./src/models/',
    help='Specify model persistent directory.'
)
parser.add_argument(
    '--importance',
    action='store_true',
    help="Specify whether to output feature importances."
)
parser.add_argument(
    '--shap',
    action='store_true',
    help='Specify whether to compute normalized shap values.'
)
parser.add_argument(
    '--roc',
    action='store_true',
    help='Specify whether to analyze education grouped classification ROCs.'
)
args = parser.parse_args()

info_dir = args.info_dir
log_dir = args.log_dir
model_name = args.model
model_dir = args.model_dir
_imp = args.importance
_shap = args.shap
_roc = args.roc

logger = get_logger(log_dir)

## train
train_data = get_train_data(info_dir)

if model_name == 'gbrt':
    X_train, X_test, y_train, y_test, cols = feature_preprocess(train_data=train_data, normalize=False)
    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1
    )
else:
    X_train, X_test, y_train, y_test, cols = feature_preprocess(train_data=train_data, normalize=True)
    if model_name == 'mlp':
        model = MLPRegressor(
            hidden_layer_sizes=(16, 32, 16, 6),
            activation='relu',
            solver='adam',
            alpha=0.05,
            max_iter=1000
        )
    elif model_name == 'svr':
        model = SVR(
            kernel='linear',
            C=0.5,
            shrinking=True,
        )
    elif model_name == 'lasso':
        model = Lasso(
            alpha=0.1,
            max_iter=50
        )

model.fit(X_train, y_train)
joblib.dump(model, f'{model_dir}{model_name}.joblib')
logger.info(f'Trained {model_name} has been saved to {model_dir}')
logger.info('--------------------------------------------------------')
## evaluation
y_pred = model.predict(X_test)
y_baseline = X_test[:, 6:].mean(axis=1)*30
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
corr = pearsonr(y_pred, y_test)[0]
base_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
base_mae = mean_absolute_error(y_test, y_baseline)
base_medae = median_absolute_error(y_test, y_baseline)
base_corr = pearsonr(y_baseline, y_test)[0]
logger.info(f'{model_name} MedianAE on testing data: {medae} | Baseline: {base_medae}.')
logger.info(f'{model_name} MAE on testing data: {mae} | Baseline: {base_mae}.')
logger.info(f'{model_name} MSE on testing data: {rmse} | Baseline: {base_rmse}.')
logger.info(f'{model_name} Pearson on testing data: {corr} | Baseline: {base_corr}.')
logger.info('--------------------------------------------------------')

if _imp:
    imp_model = GradientBoostingRegressor()
    imp_model.fit(X_train, y_train)
    imp = imp_model.feature_importances_
    ovr_ratio = [imp[0:2].sum(), imp[2:5].sum(), imp[5], imp[6:].sum()]
    feat = ['Gender', 'Edu', 'Age', 'Task']
    logger.info(
        f'Feature importances from Gradient Boosting Regression.\
            \nGender: {ovr_ratio[0]:.3f}\
            \nEdu: {ovr_ratio[1]:.3f}\
            \nAge: {ovr_ratio[2]:.3f}\
            \nTask: {ovr_ratio[3]:.3f}'
    )
    logger.info('--------------------------------------------------------')

if _shap:
    import shap
    shap_data = X_test
    logger.info('Conducting shap analysis.')
    logger.info('--------------------------------------------------------')
    explainer = shap.KernelExplainer(model.predict, data=shap_data)
    shape_values = explainer(shap_data)
    ordered_shap = np.sort(np.abs(shape_values.values).mean(axis=0))
    sum = ordered_shap.sum()
    norm_shap = (ordered_shap/sum)
    argsorts = np.argsort(np.abs(shape_values.values).mean(axis=0))
    logger.info('--------------------------------------------------------')
    logger.info('Shapley Values of Features')
    logger.info('--------------------------------------------------------')
    for col, imp in zip(cols[argsorts], norm_shap):
        logger.info(f'{col:20}{imp:.4f}')
    logger.info('--------------------------------------------------------')

if _roc:
    logger.info('Computing education grouped ROCs.')
    logger.info('--------------------------------------------------------')
    data = pd.get_dummies(train_data, prefix_sep='_', columns=['gender', 'edu'])
    data['diag'] = data.apply(lambda x: diag(x), axis=1)
    edu_group_0 = data.loc[data['edu_0'] ==  1, :]
    edu_group_1 = data.loc[data['edu_1'] ==  1, :]
    edu_group_2 = data.loc[data['edu_2'] ==  1, :]

    from sklearn.metrics import roc_auc_score
    ai_score_0 = model.predict(edu_group_0[cols].values)
    ai_score_1 = model.predict(edu_group_1[cols].values)
    ai_score_2 = model.predict(edu_group_1[cols].values)
    diag0 = edu_group_0['diag'].values
    diag1 = edu_group_1['diag'].values
    diag2 = edu_group_2['diag'].values
    roc_0 = roc_auc_score(diag0, ai_score_0)
    roc_1 = roc_auc_score(diag0, ai_score_0)
    roc_2 = roc_auc_score(diag0, ai_score_0)
    logger.info(f'Group 1 ROC: {roc_0}')
    logger.info(f'Group 2 ROC: {roc_1}')
    logger.info(f'Group 3 ROC: {roc_2}')
    logger.info('--------------------------------------------------------')