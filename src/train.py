import argparse
import numpy as np
import joblib
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from utils.preprocess import get_train_data, feature_preprocess
from utils.logger import get_logger

parser = argparse.ArgumentParser()
parser.add_argument(
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
args = parser.parse_args()

info_dir = args.info_path
log_dir = args.log_dir
model_name = args.model
model_dir = args.model_dir
logger = get_logger(log_dir)

## train
train_data = get_train_data(log_dir)

if model_name == 'gbrt':
    X_train, X_test, y_train, y_test = feature_preprocess(train_data=train_data, normalize=False)
    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1
    )
else:
    X_train, X_test, y_train, y_test = feature_preprocess(train_data=train_data, normalize=True)
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

## evaluation
y_pred = model.predict(X_test)
y_baseline = X_test[:, 6:].mean(axis=1)*30
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
corr = pearsonr(y_pred, y_test)[0][0]
base_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
base_mae = mean_absolute_error(y_test, y_baseline)
base_medae = median_absolute_error(y_test, y_baseline)
base_corr = pearsonr(y_baseline, y_test)[0][0]
logger.info(f'{model_name} MedianAE on testing data: {medae} | Baseline: {base_medae}.')
logger.info(f'{model_name} MAE on testing data: {mae} | Baseline: {base_mae}.')
logger.info(f'{model_name} MSE on testing data: {rmse} | Baseline: {base_rmse}.')
logger.info(f'{model_name} Pearson on testing data: {corr} | Baseline: {base_corr}.')




