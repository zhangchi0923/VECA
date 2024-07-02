import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
from utils.eyeMovement import DataUtils, EyeMovement
from config.settings import *
import warnings
warnings.filterwarnings('ignore')

fix_base_cat = ['abs', 'calc4', 'calc5', 'calc6', 'exec', 'mem8', 'mem9', 'mem10', 'recall']
fix_eye_cat = ['_aoi_ratio']
fix_feat_name = ['att'] + [x[0] + x[1] for x in itertools.product(fix_base_cat, fix_eye_cat)]

def get_train_data(info_path: str):
    info = pd.read_excel(info_path, index_col=0)

    levels = list(range(3, 12, 1))
    for i in range(info.shape[0]):
        # rocket
        url = info.iloc[i]["resources_url"]
        id = info.iloc[i]['id']
        gaze_data = pd.read_csv(url)
        util = DataUtils(gaze_data)
        x, y, time = util.get_lvl_state(util.prepare_data(), 2, 2)
        detector_l2 = EyeMovement(x, y, time, AOIs, BEZIER_POINTS)
        att = detector_l2.measureFollowRate()
        feats = [att]
        # other
        for level in levels:
            x, y, time = util.get_lvl_state(util.prepare_data(), level, 2)
            detector = EyeMovement(x, y, time, AOIs[level], BEZIER_POINTS)
            fix_data = detector.eye_movements_detector(x, y, time)
            _, _, merged = detector.merge_fixation(fix_data)
            feats.append(detector.AOI_fixation_ratio(merged))
        info.loc[info["id"] == id, fix_feat_name] = feats

    info.drop(['resources_url', 'id'], axis=1, inplace=True)
    return info


def feature_preprocess(train_data: pd.DataFrame, normalize=False):
    data = train_data.copy()
    cat_cols = ['gender_F', 'gender_M', 'edu_0', 'edu_1', 'edu_2']
    num_cols = fix_feat_name
    label_col = ['moca']

    data = pd.get_dummies(data, prefix_sep='_', columns=['gender', 'edu'])
    X, y = data[cat_cols + num_cols], data[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values
    
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train.astype(float).flatten(), y_test.astype(float).flatten(), np.array(cat_cols + num_cols)

def diag(row):
    if row['edu_0'] == 1:
        return row['moca'] < 14
    if row['edu_1'] == 1:
        return row['moca'] < 20
    return row['moca'] < 25