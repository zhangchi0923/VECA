
## The VR Eye-tracking Cognitive Assessment (VECA): A Portable and Efficient Dementia Screening Tool Using Eye-Tracking Technology, Machine Learning, and Virtual Reality 

Programs of building and evaluating models mentioned in this paper were developed using Python 3.8.
```
# install other required packages
pip install -r requirements.txt
```

### About data

Unreal eye tracking series and the corresponding info data were randomly generated in `data` folder only to run the code.

*Results in the paper could be reproduced only via original patient data. Non-identifiable patient data are available upon request to the corresponding author.*

### How to use

This repository provides a pipeline of eye tracking data processing, feature extraction, modeling and evaluation based on the framework of VECA. The components or configuration such as `/src/utils/eyeMovement.py` and `/src/config/settings.py` could be customized for other research.
```shell
# train a specific model
python train.py -m gbrt

# train a specific model and output feature importance
python train.py -m svr --importance

# train a specific model and output shap value analysis
python train.py -m gbrt --shap
```
```shell
# check all options
python train.py -h

usage: train.py [-h] [-m {svr,mlp,gbrt,lasso}] [--info_dir INFO_DIR] [--log_dir LOG_DIR] [--model_dir MODEL_DIR] [--importance] [--shap] [--roc]

optional arguments:
  -h, --help            show this help message and exit
  -m {svr,mlp,gbrt,lasso}, --model {svr,mlp,gbrt,lasso}
                        Specify which ML model to train and evaluate.
  --info_dir INFO_DIR   Specify file path of data info excel of training data.
  --log_dir LOG_DIR     Specify directory path of log files.
  --model_dir MODEL_DIR
                        Specify model persistent directory.
  --importance          Specify whether to output feature importances.
  --shap                Specify whether to compute normalized shap values.
  --roc                 Specify whether to analyze education grouped classification ROCs.
```