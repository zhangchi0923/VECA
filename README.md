## About data

Unreal eye tracking series and the corresponding info data were randomly generated in `data` folder only to run the code.

*Results in the paper could be reproduced only via original patient data. Non-identifiable patient data are available upon request to the corresponding author.*

## How to use

```python
# train a specific model
python train.py --model gbrt

# train a specific model while output feature importance
python train.py --model svr --importance

# train a specific model while output shap value analysis
python train.py --model gbrt --shap
```