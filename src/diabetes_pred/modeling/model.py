from xgboost import XGBClassifier

def build_model(params=None):
    """
    Build and return an XGBoost model with the given parameters.

    Args:
        params (dict, optional): A dictionary of parameters to configure the XGBoost model.
                                 If None, default parameters will be used.

    Returns:
        xgboost.XGBClassifier: An instance of the XGBoost classifier.
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
    
    model = XGBClassifier(**params)
    return model