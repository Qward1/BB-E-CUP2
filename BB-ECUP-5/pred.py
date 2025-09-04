import numpy as np
import joblib


def ensemble_predict_proba(X):

    model_paths = {
        "catboost_seed_42": ".\\models\\model1.pkl",
        "catboost_seed_123": ".\\models\\model2.pkl",
        "catboost_seed_456": ".\\models\\model3.pkl",
        "catboost_seed_789": ".\\models\\model4.pkl",
        "catboost_seed_2024": ".\\models\\model5.pkl",
    }

    models = {name: joblib.load(path) for name, path in model_paths.items()}
    weights = {'catboost_seed_42': 0.2,'catboost_seed_123': 0.2,'catboost_seed_456': 0.2,'catboost_seed_789': 0.2,'catboost_seed_2024': 0.2}

    predictions = np.zeros(len(X))

    for name, model in models.items():
        preds = model.predict_proba(X)[:, 1]
        predictions += weights[name] * preds

    return predictions