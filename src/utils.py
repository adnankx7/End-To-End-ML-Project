import os 
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import dill
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        
        # Correcting the iteration over the models dictionary
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_pred_train)  # Correcting the order of arguments
            test_model_score = r2_score(y_test, y_pred_test)     # Correcting the order of arguments
            
            report[model_name] = {'Train R2 Score': train_model_score, 'Test R2 Score': test_model_score}
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
