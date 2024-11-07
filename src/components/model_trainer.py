import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            
            # Correct the splitting of X_train, y_train, X_test, and y_test
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],  # All rows, all columns except last (features)
                test_array[:, :-1],   # All rows, all columns except last (features)
                train_array[:, -1],   # All rows, last column (target)
                test_array[:, -1],    # All rows, last column (target)
            )
            
            # Define models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            # Evaluate models and get the report
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            logging.info("Training model")
            
            # Get the best model from the report by comparing Test R2 Scores
            best_model_score = max(model_report.values(), key=lambda x: x['Test R2 Score'])
            best_model_name = [name for name, report in model_report.items() if report == best_model_score][0]
            
            logging.info(f"Best Model: {best_model_name}")
            
            best_model = models[best_model_name]
            
            # Check if the model's performance is satisfactory
            if best_model_score['Test R2 Score'] < 0.6:
                raise CustomException("Model performance is not satisfactory")
            
            logging.info("Best model found for both train and test data")
            
            # Save the trained best model to the specified path
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Use the best model to make predictions
            predicted = best_model.predict(X_test)
            
            # Return R2 score of the model
            return r2_score(y_test, predicted)
        
        except Exception as e:
            raise CustomException(e, sys)
