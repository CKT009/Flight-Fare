import pandas as pd
import os
import sys
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from dataclasses import dataclass
from src.logger import logging
from src.exception import customexception
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "best_model.pkl")
    model_scores_file_path: str = os.path.join("artifacts", "model_scores.csv")

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def fine_tune_model(self, model, param_grid, X_train, y_train):
        # Initialize GridSearchCV for each model
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        logging.info(f"Best parameters for {type(model).__name__}: {grid_search.best_params_}")

        return grid_search.best_estimator_

    def initiate_model_training(self, train_data_path: str, test_data_path: str):
        logging.info("Model training started")

        try:
            # Load train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            # Separate features and target
            X_train = train_df.drop(columns=['Price'])
            y_train = train_df['Price']
            X_test = test_df.drop(columns=['Price'])
            y_test = test_df['Price']

            # Initialize models and their hyperparameter grids
            models = {
                "Linear Regression": (LinearRegression(), {}),
                "Decision Tree": (DecisionTreeRegressor(), {
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10]
                }),
                "Random Forest": (RandomForestRegressor(), {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10]
                }),
                "XGBoost": (XGBRegressor(), {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }),
                "AdaBoost Regressor": (AdaBoostRegressor(), {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                }),
                "KNN Regressor": (KNeighborsRegressor(), {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                })
            }
            model_performance = []

            # Train, fine-tune, and evaluate models
            for model_name, (model, param_grid) in models.items():
                logging.info(f"Training and tuning {model_name}")

                if param_grid:  # Only fine-tune models with a defined parameter grid
                    model = self.fine_tune_model(model, param_grid, X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                # Predict on test data
                y_pred = model.predict(X_test)

                # Evaluate model
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                # Append model performance
                model_performance.append({
                    "Model": model_name,
                    "RMSE": rmse,
                    "R2": r2
                })

                logging.info(f"{model_name} - RMSE: {rmse}, R2: {r2}")

            performance_df = pd.DataFrame(model_performance)
            os.makedirs(os.path.dirname(self.trainer_config.model_scores_file_path), exist_ok=True)
            performance_df.to_csv(self.trainer_config.model_scores_file_path, index=False)  # Ensure 'Model' column is saved

            best_model_name = performance_df.loc[performance_df['R2'].idxmax(), 'Model']
            best_model = models[best_model_name][0]  # Get the best model instance after tuning

            logging.info(f"Best model selected: {best_model_name} with R2: {performance_df.loc[performance_df['R2'].idxmax(), 'R2']}")
            print(f"Best model name: {best_model_name} and R2 score: {performance_df.loc[performance_df['R2'].idxmax(), 'R2']}")

            os.makedirs(os.path.dirname(self.trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(best_model, self.trainer_config.trained_model_file_path)

            logging.info("Model training completed and best model saved")

            return self.trainer_config.trained_model_file_path

        except Exception as e:
            logging.error("Exception occurred in initiate_model_training")
            raise customexception(e, sys)
