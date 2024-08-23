import pandas as pd
import os
import sys
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from dataclasses import dataclass
from src.logger import logging
from src.exception import customexception
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "best_model.pkl")
    model_scores_file_path: str = os.path.join("artifacts", "model_scores.csv")

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

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

            # Initialize models
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBoost": XGBRegressor(objective='reg:squarederror')
            }

            model_performance = {}

            # Train and evaluate models
            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                model.fit(X_train, y_train)

                # Predict on test data
                y_pred = model.predict(X_test)

                # Evaluate model
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                model_performance[model_name] = {
                    "RMSE": rmse,
                    "R2": r2
                }

                logging.info(f"{model_name} - RMSE: {rmse}, R2: {r2}")

            # Convert model performance to DataFrame and save to CSV
            performance_df = pd.DataFrame(model_performance).T
            os.makedirs(os.path.dirname(self.trainer_config.model_scores_file_path), exist_ok=True)
            performance_df.to_csv(self.trainer_config.model_scores_file_path)

            # Select the best model
            best_model_name = performance_df['R2'].idxmax()
            best_model = models[best_model_name]

            logging.info(f"Best model selected: {best_model_name} with R2: {performance_df.loc[best_model_name, 'R2']}")

            # Save the best model
            os.makedirs(os.path.dirname(self.trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(best_model, self.trainer_config.trained_model_file_path)

            logging.info("Model training completed and best model saved")

            return self.trainer_config.trained_model_file_path

        except Exception as e:
            logging.error("Exception occurred in initiate_model_training")
            raise customexception(e, sys)

if __name__ == "__main__":
    data_transformation = DataTransformation()
    train_data_path, test_data_path = DataIngestion().initiate_data_ingestion()
    transformed_train_data_path, transformed_test_data_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(transformed_train_data_path, transformed_test_data_path)
