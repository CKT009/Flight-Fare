import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import customexception
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Starting the training pipeline")

            logging.info("Step 1: Data Ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            logging.info("Step 2: Data Transformation")
            data_transformation = DataTransformation()
            transformed_train_data_path, transformed_test_data_path, _ = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )

            logging.info("Step 3: Model Training")
            model_trainer = ModelTrainer()
            best_model_path = model_trainer.initiate_model_training(transformed_train_data_path, transformed_test_data_path)

            model_scores_df = pd.read_csv(model_trainer.trainer_config.model_scores_file_path)
            print("Model Scores DataFrame:")
            print(model_scores_df.head())

            if 'Model' not in model_scores_df.columns:
                raise ValueError("Column 'Model' is missing from the model scores DataFrame")

            best_model_name = model_scores_df.loc[model_scores_df['R2'].idxmax(), 'Model']

            logging.info(f"Training pipeline completed successfully. Best model saved at: {best_model_path}")

            return model_scores_df.to_dict(orient='records'), best_model_name

        except Exception as e:
            logging.error("An error occurred in the training pipeline")
            raise customexception(e, sys)