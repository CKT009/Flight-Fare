import os
import sys
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

            # Step 1: Data Ingestion
            logging.info("Step 1: Data Ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            logging.info("Step 2: Data Transformation")
            data_transformation = DataTransformation()
            transformed_train_data_path, transformed_test_data_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )

            # Step 3: Model Training
            logging.info("Step 3: Model Training")
            model_trainer = ModelTrainer()
            best_model_path = model_trainer.initiate_model_training(transformed_train_data_path, transformed_test_data_path)

            logging.info(f"Training pipeline completed successfully. Best model saved at: {best_model_path}")

        except Exception as e:
            logging.error("An error occurred in the training pipeline")
            raise customexception(e, sys)


if __name__ == "__main__":
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Exception occurred while running the pipeline: {e}")
        raise customexception(e, sys)
