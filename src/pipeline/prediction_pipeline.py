# src/pipeline/prediction_pipeline.py

import pandas as pd
import joblib
import os
import sys
from src.exception import customexception
from src.logger import logging

class PredictionPipeline:
    def __init__(self, preprocessor_path, model_path):
        self.preprocessor_path = preprocessor_path
        self.model_path = model_path
        
        self.preprocessor = self.load_object(self.preprocessor_path)
        self.model = self.load_object(self.model_path)

    def load_object(self, file_path):
        """
        Utility function to load a saved object using joblib
        """
        try:
            return joblib.load(file_path)
        except Exception as e:
            raise customexception(f"Error loading object from path: {file_path} - {e}", sys)

    def preprocess_input(self, input_df):
        """
        Preprocess the input dataframe as per the training pipeline
        """
        try:
            
            input_df['Journey_Day'] = pd.to_datetime(input_df['Date_of_Journey'], format='%d/%m/%Y').dt.day
            input_df['Journey_Month'] = pd.to_datetime(input_df['Date_of_Journey'], format='%d/%m/%Y').dt.month

            # Convert 'Duration' to minutes
            def convert_to_minutes(duration):
                hours = 0
                minutes = 0
                if 'h' in duration:
                    hours = int(duration.split('h')[0].strip())
                    duration = duration.split('h')[1].strip()
                if 'm' in duration:
                    minutes = int(duration.split('m')[0].strip())
                return hours * 60 + minutes

            input_df['Duration'] = input_df['Duration'].apply(convert_to_minutes)
            
            # Dropping unnecessary columns
            
            # Encode categorical features as in training
            input_df['Total_Stops'] = input_df['Total_Stops'].replace({
                'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4
            })

            # Handle categorical features using pd.get_dummies
            categorical_features = ['Airline', 'Source', 'Destination']
            input_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)
            
            # Align columns with training set columns
            input_df = input_df.reindex(columns=self.preprocessor.feature_names_in_, fill_value=0)

            # Scale the numerical features
            numerical_features = ['Duration', 'Journey_Day', 'Journey_Month', 'Total_Stops']
            input_df[numerical_features] = self.preprocessor.transform(input_df[numerical_features])
            
            return input_df

        except Exception as e:
            logging.error(f"Error during preprocessing: {str(e)}")
            raise customexception(e, sys)

    def predict(self, input_data):
        """
        Make predictions on input data after preprocessing
        """
        try:
            # Preprocess the input data
            processed_data = self.preprocess_input(input_data)

            # Predict using the loaded model
            predictions = self.model.predict(processed_data)

            return predictions

        except Exception as e:
            logging.error(f"Exception occurred in prediction: {str(e)}")
            raise customexception(e, sys)

