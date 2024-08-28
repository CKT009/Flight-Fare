import pandas as pd
import joblib
from src.logger import logging
from src.exception import customexception
import sys

class PredictionPipeline:
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = self.load_model()
        self.preprocessor = self.load_preprocessor()

    def load_model(self):
        try:
            model = joblib.load(self.model_path)
            assert hasattr(model, 'predict'), "Model is not in a state to predict"
            logging.info("Model loaded successfully")
            return model
        except Exception as e:
            logging.error("Error loading model")
            raise customexception("Error loading model", sys)

    def load_preprocessor(self):
        try:
            preprocessor = joblib.load(self.preprocessor_path)
            logging.info("Preprocessor loaded successfully")
            return preprocessor
        except Exception as e:
            logging.error("Error loading preprocessor")
            raise customexception("Error loading preprocessor", sys)

    def preprocess_input(self, input_data: pd.DataFrame):
        try:

            required_columns = ['date_of_journey', 'Duration', 'Total_Stops', 'Airline', 'Source', 'Destination']
            for col in required_columns:
                if col not in input_data.columns:
                    raise ValueError(f"Missing column in input data: {col}")

            input_data['Journey_Day'] = pd.to_datetime(input_data['date_of_journey'], format='%Y-%m-%d', errors='coerce').dt.day
            input_data['Journey_Month'] = pd.to_datetime(input_data['date_of_journey'], format='%Y-%m-%d', errors='coerce').dt.month
            input_data.drop(columns='date_of_journey', axis=1, inplace=True)
            
            def convert_to_minutes(duration):
                try:
                    if pd.isna(duration):
                        return None
                    hours = 0
                    minutes = 0
                    if 'h' in duration:
                        hours = int(duration.split('h')[0].strip())
                        duration = duration.split('h')[1].strip()
                    if 'm' in duration:
                        minutes = int(duration.split('m')[0].strip())
                    return hours * 60 + minutes
                except:
                    return None

            input_data['Duration'] = input_data['Duration']
            input_data['Total_Stops'] = input_data['Total_Stops'].replace({
                'non-stop': 0, 
                '1 stop': 1, 
                '2 stops': 2, 
                '3 stops': 3, 
                '4 stops': 4
            })            

            numerical_features = ['Duration', 'Journey_Day', 'Journey_Month', 'Total_Stops']
            input_data[numerical_features] = self.preprocessor['scaler'].transform(input_data[numerical_features])
            logging.info("numerical done")
            
            categorical_features = ['Airline', 'Source', 'Destination']
            for feature in categorical_features:
                if feature in self.preprocessor['label_encoders']:
                    label_encoder = self.preprocessor['label_encoders'][feature]
                    input_data[feature] = label_encoder.transform(input_data[feature])
                else:
                    raise ValueError(f"Missing LabelEncoder for feature: {feature}")

            return input_data
        except Exception as e:
            logging.error("Error in preprocessing input data")
            raise customexception("Error in preprocessing input data", sys)

    def predict(self, input_data: dict):
        try:
            input_df = pd.DataFrame([input_data])
            logging.info(f"Input data for prediction: {input_df}")
            
            processed_data = self.preprocess_input(input_df)
            logging.info(f"Preprocessed data: {processed_data}")
            
            if not hasattr(self.model, 'predict'):
                raise ValueError("Loaded model is not callable or does not have a predict method.")
            
            logging.info(f"Using model of type {type(self.model)} for prediction")
            
            prediction = self.model.predict(processed_data)
            logging.info(f"Prediction: {prediction}")
            
            return prediction
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise customexception("Error during prediction", sys)
