from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import os
import sys
import joblib
from src.logger import logging
from src.exception import customexception
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion

@dataclass
class DataTransformationConfig:
    transformed_train_data_path: str = os.path.join("artifacts", "transformed_train.csv")
    transformed_test_data_path: str = os.path.join("artifacts", "transformed_test.csv")
    preprocessor_obj_path: str = os.path.join("artifacts", "preprocessor.pkl")  

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def display_label_encoder_mappings(self, label_encoders):
        for feature, encoder in label_encoders.items():
            logging.info(f"Label Encoder mappings for {feature}:")
            for class_, index in zip(encoder.classes_, encoder.transform(encoder.classes_)):
                logging.info(f"{class_} -> {index}")


    def initiate_data_transformation(self, train_data_path: str, test_data_path: str):
        logging.info("Data transformation started")
        
        try:
            # Load train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Loaded train and test data for transformation")

            # Drop rows with missing values
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)
            logging.info("Dropped rows with missing values")
            
            # Drop 'Route' and 'Additional_Info' columns
            train_df.drop(['Route', 'Additional_Info'], axis=1, inplace=True)
            test_df.drop(['Route', 'Additional_Info'], axis=1, inplace=True)

            # Extract day, month from Date_of_Journey
            train_df['Journey_Day'] = pd.to_datetime(train_df['Date_of_Journey'], format='%d/%m/%Y').dt.day
            train_df['Journey_Month'] = pd.to_datetime(train_df['Date_of_Journey'], format='%d/%m/%Y').dt.month
            test_df['Journey_Day'] = pd.to_datetime(test_df['Date_of_Journey'], format='%d/%m/%Y').dt.day
            test_df['Journey_Month'] = pd.to_datetime(test_df['Date_of_Journey'], format='%d/%m/%Y').dt.month
            
            # Drop the 'Date_of_Journey' column
            train_df.drop(['Date_of_Journey'], axis=1, inplace=True)
            test_df.drop(['Date_of_Journey'], axis=1, inplace=True)

            # Convert 'Duration' to minutes
            def convert_to_minutes(duration):
                try:
                    hours = 0
                    minutes = 0
                    if 'h' in duration:
                        hours = int(duration.split('h')[0].strip())
                        duration = duration.split('h')[1].strip()
                    if 'm' in duration:
                        minutes = int(duration.split('m')[0].strip())
                    return hours * 60 + minutes
                except:
                    return None  # Or a default value if needed

            train_df['Duration'] = train_df['Duration'].apply(convert_to_minutes)
            test_df['Duration'] = test_df['Duration'].apply(convert_to_minutes)
            
            train_df.drop(['Dep_Time', 'Arrival_Time'], axis=1, inplace=True)
            test_df.drop(['Dep_Time', 'Arrival_Time'], axis=1, inplace=True)

            train_df['Total_Stops'] = train_df['Total_Stops'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})
            test_df['Total_Stops'] = test_df['Total_Stops'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})


            categorical_features = ['Airline', 'Source', 'Destination']
            
            label_encoders = {}
            for feature in categorical_features:
                label_encoder = LabelEncoder()
                train_df[feature] = label_encoder.fit_transform(train_df[feature])
                test_df[feature] = label_encoder.transform(test_df[feature])
                label_encoders[feature] = label_encoder
            
            self.display_label_encoder_mappings(label_encoders)
            
            scaler = StandardScaler()
            numerical_features = ['Duration', 'Journey_Day', 'Journey_Month', 'Total_Stops']
            train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
            test_df[numerical_features] = scaler.transform(test_df[numerical_features])
            
            preprocessor = {
                'label_encoders': label_encoders,
                'scaler': scaler
            }
            os.makedirs(os.path.dirname(self.transformation_config.preprocessor_obj_path), exist_ok=True)
            joblib.dump(preprocessor, self.transformation_config.preprocessor_obj_path)
            logging.info("Preprocessor (LabelEncoders and scaler) saved successfully")

            os.makedirs(os.path.dirname(self.transformation_config.transformed_train_data_path), exist_ok=True)
            train_df.to_csv(self.transformation_config.transformed_train_data_path, index=False)
            test_df.to_csv(self.transformation_config.transformed_test_data_path, index=False)
            logging.info("Data transformation completed and saved to CSV")

            return (
                self.transformation_config.transformed_train_data_path,
                self.transformation_config.transformed_test_data_path,
                self.transformation_config.preprocessor_obj_path
            )
        
        except Exception as e:
            logging.error("Exception occurred in initiate_data_transformation")
            raise customexception(e, sys)
