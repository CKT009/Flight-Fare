import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import sys
from src.logger import logging
from src.exception import customexception
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion
import re

@dataclass
class DataTransformationConfig:
    transformed_train_data_path: str = os.path.join("artifacts", "transformed_train.csv")
    transformed_test_data_path: str = os.path.join("artifacts", "transformed_test.csv")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_data_path: str, test_data_path: str):
        logging.info("Data transformation started")
        
        try:
            # Load train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Loaded train and test data for transformation")

            # Feature Engineering
            
            # Drop rows with missing values
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)
            logging.info("Dropped rows with missing values")
            logging.info(f"total length of train: {len(train_df)}")
            
            # Drop 'Route' column as it's not needed
            train_df.drop(['Route'], axis=1, inplace=True)
            test_df.drop(['Route'], axis=1, inplace=True)

            # Handle 'Additional_Info' column by dropping it as it's not needed
            train_df.drop(['Additional_Info'], axis=1, inplace=True)
            test_df.drop(['Additional_Info'], axis=1, inplace=True)

            # Extract day, month from Date_of_Journey
            train_df['Journey_Day'] = pd.to_datetime(train_df['Date_of_Journey'], format='%d/%m/%Y').dt.day
            train_df['Journey_Month'] = pd.to_datetime(train_df['Date_of_Journey'], format='%d/%m/%Y').dt.month
            test_df['Journey_Day'] = pd.to_datetime(test_df['Date_of_Journey'], format='%d/%m/%Y').dt.day
            test_df['Journey_Month'] = pd.to_datetime(test_df['Date_of_Journey'], format='%d/%m/%Y').dt.month
            
            # Drop the 'Date_of_Journey' column as it's now redundant
            train_df.drop(['Date_of_Journey'], axis=1, inplace=True)
            test_df.drop(['Date_of_Journey'], axis=1, inplace=True)

            # Use 'Duration' directly instead of extracting hours and minutes
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
            
            # Drop 'Departure_Time' and 'Arrival_Time' columns
            train_df.drop(['Dep_Time', 'Arrival_Time'], axis=1, inplace=True)
            test_df.drop(['Dep_Time', 'Arrival_Time'], axis=1, inplace=True)

            # Handle 'Total_Stops' as numeric value
            train_df['Total_Stops'] = train_df['Total_Stops'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})
            test_df['Total_Stops'] = test_df['Total_Stops'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})

            # Handle categorical features using pd.get_dummies
            categorical_features = ['Airline', 'Source', 'Destination']

            # Train data
            Airline_train = pd.get_dummies(train_df[['Airline']], drop_first=True, dtype=int)
            Source_train = pd.get_dummies(train_df[['Source']], drop_first=True, dtype=int)
            Destination_train = pd.get_dummies(train_df[['Destination']], drop_first=True, dtype=int)
            
            # Test data
            Airline_test = pd.get_dummies(test_df[['Airline']], drop_first=True, dtype=int)
            Source_test = pd.get_dummies(test_df[['Source']], drop_first=True, dtype=int)
            Destination_test = pd.get_dummies(test_df[['Destination']], drop_first=True, dtype=int)
            
            # Align the columns of train and test data
            combined_columns = list(set(Airline_train.columns) | set(Airline_test.columns))
            Airline_train = Airline_train.reindex(columns=combined_columns, fill_value=0)
            Airline_test = Airline_test.reindex(columns=combined_columns, fill_value=0)
            Source_train = Source_train.reindex(columns=combined_columns, fill_value=0)
            Source_test = Source_test.reindex(columns=combined_columns, fill_value=0)
            Destination_train = Destination_train.reindex(columns=combined_columns, fill_value=0)
            Destination_test = Destination_test.reindex(columns=combined_columns, fill_value=0)

            # Combine encoded features with original features
            train_df = pd.concat([train_df.drop(categorical_features, axis=1), Airline_train, Source_train, Destination_train], axis=1)
            test_df = pd.concat([test_df.drop(categorical_features, axis=1), Airline_test, Source_test, Destination_test], axis=1)
            logging.info(f"total nan values: {train_df.isna().sum()}")

            # Standardizing the numerical features
            scaler = StandardScaler()
            numerical_features = ['Duration', 'Journey_Day', 'Journey_Month', 'Total_Stops']
            train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
            test_df[numerical_features] = scaler.transform(test_df[numerical_features])
            logging.info(f"{train_df.sample(3)}")
            # Save transformed data
            os.makedirs(os.path.dirname(self.transformation_config.transformed_train_data_path), exist_ok=True)
            train_df.to_csv(self.transformation_config.transformed_train_data_path, index=False)
            test_df.to_csv(self.transformation_config.transformed_test_data_path, index=False)
            logging.info("Data transformation completed and saved to CSV")

            return (
                self.transformation_config.transformed_train_data_path,
                self.transformation_config.transformed_test_data_path
            )
        
        except Exception as e:
            logging.error("Exception occurred in initiate_data_transformation")
            raise customexception(e, sys)

if __name__ == '__main__':
    obj = DataTransformation()
    train_data_path, test_data_path = DataIngestion().initiate_data_ingestion()
    obj.initiate_data_transformation(train_data_path, test_data_path)
