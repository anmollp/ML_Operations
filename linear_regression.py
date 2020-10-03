import pandas as pd
import numpy as np
import os
import sys
PARENT_FOLDER = str(os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0])
sys.path.insert(0, PARENT_FOLDER)
ROOT_FOLDER = PARENT_FOLDER.rsplit('/', 1)[0]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from utility.MysqlConnector import MySQLConnector
from utility.Decorators import Decorator
from config.log_config import pipeline_logger
from config.db_config import mysql_input_connection, mysql_output_connection
import random


class LinearReg:
    deco = Decorator(pipeline_logger)
    
    def __init__(self):
        self.mysql_config_input = mysql_input_connection['local']
        self.mysql_config_output = mysql_output_connection['local']
        self.actual_v_predict_df = None
        self.regressor_model = None
        self.count = 0
        self.offset = 0
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    @deco.log_it
    def predict(self, batch_size, train_set_percent):
        """
        Max Temperature prediction function given Min Temperature
        :param batch_size: batch size of data for modelling and prediction.
        :param train_set_percent: Training set percentage.
        :return: A dataframe with actual and predicted values.
        """
        input_connection = MySQLConnector(self.mysql_config_input, 'mysql')
        output_connection = MySQLConnector(self.mysql_config_output, 'mysql')
        self.make_train_test_split(batch_size, train_set_percent, connection=input_connection)
        regressor_model = self.get_trained_model()
        y_pred = regressor_model.predict(self.x_test)
        self.actual_v_predict_df = pd.DataFrame({'Actual': self.y_test.flatten(), 'Predicted': y_pred.flatten()})
        self.save_prediction_to_db(batch_size, self.get_mean_abs_error(self.y_test, y_pred),
                                   self.get_mean_squared_error(self.y_test, y_pred),
                                   self.get_root_mean_squared_error(self.y_test, y_pred), connection=output_connection)
        return self.actual_v_predict_df

    @deco.log_it
    @deco.db_connect
    def get_training_data_for_real_time_prediction(self, connection):
        """
        Function to retrieve training data from database.
        :param connection: Database connection, belongs to training database.
        :return: MinTemperature and MaxTemperature as features array.
        """
        train_data = connection.get_dataframe_from_select_query("select MaxTemp, MinTemp from weather")
        x_train, y_train = train_data["MinTemp"].to_list(), train_data["MaxTemp"].to_list()
        return np.expand_dims(x_train, -1), np.expand_dims(y_train, -1)
        
    @deco.log_it
    def predict_real_time_with_metrics(self, json_data_list):
        """
        Function to describe prediction and metrics.
        :param json_data_list: Input dataset as real time data
        :return: Prediction and metrics dataframes.
        """
        test_data = pd.DataFrame.from_records(json_data_list)
        self.x_test = test_data['MinTemp'].values.reshape(-1, 1)
        self.y_test = test_data['MaxTemp'].values.reshape(-1, 1)
        connection = MySQLConnector(self.mysql_config_input['local'], 'mysql')
        self.x_train, self.y_train = self.get_training_data_for_real_time_prediction(connection=connection)
        regressor_model = self.get_trained_model()
        y_pred = regressor_model.predict(self.x_test)
        self.actual_v_predict_df = pd.DataFrame({'Actual': self.y_test.flatten(), 'Predicted': y_pred.flatten()})
        mae = self.get_mean_abs_error(self.y_test, y_pred)
        mse = self.get_mean_squared_error(self.y_test, y_pred)
        rmse = self.get_root_mean_squared_error(self.y_test, y_pred)
        accuracy_df = pd.DataFrame.from_records([{"Mean Absolute Error": mae, "Mean Squared Error": mse,
                                                 "Root Mean Squared Error": rmse}])
        return self.actual_v_predict_df, accuracy_df
        
    @deco.log_it
    def get_trained_model(self):
        """
        Function that returns a trained linear regression model.
        :return: Linear regression model.
        """
        self.regressor_model = LinearRegression()
        self.regressor_model.fit(self.x_train, self.y_train)  # training the algorithm
        return self.regressor_model

    @deco.log_it
    @deco.db_connect
    def make_train_test_split(self, batch_size, train_set_percent, connection):
        """
        A test, train data splitter function.
        :param batch_size: Batch size of the computational data
        :param train_set_percent: Percentage of training data to be considered.
        :param connection: database connection to input dataset.
        :return: None
        """
        self.count = connection.execute_select_raw_sql_query_as_dict("select count(*) as count from weather")
        self.offset = random.randint(0, self.count[0].get("count"))
        rows = connection.execute_select_raw_sql_query_as_dict(
            "select * from weather limit {} offset {}".format(batch_size, self.offset))
        dataset = connection.convert_to_data_frame(rows)
        x = dataset['MinTemp'].values.reshape(-1, 1)
        y = dataset['MaxTemp'].values.reshape(-1, 1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=float(
            int(train_set_percent) / 100),
                                                                                random_state=0)

    @deco.log_it
    def get_actual_v_predicted_df(self):
        """
        A vs dataframe.
        :return: Dataframe containing actual vs predicted max temperatures.
        """
        return self.actual_v_predict_df

    @deco.log_it
    def get_count(self):
        """
        Function to calculate database records count
        :return: count
        """
        return self.count

    @deco.log_it
    def get_offset(self):
        """
        Function to return random offset generated.
        :return: offset
        """
        return self.offset

    @deco.log_it
    def get_intercept(self):
        return self.regressor_model.intercept_

    @deco.log_it
    def get_coeffecient(self):
        return self.regressor_model.coef_
    
    @staticmethod
    def get_mean_abs_error(y_test, y_pred):
        return metrics.mean_absolute_error(y_test, y_pred)

    @staticmethod
    def get_mean_squared_error(y_test, y_pred):
        return metrics.mean_squared_error(y_test, y_pred)

    @staticmethod
    def get_root_mean_squared_error(y_test, y_pred):
        return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        
    @staticmethod
    def generate_batch_result_identifier(batch_size, offset):
        """
        An identifier for every batch modelled.
        :param batch_size: input batch size of model
        :param offset: random offset to randomise data picking.
        :return: Identifier
        """
        prime = 31
        result_id = int(offset)*prime + int(batch_size)
        return result_id
    
    @deco.log_it
    @deco.db_connect
    def save_prediction_to_db(self, size, mar, mse, rms, connection):
        """
        Save a model's predictions and metrics to output database.
        :param size: Batch size of data used for prediction.
        :param mar: Mean Absolute Error
        :param mse: Mean Squared Error
        :param rms: Root Mean Squared Error
        :param connection: database connection to output data.
        :return: None
        """
        pkey = self.generate_batch_result_identifier(size, self.offset)
        query = "INSERT IGNORE INTO weather_models(id, size, offset, intercept, coefficient, mean_absolute_error, " \
                "mean_squared_error, root_mean_squared_error) values({}, {}, {}, {}, {}, {}, {}, {})".format(pkey, size,
                self.offset, self.get_intercept()[0], self.get_coeffecient()[0][0], mar, mse, rms)
        connection.execute_commit_raw_sql_query(query)
        

