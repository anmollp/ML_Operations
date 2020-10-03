import pandas as pd
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
import sys
PARENT_FOLDER = str(os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0])
sys.path.insert(0, PARENT_FOLDER)
ROOT_FOLDER = PARENT_FOLDER.rsplit('/', 1)[0]
from utility.MysqlConnector import MySQLConnector
from config.log_config import pipeline_logger
from config.db_config import mysql_input_connection

my_file = os.path.join(THIS_FOLDER, 'DataSet/Summary of Weather.csv')

dataset = pd.read_csv(my_file, low_memory=False)

chunk_size = 5000

mysql = MySQLConnector(mysql_input_connection['local'], 'mysql')
mysql.create_connection()

for i in range(1, 119041, chunk_size):
    pipeline_logger.info("Inserting chunk # {}".format(i))
    partition = dataset[i:i+chunk_size]
    partition.to_sql('weather', mysql.conn, index=False, if_exists='replace', method='multi')
