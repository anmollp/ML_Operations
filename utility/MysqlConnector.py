import sqlalchemy as db
import pandas as pd


class MySQLConnector:
    def __init__(self, db_conf, connection_type):
        host, port, db_name, user, password = db_conf['host'], db_conf['port'], db_conf['database'], db_conf[
            'user_name'], db_conf['password']
        self.connect_string = ''
        if 'host' not in db_conf or 'port' not in db_conf or 'database' not in db_conf or 'user_name' not in db_conf or 'password' not in db_conf:
            raise KeyError('db_conf should have keys --> host, port, database, user_name, password!')
        
        if connection_type == 'mysql':
            self.connect_string = "mysql+pymysql://{user}:{password}@{host}/{dbname}" \
                .format(user=user, password=password, host=host, dbname=db_name)
        
        self.engine = None
        self.conn = None
        self.metadata = None

    def create_connection(self):
        """
        throws sqlalchemy.exc.SQLAlchemyError --> if wrong connection string / wrong host name / authentication failure / database doesn't exist
        :return: None, creates connection object upon success, else Exception
        """
        if self.conn is None:
            self.engine = db.create_engine(self.connect_string, pool_recycle=100, pool_pre_ping=True)
            self.conn = self.engine.connect()
            self.metadata = db.MetaData()

    @staticmethod
    def convert_to_data_frame(result):
        """"
        :param result: List of dictionaries as returned by execute_select_raw_sql_query_as_dict method
        :return: Dataframe object comprising the result
        """
        df = pd.DataFrame([i.values() for i in result])
        df.columns = result[0].keys()
        return df

    def get_dataframe_from_select_query(self, query):
        return pd.read_sql(query, con=self.conn)

    def execute_select_raw_sql_query_as_dict(self, query):
        """
        Throws AttributeError --> When conn object is None (make sure connection is established before running this method)
        Throws sqlalchemy.exc.SQLAlchemyError --> if something wrong in query
        :return: List of dicts, each of which denotes each row in key/value form (keys represent columns)
        """
        if self.conn is None:
            raise AttributeError('Connection is None -- None type doesnt have any cursor!')
    
        execute_obj = self.conn.execute(query)
        result_set = execute_obj.fetchall()
        columns = execute_obj.keys()
        list_of_dict = [[(columns[i], result[i]) for i in range(len(columns))] for result in result_set]
        return [dict(l) for l in list_of_dict]

    def execute_commit_raw_sql_query(self, query):
        """
        Throws AttributeError --> When conn object is None (make sure connection is established before running this method)
        Throws sqlalchemy.exc.SQLAlchemyError --> if something wrong in query
        :return: None (query just gets executed upon success, else Exception)
        """
        if self.conn is None:
            raise AttributeError('Connection is None -- None type doesnt have any cursor!')
    
        transaction = self.conn.begin()
        try:
            self.conn.execute(query)
            transaction.commit()
        except Exception as e:
            transaction.rollback()
            raise e

    def close_connection(self):
        if self.conn is not None:
            self.conn.close()
