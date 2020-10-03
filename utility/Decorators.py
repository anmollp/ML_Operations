from functools import wraps
from config.global_logger import GlobalLogger


class Decorator:
    def __init__(self, logger=GlobalLogger.get_logger()):
        self.logger = logger

    def log_it(self, func):
        """
        Function logger
        :param func: function whose execution is to be logged
        :return: logger wrapper function
        """
        @wraps(func)
        def log_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if type(result) == Exception:
                self.logger.error("Exception occurred in {}: {}".format(func.__qualname__, result))
            else:
                self.logger.info("Executed {}".format(func.__qualname__))
            return result
    
        return log_wrapper

    def db_connect(self, func):
        """
        Graceful database connector/closer.
        :param func: Function involving connection to database.
        :return: Connectivity wrapper.
        """
        @wraps(func)
        def connection_wrapper(*args, **kwargs):
            database_connection = kwargs.get("connection")
            if database_connection:
                database_connection.create_connection()
                self.logger.info("Connection established or reusing existing connection.")
            else:
                self.logger.info("Connection Error")
                raise Exception("Database Connection Error")
            result = func(*args, **kwargs)
            database_connection.close_connection()
            self.logger.info("Connection Closed")
            return result
        
        return connection_wrapper
