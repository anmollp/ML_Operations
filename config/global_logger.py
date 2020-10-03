import logging
from logging import handlers


class GlobalLogger:

    @staticmethod
    def get_logger(log_file_path='default_file.log', logger_name='Default Logger'):
        log_format = '[%(levelname)s] %(asctime)s %(filename)s %(module)s %(funcName)s %(lineno)d : %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        logger_cron = logging.getLogger(logger_name)
        log_handler = logging.handlers.RotatingFileHandler(log_file_path, mode='a')
        logger_cron.setLevel(logging.INFO)
        log_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
        logger_cron.addHandler(log_handler)
        return logger_cron
