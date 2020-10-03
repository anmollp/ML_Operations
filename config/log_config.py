from config.global_logger import GlobalLogger

log_base_dir = ''
global_logger = GlobalLogger()

pipeline_log_path = log_base_dir + 'ML_Pipeline.log'
pipeline_logger_name = 'ML_Pipeline_logger'
pipeline_logger = global_logger.get_logger(pipeline_log_path, pipeline_logger_name)
