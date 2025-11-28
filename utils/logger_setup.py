import logging
import sys
import os
from contextlib import contextmanager

def setup_logger(name, log_file, level=logging.INFO):
    """
    配置一个 logger，只将日志输出到指定的文件中。
    """
    logger = logging.getLogger(name)
    # 如果 logger 已经配置过，直接返回，防止重复添加 handler
    if logger.hasHandlers():
        logger.handlers.clear() # 清除旧的 handlers，确保配置唯一
        
    logger.setLevel(level)
    
    # 文件处理器 - 记录所有信息
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w') # 使用追加模式
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
        
    return logger

@contextmanager
def suppress_stdout():
    """
    一个上下文管理器，用于临时抑制所有到标准输出的 print()。
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout