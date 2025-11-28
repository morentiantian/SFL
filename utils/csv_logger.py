import csv
import os
from datetime import datetime

class CSVLogger:
    """
    一个简单的CSV日志记录器，用于将每轮的实验数据结构化地保存下来。
    """
    def __init__(self, log_dir: str, file_prefix: str, fieldnames: list):
        """
        初始化Logger。

        :param log_dir: 日志文件存放的目录。
        :param file_prefix: 日志文件的前缀，例如 'eval_mappo' 或 'eval_fixed_policy'。
        :param fieldnames: CSV文件的表头字段列表。
        """
        # 创建一个带时间戳的、独一无二的文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filepath = os.path.join(log_dir, f"{file_prefix}_{timestamp}.csv")
        self.fieldnames = fieldnames
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 打开文件并写入表头
        self.csv_file = open(self.filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        
        print(f"CSVLogger initialized. Logging to: {self.filepath}")

    def log_row(self, data_dict: dict):
        """
        写入一行数据。

        :param data_dict: 一个包含本行数据的字典，键应与fieldnames匹配。
        """
        # 过滤掉不在表头中的多余数据，以防出错
        filtered_data = {k: data_dict.get(k, '') for k in self.fieldnames}
        self.writer.writerow(filtered_data)
        # 立即将缓冲区数据写入磁盘，以防程序意外中断时数据丢失
        self.csv_file.flush()

    def close(self):
        """
        关闭文件。
        """
        if self.csv_file:
            self.csv_file.close()