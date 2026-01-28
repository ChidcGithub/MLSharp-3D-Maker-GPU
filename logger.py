import sys
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.log_file = None
        self.setup_logger()
    
    def setup_logger(self):
        """设置日志系统"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'auto_diagnose_{timestamp}.log')
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"自动诊断日志 - {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")
    
    def log(self, message, level='INFO'):
        """记录日志"""
        if self.log_file and os.path.exists(self.log_file):
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] [{level}] {message}\n")
    
    def log_output(self, output, source='stdout'):
        """记录脚本输出"""
        if self.log_file and os.path.exists(self.log_file):
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n--- {source} 输出 ---\n")
                f.write(output)
                f.write(f"\n--- {source} 结束 ---\n\n")
    
    def log_error(self, error, context=''):
        """记录错误"""
        if self.log_file and os.path.exists(self.log_file):
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] [ERROR] {context}\n")
                f.write(f"  错误信息: {error}\n\n")

logger = Logger()
