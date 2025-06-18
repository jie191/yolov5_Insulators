# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# models/__init__.py
# 导入必要的模块
from .common import *
from .experimental import *
from .yolo import Model

# 如果需要加载标签，可以在这里添加相关代码
labels = []  # 初始化标签列表


# 或者定义一个类来管理模型加载
class ModelLoader:
    def __init__(self, labels_path=None):
        self.labels = []
        if labels_path:
            self.load_labels(labels_path)

    def load_labels(self, path):
        # 实现加载标签的逻辑
        with open(path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        print(f"加载标签文件: {len(self.labels)} 个")