from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class MyItem(QListWidgetItem):
    def __init__(self, name=None, parent=None):
        super(MyItem, self).__init__(name, parent=parent)
        self.mainwindow = parent
        self.setIcon(QIcon('icons/color.png'))
        self.setSizeHint(QSize(60, 60))  # size

    def get_params(self):
        protected = [v for v in dir(self) if v.startswith('_') and not v.startswith('__')]
        param = {}
        for v in protected:
            param[v.replace('_', '', 1)] = self.__getattribute__(v)
        return param

    def update_params(self, param):
        for k, v in param.items():
            if '_' + k in dir(self):
                self.__setattr__('_' + k, v)

class InitItem(MyItem):
    def __init__(self, parent=None):
        super(InitItem, self).__init__(' 初始化 ', parent=parent)

    def __call__(self):
       pass

'检测视频'
class VideoItem(MyItem):
    def __init__(self, parent=None):
        super(VideoItem, self).__init__(' 检测 ', parent=parent)

    def __call__(self):
       pass


