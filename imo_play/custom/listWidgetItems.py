from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from flags import *


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


'载入基础文件'
class LoadItem(MyItem):
    def __init__(self, parent=None):
        super(LoadItem, self).__init__(' 载入 ', parent=parent)

    def __call__(self):
       pass


'添加搜索框'
class SearchItem(MyItem):
    def __init__(self, parent=None):
        super(SearchItem, self).__init__(' 搜索 ', parent=parent)

    def __call__(self):
       pass

'搜索 并由name获得link 在link基础上 进入各链接对应网页 获取视频链接'
class GetItem(MyItem):

    def __init__(self, parent=None):
        super().__init__('提取', parent=parent)

    def __call__(self):
        pass

'在右侧显示集数'
class DisplayItem(MyItem):
    def __init__(self, parent=None):
        super(DisplayItem, self).__init__(' 显示 ', parent=parent)

    def __call__(self):
       pass