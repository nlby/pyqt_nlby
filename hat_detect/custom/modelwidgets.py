from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from config import items

class MyListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.mainwindow = parent
        self.setDragEnabled(True)
        # 选中不显示虚线
        # self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setFocusPolicy(Qt.NoFocus)



class ReListWidget(MyListWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # self.setFixedHeight(64)
        self.setFlow(QListView.TopToBottom)  # 设置列表方向 为水平从上到下
        # self.setViewMode(QListView.IconMode)  # 设置列表模式
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # 关掉滑动条
        self.setAcceptDrops(False)
        # self.itemClicked.connect(self.dispatcher)  # 点击该功能 选定操作区域就会显示该操作
        self.setMinimumWidth(200)

    def dispatcher(self):
        self.num = self.mainwindow.funcListWidget.num
        self.videolist = self.mainwindow.funcListWidget.videolist
        print(str(self.num))
        print(self.videolist)
        for i in range(self.num):
            if self.item(i) is self.currentItem():
                # print("二")
                self.mainwindow.videoPlayer.player.setMedia(QMediaContent(QUrl(self.videolist[i])))

    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        self.setCurrentRow(-1)  # 取消选中状态


class FuncListWidget(MyListWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedHeight(64)  #
        self.setFlow(QListView.LeftToRight)  # 设置列表方向 为水平从左到右
        self.setViewMode(QListView.IconMode)  # 设置列表模式
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关掉滑动条
        self.setAcceptDrops(False)
        for itemType in items:
            self.addItem(itemType())
        self.itemClicked.connect(self.dispatcher)
        self.yolo = None



    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        self.setCurrentRow(-1)  # 取消选中状态

    def dispatcher(self):
        if type(self.currentItem()) is items[0]:
            self.init_param()
        if type(self.currentItem()) is items[1]:
            self.detect_video()

    def init_param(self):
        self.mainwindow.resultlist.clear()
        self.text = QDockWidget(self)
        self.te = QLineEdit()
        self.te.setText("输出路径")
        self.text.setWidget(self.te)
        self.mainwindow.addDockWidget(Qt.LeftDockWidgetArea, self.text)

        # self.mainwindow.led.click()

    def detect_video(self):
        print(self.mainwindow.src_video)
        print(self.te.text())
        if (self.te.text() == "测试"):
            self.mainwindow.resultlist.addItem("右侧为模型输出");
            self.mainwindow.led.click()
            import winsound

            duration = 1000  # millisecond
            freq = 440  # Hz
            winsound.Beep(freq, duration)
            winsound.PlaySound('Tik Tok.wav', winsound.SND_FILENAME)




