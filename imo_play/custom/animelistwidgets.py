from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import requests
from lxml import etree
import re
from selenium import webdriver
from config import items

class MyListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.mainwindow = parent
        self.setDragEnabled(True)
        # 选中不显示虚线
        # self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setFocusPolicy(Qt.NoFocus)


names = []
links = []
contains = []
class AnimeListWidget(MyListWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # self.setFixedHeight(64)
        self.setFlow(QListView.TopToBottom)  # 设置列表方向 为水平从上到下
        # self.setViewMode(QListView.IconMode)  # 设置列表模式
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关掉滑动条
        self.setAcceptDrops(False)
        self.itemClicked.connect(self.dispatcher)  # 点击该功能 选定操作区域就会显示该操作
        self.setMinimumWidth(10)

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



    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        self.setCurrentRow(-1)  # 取消选中状态

    def dispatcher(self):
        if type(self.currentItem()) is items[0]:
            self.load_data()
        if type(self.currentItem()) is items[1]:
            self.add_widget()
        if type(self.currentItem()) is items[2]:
            self.get_info()
        if type(self.currentItem()) is items[3]:
            self.display_anime()

    def load_data(self):
        s = ""
        with open('name.txt', 'r', encoding='utf-8') as f:   # 注意此处必须有 encoding='utf-8' 否则报错 且注意name的路径是相对main.py的
            s = f.read()
        self.names = s.split('\n')
        with open('link.txt', 'r', encoding='utf-8') as f:
            s = f.read()
        self.links = s.split('\n')

    def add_widget(self):
        self.mainwindow.animelist.clear()
        self.text = QDockWidget(self)
        self.te = QLineEdit()
        self.text.setWidget(self.te)
        self.mainwindow.addDockWidget(Qt.LeftDockWidgetArea, self.text)


    def get_info(self):
        # print(len(self.names))
        if self.te.text() in self.names:
            index = self.names.index(self.te.text())
        base_url = self.links[index]
        if len(base_url) > 0:
            self.parse_detail(base_url)

    def display_anime(self):
        for t in self.titlelist:
            self.mainwindow.animelist.addItem(t)

    def parse_detail(self, base_url):
        '公共部分'
        header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.87 Safari/537.36'
        }
        options = webdriver.ChromeOptions()  # 针对动态显示的视频链接
        options.add_argument('--headless')
        chrome = webdriver.Chrome(executable_path="chromedriver.exe", chrome_options=options)
        url = "http://www.imomoe.in"

        '爬取视频页链接和标题'
        response = requests.get(base_url, headers=header)
        html = response.content.decode("gbk")
        exml = etree.HTML(html)
        pagelist = exml.xpath("//div[@id='play_0']/ul//a/@href")
        for i in range(0, len(pagelist)):
            pagelist[i] = url + pagelist[i]
        self.titlelist = exml.xpath("//div[@id='play_0']/ul//a/text()")
        print(len(self.titlelist))
        print(len(pagelist))
        self.num = len(self.titlelist)

        self.videolist = []
        print('请稍等 正在获取视频链接...')
        '爬取动态显示的视频链接 速度会特别慢（几min 考虑之后优化）'
        for p in pagelist:
            chrome.get(p)
            html = chrome.page_source
            exml = etree.HTML(html)
            video = exml.xpath("//div[@class='player']/iframe[@id='play2']/@src")[0]
            video = re.findall("vid=(.+)&userlink", video)[0]
            self.videolist.append(video)
            print(video)
        print(len(self.videolist))
        print('已获取所有视频链接')
        chrome.quit()
