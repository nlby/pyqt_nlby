import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from custom.stackedWidget import StackedWidget
from custom.treeView import FileSystemTreeView
from custom.videoPlayer import videoPlayer
from custom.animelistwidgets import AnimeListWidget, FuncListWidget

class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()

        '设置工具栏样式 并添加点击事件'
        # self.tool_bar = self.addToolBar('工具栏')
        # self.action_right_rotate = QAction(QIcon("icons/右旋转.png"), "向右旋转90", self)
        # self.action_left_rotate = QAction(QIcon("icons/左旋转.png"), "向左旋转90°", self)
        # self.action_histogram = QAction(QIcon("icons/直方图.png"), "直方图", self)
        # self.action_right_rotate.triggered.connect(self.right_rotate)
        # self.action_left_rotate.triggered.connect(self.left_rotate)
        # self.action_histogram.triggered.connect(self.histogram)
        # self.tool_bar.addActions((self.action_left_rotate, self.action_right_rotate, self.action_histogram))

        '以下是对主布局各部分分别实例化并设置属性'
        # self.useListWidget = UsedListWidget(self)    # 表示选定操作
        # self.graphicsView = GraphicsView(self)
        self.funcListWidget = FuncListWidget(self)   # 表示功能菜单
        self.stackedWidget = StackedWidget(self)     # 表示操作相应属性
        self.fileSystemTreeView = FileSystemTreeView(self)  # 表示文件目录
        self.videoPlayer = videoPlayer(self)   # 视频播放器
        self.animelist = AnimeListWidget(self)

        self.dock_file = QDockWidget(self)
        self.dock_file.setWidget(self.fileSystemTreeView)
        self.dock_file.setTitleBarWidget(QLabel('目录'))
        self.dock_file.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_func = QDockWidget(self)
        self.dock_func.setWidget(self.funcListWidget)
        self.dock_func.setTitleBarWidget(QLabel('功能选项'))
        self.dock_func.setFeatures(QDockWidget.NoDockWidgetFeatures)

        # self.dock_used = QDockWidget(self)
        # self.dock_used.setWidget(self.useListWidget)
        # self.dock_used.setTitleBarWidget(QLabel('已选操作'))
        # self.dock_used.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.dock_attr = QDockWidget(self)
        self.dock_attr.setWidget(self.stackedWidget)
        self.dock_attr.setTitleBarWidget(QLabel('属性'))
        self.dock_attr.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock_attr.close()

        self.dock_anime = QDockWidget(self)
        self.dock_anime.setWidget(self.animelist)
        self.dock_anime.setTitleBarWidget(QLabel('播放列表'))
        self.dock_anime.setFeatures(QDockWidget.NoDockWidgetFeatures)

        '主页面布局'
        # self.setCentralWidget(self.graphicsView)
        # self.addDockWidget(Qt.RightDockWidgetArea, self.dock_used)
        self.videoPlayer.player.setMedia(QMediaContent(QUrl("https://gss3.baidu.com/6LZ0ej3k1Qd3ote6lo7D0j9wehsv/tieba-smallvideo/60_24e01d9734b6b1cb51a9148ead8940c0.mp4")))
        self.setCentralWidget(self.videoPlayer)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_file)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_func)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_anime)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_attr)

        '设置标题栏样式'
        self.setWindowTitle('视频播放器')
        self.setWindowIcon(QIcon('icons/main.png'))

        '全局会用到的资源文件'
        self.src_img = None
        self.cur_img = None
        self.src_video = None

    '在已有源文件的情况下 用选定操作处理当前文件获得新文件'
    def update_image(self):
        if self.src_img is None:
            return
        img = self.process_image()
        self.cur_img = img
        self.graphicsView.update_image(img)

    # '可以直接修改源文件 同时要经过选定操作的处理'
    # def change_image(self, img):
    #     self.src_img = img
    #     img = self.process_image()
    #     self.cur_img = img
    #     self.graphicsView.change_image(img)
    #
    # '根据选定操作处理图片'
    # def process_image(self):
    #     img = self.src_img.copy()
    #     for i in range(self.useListWidget.count()):
    #         img = self.useListWidget.item(i)(img)
    #     return img
    #
    # '工具栏 顺时针旋转'
    # def right_rotate(self):
    #     self.graphicsView.rotate(90)
    #
    # '工具栏 逆时针旋转'
    # def left_rotate(self):
    #     self.graphicsView.rotate(-90)
    #
    # def histogram(self):
    #     color = ('b', 'g', 'r')
    #     for i, col in enumerate(color):
    #         histr = cv2.calcHist([self.cur_img], [i], None, [256], [0, 256])
    #         histr = histr.flatten()
    #         plt.plot(range(256), histr, color=col)
    #         plt.xlim([0, 256])
    #     plt.show()

    '更改视频播放器的源'
    def changevideo(self, video):
        self.src_video = video
        self.videoPlayer.getfile(self.src_video)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(open('./custom/styleSheet.qss', encoding='utf-8').read())
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
