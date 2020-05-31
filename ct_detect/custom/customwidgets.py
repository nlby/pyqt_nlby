from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

allAttributes =   [  'colorOnBegin', 'colorOnEnd', 'colorOffBegin', 'colorOffEnd', 'colorBorderIn', 'colorBorderOut',
                      'radiusBorderOut', 'radiusBorderIn', 'radiusCircle']
allDefaultVal =   [ QColor(0, 240, 0), QColor(0, 160, 0), QColor(0, 68, 0), QColor(0, 28, 0), QColor(140, 140, 140), QColor(100, 100, 100),
                     500, 450, 400]
allLabelNames =   [ u'灯亮圆心颜色：', u'灯亮边缘颜色：', u'灯灭圆心颜色：', u'灯灭边缘颜色：', u'边框内测颜色：', u'边框外侧颜色：',
                     u'边框外侧半径：', u'边框内侧半径：', u'中间圆灯半径：']



class MyLed(QAbstractButton):
    def __init__(self, parent=None):
        super(MyLed, self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.setMinimumSize(240, 240)
        self.setCheckable(True)
        self.scaledSize = 1000.0    #为方便计算，将窗口短边值映射为1000
        self.setLedDefaultOption()

    def setLedDefaultOption(self):
        for attr, val in zip(allAttributes, allDefaultVal):
            setattr(self, attr, val)
        self.update()

    def setLedOption(self, opt='colorOnBegin', val=QColor(0,240,0)):
        if hasattr(self, opt):
            setattr(self, opt, val)
            self.update()

    def resizeEvent(self, evt):
        self.update()

    def paintEvent(self, evt):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(Qt.black, 1))

        realSize = min(self.width(), self.height())                         #窗口的短边
        painter.translate(self.width()/2.0, self.height()/2.0)              #原点平移到窗口中心
        painter.scale(realSize/self.scaledSize, realSize/self.scaledSize)   #缩放，窗口的短边值映射为self.scaledSize
        gradient = QRadialGradient(QPointF(0, 0), self.scaledSize/2.0, QPointF(0, 0))   #辐射渐变

        #画边框外圈和内圈
        for color, radius in [(self.colorBorderOut, self.radiusBorderOut),  #边框外圈
                               (self.colorBorderIn, self.radiusBorderIn)]:   #边框内圈
            gradient.setColorAt(1, color)
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(QPointF(0, 0), radius, radius)

        # 画内圆
        gradient.setColorAt(0, self.colorOnBegin if self.isChecked() else self.colorOffBegin)
        gradient.setColorAt(1, self.colorOnEnd if self.isChecked() else self.colorOffEnd)
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(QPointF(0, 0), self.radiusCircle, self.radiusCircle)

class MyColorBox(QFrame):
    sigColorChanged = pyqtSignal(QColor)
    def __init__(self, parent=None, height=20, color=QColor(0,240,0)):
        super(MyColorBox, self).__init__(parent)
        self.setFixedHeight(height)
        self.setAutoFillBackground(True)
        self.setPalette(QPalette(color))
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)

    def mousePressEvent(self, *args, **kwargs):
        color = QColorDialog.getColor(initial=self.palette().color(QPalette.Window))
        if color.isValid():
            self.setPalette(QPalette(color))
            self.sigColorChanged.emit(color)

    def setColor(self, color):
        self.setPalette(QPalette(color))