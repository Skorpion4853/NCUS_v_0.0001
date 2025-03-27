

################################################################################
## Form generated from reading UI file 'test.ui'
##
## Created by: Qt User Interface Compiler version 6.8.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QMainWindow, QPlainTextEdit, QPushButton,
    QSizePolicy, QTextEdit, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(421, 561)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QSize(421, 561))
        MainWindow.setCursor(QCursor(Qt.CursorShape.IBeamCursor))
        MainWindow.setAcceptDrops(False)
        MainWindow.setStyleSheet(u"QWidget{\n"
"	color: white;\n"
"	background-color: #121212;\n"
"	font-family: Rubik;\n"
"	font-size: 8pt;\n"
"	font-weight: 200;\n"
"}")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.chat = QPlainTextEdit(self.centralwidget)
        self.chat.setObjectName(u"chat")
        self.chat.setGeometry(QRect(10, 9, 401, 461))
        self.chat.setMinimumSize(QSize(260, 390))
        self.chat.setStyleSheet(u"")
        self.chat.setReadOnly(True)
        self.msg = QTextEdit(self.centralwidget)
        self.msg.setObjectName(u"msg")
        self.msg.setGeometry(QRect(10, 480, 311, 71))
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.msg.sizePolicy().hasHeightForWidth())
        self.msg.setSizePolicy(sizePolicy1)
        self.msg.setMinimumSize(QSize(210, 50))
        self.msg.setSizeIncrement(QSize(0, 0))
        self.msg.viewport().setProperty(u"cursor", QCursor(Qt.CursorShape.IBeamCursor))
        self.msg.setMouseTracking(True)
        self.msg.setTabletTracking(True)
        self.msg.setAutoFillBackground(False)
        self.msg.setTabChangesFocus(False)
        self.msg.setUndoRedoEnabled(False)
        self.msg.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.msg.setLineWrapColumnOrWidth(0)
        self.msg.setReadOnly(False)
        self.btn_snd = QPushButton(self.centralwidget)
        self.btn_snd.setObjectName(u"btn_snd")
        self.btn_snd.setGeometry(QRect(330, 480, 81, 71))
        self.btn_snd.setMinimumSize(QSize(50, 50))
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Test", None))
#if QT_CONFIG(statustip)
        self.msg.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.msg.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Rubik'; font-size:8pt; font-weight:200; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">Type anything...</span></p></body></html>", None))
        self.btn_snd.setText(QCoreApplication.translate("MainWindow", u"Send", None))
    # retranslateUi

