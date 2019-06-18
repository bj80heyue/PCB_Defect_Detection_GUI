import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from mainWindow import MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, qApp
from PyQt5.QtCore import Qt, QEvent, QObject

if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = MainWindow()
	sys.exit(app.exec_())
