import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from user_system.researcher import researcher
from user_system.user import user
from user_system.login_register import logindialog

#######程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = logindialog()
    if dialog.exec_() == QDialog.Accepted:
        moshi=dialog.returnmoshi()
        widget = QtWidgets.QMainWindow()
        if moshi==1:
            ui = researcher()
            ui.setupUi(widget)
            widget.show()
        else:
            ui = user()
            ui.setupUi(widget)
            widget.show()
        sys.exit(app.exec_())
