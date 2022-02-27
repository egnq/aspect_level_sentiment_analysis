from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from content import read_txt
from PyQt5.QtWidgets import QApplication, QMessageBox
class logindialog(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.i = 0
        self.setWindowTitle('餐厅情感分析系统')
        self.resize(200, 200)
        self.setFixedSize(self.width(), self.height())
        self.setWindowFlags(Qt.WindowCloseButtonHint)

        ###### 设置界面控件
        self.frame = QFrame(self)
        self.verticalLayout = QVBoxLayout(self.frame)

        self.lineEdit_account = QLineEdit()
        self.lineEdit_account.setPlaceholderText("请输入账号")
        self.verticalLayout.addWidget(self.lineEdit_account)

        self.lineEdit_password = QLineEdit()
        self.lineEdit_password.setPlaceholderText("请输入密码")
        self.lineEdit_password.setEchoMode(QLineEdit.Password)
        self.verticalLayout.addWidget(self.lineEdit_password)

        self.lbl = QLabel("")  # 创建一个 标签
        self.cb = QComboBox()  # 创建一个 Qcombo  box实例
        self.cb.addItem("用户模式")  # 添加 item
        self.cb.addItem("研究者模式")
        self.cb.currentIndexChanged.connect(self.selectionchange)
        self.verticalLayout.addWidget(self.cb)

        self.pushButton_login = QPushButton()
        self.pushButton_login.setText("登录")
        self.verticalLayout.addWidget(self.pushButton_login)

        self.pushButton_register = QPushButton()
        self.pushButton_register.setText("注册")
        self.verticalLayout.addWidget(self.pushButton_register)

        ###### 绑定按钮事件
        self.pushButton_login.clicked.connect(self.on_pushButton_login_clicked)
        self.pushButton_register.clicked.connect(self.on_pushButton_register_clicked)

    def returnmoshi(self):
        return self.i

    def selectionchange(self, i):
        self.lbl.setText(self.cb.currentText())  # 将当前选项 文字设置子lab 标签上
        self.lbl.adjustSize()
        if str(self.cb.currentText()) == "用户模式":
            self.i = 0
        else:
            self.i = 1

    def on_pushButton_register_clicked(self, event):  # 消息：信息
        strs = str(self.lineEdit_account.text()) + ' ' + str(self.lineEdit_password.text()) +' '+ str(self.i) + '\n'
        logs = read_txt(r"./user_system/register_login.txt")
        for sen in logs:
            l = sen.split(" ")
            name = l[0]
            # 账号判断
            if self.lineEdit_account.text() == name:
                i = 1
                QMessageBox.information(self,
                                        "账号已被注册",
                                        "请重新输入注册账号",
                                        QMessageBox.Yes)
                return
        if str(self.lineEdit_account.text()) == '' or str(self.lineEdit_account.text()) == '请输入账号' or str(
                self.lineEdit_password.text()) == '' or str(self.lineEdit_password.text()) == '请输入密码':
            QMessageBox.information(self,
                                    "检测错误",
                                    "请完整输入账号密码",
                                    QMessageBox.Yes)
            return
        reply = QMessageBox.information(self,
                                        "确认提醒",
                                        "是否确认注册该账号，一旦确认无法更改",
                                        QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No:
            return
        file = open(r"./user_system/register_login.txt", 'a', encoding='utf-8')
        file.write(strs)
        file.close()
        self.accept()
        return

    def on_pushButton_login_clicked(self):
        logs = read_txt(r"./user_system/register_login.txt")
        i = 0
        j = 0
        k = 0
        for sen in logs:
            l = sen.split(" ")
            name = l[0]
            password = l[1]
            moshi = int(l[2])
            # 账号判断
            if self.lineEdit_account.text() == name:
                i = 1
                # 密码判断
                if self.lineEdit_password.text() == password:
                    j = 1
                    if self.i == moshi:
                        k = 1
            if i == 1 and j == 1 and k == 1:
                # 通过验证，关闭对话框并返回1
                self.accept()
                return
            if i == 1:
                reply = QMessageBox.information(self,
                                                "账号存在",
                                                "但请检查密码或模式是否正确",
                                                QMessageBox.Yes | QMessageBox.No)
                return
        reply = QMessageBox.information(self,
                                        "账号不存在",
                                        "请进行注册账号",
                                        QMessageBox.Yes | QMessageBox.No)
        return