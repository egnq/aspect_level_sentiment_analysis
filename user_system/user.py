# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
from PyQt5.QtWidgets import QMessageBox,QFileDialog
from user_system.restore_model import clean_data, get_w2v,test
from PIL import Image
from wordcloud import ImageColorGenerator
from time import time
from wordcloud import WordCloud

#显示图片
class showp(QWidget):
    def __init__(self,path):
        super().__init__()
        self.sec = 0
        self.setWindowTitle('就是这图')
        self.resize(700, 700)
        self.l1 = QLabel(self)
        # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
        self.png = QtGui.QPixmap(path)
        # 在l1里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
        self.l1.setPixmap(self.png)

class user(object):
    def __init__(self):
        self.divide_sentences = None
        self.add_each_length = None
        self.each_length = None
        self.segMatrix = None
        self.segLen = None
        self.model_name = "BiLSTM_selfattention"
        self.all_label_predict = None
        self.all_label_attention_vec = None
        self.filepath = None
        self.content = None
        self.w2vpath = None
        self.cwd = "E:"
        self.maxseqlength=150
        self.word_cloud_attention=None
        self.label_str = ['location_traffic_convenience', 'location_distance_from_business_district',
                     'location_easy_to_find',
                     'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience',
                     'service_serving_speed',
                     'price_level', 'price_cost_effective', 'price_discount',
                     'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
                     'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
                     'others_overall_experience', 'others_willing_to_consume_again']
        self.label_chinese = ['交通是否便利', '距离商圈远近', '是否容易寻找',
                              '排队等候时间', '服务人员态度', '是否容易停车',
                              '上菜速度如何', '消费价格水平', '食物的性价比',
                              '消费折扣力度', '餐厅装修情况', '餐厅嘈杂情况',
                              '餐厅就餐空间', '餐厅卫生情况', '份量是否合理',
                              '口感是否舒适', '食物外观如何', '推荐程度如何',
                              '本次消费感受', '再次消费意愿']

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.input_edit = QtWidgets.QTextEdit(self.centralwidget)
        self.input_edit.setEnabled(True)
        self.input_edit.setGeometry(QtCore.QRect(30, 10, 741, 131))
        self.input_edit.setPlaceholderText("请输入待分析的餐厅评论文本，注意换行即为下一条评论,否则按一条评论处理")
        self.input_edit.setObjectName("input_edit")

        self.toolButton = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton.setGeometry(QtCore.QRect(740, 120, 31, 21))
        self.toolButton.setObjectName("toolButton")

        self.sentiment_analysis = QtWidgets.QPushButton(self.centralwidget)
        self.sentiment_analysis.setGeometry(QtCore.QRect(40, 180, 141, 51))
        self.sentiment_analysis.setObjectName("sentiment_analysis")

        self.word_cloud = QtWidgets.QPushButton(self.centralwidget)
        self.word_cloud.setGeometry(QtCore.QRect(40, 260, 141, 51))
        self.word_cloud.setObjectName("word_cloud")

        self.output_edit = QtWidgets.QTextEdit(self.centralwidget)
        self.output_edit.setGeometry(QtCore.QRect(230, 160, 541, 411))
        self.output_edit.setObjectName("output_edit")

        self.exit_system = QtWidgets.QPushButton(self.centralwidget)
        self.exit_system.setGeometry(QtCore.QRect(40, 500, 141, 41))
        self.exit_system.setObjectName("exit_system")

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.sentiment_analysis.clicked.connect(self.show_classfier)
        self.word_cloud.clicked.connect(self.show_cloud)
        self.exit_system.clicked.connect(QCoreApplication.quit)

    def logout(self):
        sys.exit(0)

    # 选择文件
    def slot_btn_chooseFile(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(None,
                                                                "选取文件",
                                                                self.cwd,  # 起始路径
                                                                "Text Files (*.txt)")  # 设置文件扩展名过滤,用双分号间隔
        self.filepath = fileName_choose
        if fileName_choose == "":
            return
        self.input_edit.setText("文件读取完成，请开始相关操作")

    # 将输入处理为[sentence1,sentence2,sentence3....]的形式
    def deal_input_edit(self):
        self.content = self.input_edit.toPlainText()
        if self.content == "" or self.content == "请输入待分析的餐厅评论文本，注意换行即为下一条评论,否则按一条评论处理" or self.content == "文件读取完成，请开始相关操作":
            self.content = None
            return
        self.content = self.content.split("\n")

        # 进行分词

    def divide_clean_content(self):
        self.deal_input_edit()
        if self.filepath == None and self.content == None:
            QMessageBox.information(None,
                                    "输入错误",
                                    "请确保输入或者选择了文件",
                                    QMessageBox.Yes)
            return
        self.divide_sentences, self.add_each_length, self.each_length = clean_data(file_path=self.filepath,
                                                                                   sentences=self.content)
        strs = ''
        first = 0
        end = 0
        for i in self.add_each_length:
            if i == 0:
                continue
            end = i
            for word in self.divide_sentences[first:end]:
                strs = strs + word + " "
            strs = strs + '\n'
            first = end
        self.output_edit.setText(strs)
        return

    # 生成predict和attention
    def the_classifier(self):
        self.divide_clean_content()
        if self.divide_sentences is None:
            return
        self.segMatrix, self.each_length = get_w2v(divide_sentences=self.divide_sentences,
                                                   each_length=self.add_each_length)
        for the_label in self.label_str:
            predict, attention_vec = test(the_labels=the_label, model_names=self.model_name, egMatrix0=self.segMatrix,
                                          each_segLen0=self.each_length)
            if self.all_label_predict is None:
                self.all_label_predict = np.array(predict)
            else:
                self.all_label_predict = np.vstack([np.array(self.all_label_predict), np.array(predict)])
            if attention_vec is None:
                continue
            else:
                if self.all_label_attention_vec is None:
                    self.all_label_attention_vec = np.array(attention_vec).reshape(1, -1, self.maxseqlength)
                else:
                    self.all_label_attention_vec = np.concatenate([np.array(self.all_label_attention_vec),
                                                                   np.array(attention_vec).reshape(1, -1,
                                                                                                   self.maxseqlength)],
                                                                  axis=0)
        return

    def show_classfier(self):
        self.the_classifier()
        if self.all_label_predict is None:
            return
        sentiment = ['未提及', '负面', '中性', '正面']
        # 最后的输出
        output_str = ''
        # 对分类样本做概述：有多少个句子
        output_str += "\n一共获取得到%s个评论文本,分析结果如下：\n" % (np.size(np.array(self.each_length)))
        # 各个句子在20个标签上面的分类情况统计
        predict_each_label_each_classifer = np.zeros(shape=[20, 4])
        for i in range(20):
            for value in self.all_label_predict[i, :]:
                # 计算每个标签下，4个分类各自有多少个
                predict_each_label_each_classifer[i, value + 2] += 1
        # 输出分类结果
        output_str += "            未提及    消极    中性    积极    \n"
        predict_each_label_each_classifer.astype(np.int)
        for i in range(20):
            label = self.label_chinese[i]
            strs = '' + label + ':  '
            for j in range(4):
                strs += str(predict_each_label_each_classifer[i, j]) + '      '
            strs += ' \n'
            output_str += strs
        self.output_edit.append(output_str)

        if self.all_label_attention_vec is None:
            return

        all_attenton = np.sum(self.all_label_attention_vec, axis=0)
        numsectence = np.size(np.array(self.each_length))
        attention = None
        for i in range(numsectence):
            length = self.each_length[i]
            userful_attention_in_a_sentence = all_attenton[i, :length]
            if attention is None:
                attention = userful_attention_in_a_sentence
            else:
                attention = np.hstack([np.array(attention), userful_attention_in_a_sentence])
        self.word_cloud_attention = attention

    def show_cloud(self):
        if self.divide_sentences is None:
            self.divide_clean_content()
            if self.divide_sentences is None:
                return
        if self.word_cloud_attention is None:
            text = " ".join(self.divide_sentences)
            background_image = np.array(Image.open('./background.jpg'))
            wordcloud = WordCloud(
                font_path="./STSONG.TTF",
                background_color="white",
                mask=background_image,
                max_words=200,  # 设置最大显示的词数
                max_font_size=100,  # 设置字体最大值
                random_state=50,  # 设置随机生成状态，即多少种配色方案
                width=700,
                height=700,
                min_font_size=20,
            ).generate(text)
            # 下面代码表示显示图片
            image_colors = ImageColorGenerator(background_image)
            method = int(time())
            path = "./dataset/user_content/%s.png" % (method)
            wordcloud.to_file(path)
            self.ui = showp(path=path)
            self.ui.show()
        else:
            background_image = np.array(Image.open('./background.jpg'))
            ca = (self.word_cloud_attention) * 10000
            ca.astype(np.int)
            ds = self.divide_sentences
            dic = dict(zip(ds, ca))
            wordcloud = WordCloud(
                font_path="./STSONG.TTF",
                background_color="white",
                mask=background_image,
                max_words=200,  # 设置最大显示的词数
                max_font_size=100,  # 设置字体最大值
                random_state=50,  # 设置随机生成状态，即多少种配色方案
                width=700,
                height=700,
                min_font_size=20,
            ).generate_from_frequencies(dic)
            # 下面代码表示显示图片
            image_colors = ImageColorGenerator(background_image)
            method = int(time())
            path = "./dataset/user_content/%s.png" % (method)
            wordcloud.to_file(path)
            self.ui = showp(path=path)
            self.ui.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "用户你好"))
        self.toolButton.setText(_translate("MainWindow", "..."))
        self.sentiment_analysis.setText(_translate("MainWindow", "计算情感分类结果"))
        self.word_cloud.setText(_translate("MainWindow", "生成词云图"))
        self.exit_system.setText(_translate("MainWindow", "退出系统"))

