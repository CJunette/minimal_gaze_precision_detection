import itertools
import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QDesktopWidget
from PyQt5.QtCore import Qt, QTimer, QRectF, QCoreApplication
from PyQt5.QtGui import QBrush, QColor
import time
import configs


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.col_num = configs.col_num  # Columns
        self.row_num = configs.row_num  # Rows
        self.current_col = configs.col_num - 1
        self.current_row = configs.row_num - 1
        self.exp_index = -1
        self.random_indices = self.create_random_indices()
        self.point = None

        self.cap = cv2.VideoCapture(configs.camera_index)
        ret, frame = self.cap.read()

        self.setStyleSheet("background-color: #777777;")

        self.initUI()

    def initUI(self):
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(0, 0, screen.width(), screen.height())

        # if configs.bool_full_screen:
        #     self.setGeometry(0, 0, screen.width(), screen.height())
        # else:
        #     self.setGeometry(0, 0, configs.screen_width, configs.screen_height)

        self.view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)

        self.view.setScene(self.scene)
        self.view.setGeometry(0, 0, self.width(), self.height())
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        # self.show_all_points()

        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.show_next_point)
        # self.show_next_point()

        if configs.bool_full_screen:
            self.showFullScreen()

    def create_random_indices(self):
        combinations = np.array(list(itertools.product(range(configs.col_num), range(configs.row_num))))
        np.random.shuffle(combinations)
        combinations = combinations.tolist()

        # write to file
        file_path = f"output/subject_{configs.subject_num}/{configs.mode}/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open(f"{file_path}/random_indices.txt", "w") as f:
            for i in range(len(combinations)):
                f.write(f"{combinations[i][0]} {combinations[i][1]}\n")

        return combinations

    def show_all_points(self):
        for i in range(self.row_num):
            for j in range(self.col_num):
                point = self.generate_point(j, i)
                self.scene.addItem(point)
        self.scene.setSceneRect(0, 0, self.width(), self.height())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_S:
            if self.exp_index >= 0:
                self.change_exp_index("increase")
                self.capture_video()
            else:
                self.exp_index = 0

            if self.point is not None:
                self.scene.removeItem(self.point)

            self.jump_to_next()
            self.show_current_point()
            print(self.exp_index)
            # print("next")
        elif event.key() == Qt.Key_W:
            if self.exp_index >= 0:
                self.change_exp_index("decrease")
                self.capture_video()
            else:
                self.exp_index = 0

            if self.point is not None:
                self.scene.removeItem(self.point)

            self.jump_to_last()
            self.show_current_point()
            print(self.exp_index)
            # print("last")

    def change_exp_index(self, change):
        if change == "increase":
            self.exp_index += 1
            if self.exp_index >= configs.row_num * configs.col_num:
                self.close() # 到实验最后一个点，则关闭。
        elif change == "decrease":
            self.exp_index -= 1
            if self.exp_index < 0:
                self.exp_index = configs.row_num * configs.col_num - 1

    def show_current_point(self):
        # print(self.current_row, self.current_col)

        self.point = self.generate_point(self.current_col, self.current_row)
        self.scene.addItem(self.point)

    def generate_point(self, index_col, index_row):
        x = (index_col + 0) * (self.width() - configs.screen_padding_horizontal * 2) / (self.col_num - 1) + configs.screen_padding_horizontal
        y = (index_row + 0) * (self.height() - configs.screen_padding_vertical * 2) / (self.row_num - 1) + configs.screen_padding_vertical
        # print(self.width(), self.height(), x, y)

        point = QGraphicsEllipseItem(QRectF(x - 5, y - 5, 10, 10))
        point.setBrush(QBrush(QColor(0, 0, 0)))
        self.scene.addItem(point)
        self.scene.setSceneRect(0, 0, self.width(), self.height())
        return point

    def jump_to_next(self):
        # TODO 在不同模式下，跳转到下一个点的方式不同。
        print(self.current_row, self.current_col)
        if configs.mode == "horizontal_first":
            self.current_row, self.current_col = self.jump_to_next_specific(self.current_row, self.current_col, self.row_num, self.col_num)
        elif configs.mode == "vertical_first":
            self.current_col, self.current_row = self.jump_to_next_specific(self.current_col, self.current_row, self.col_num, self.row_num)
        elif configs.mode == "random":
            self.current_col, self.current_row = self.jump_to_random()

        if (self.current_row == self.row_num and self.current_col == 0) or (self.current_col == self.col_num and self.current_row == 0):
            self.current_row = 0
            self.current_col = 0

    def jump_to_last(self):
        # TODO 在不同模式下，跳转到上一个点的方式不同。
        if configs.mode == "horizontal_first":
            self.current_row, self.current_col = self.jump_to_last_specific(self.current_row, self.current_col, self.row_num, self.col_num)
        elif configs.mode == "vertical_first":
            self.current_col, self.current_row = self.jump_to_last_specific(self.current_col, self.current_row, self.col_num, self.row_num)
        elif configs.mode == "random":
            self.current_col, self.current_row = self.jump_to_random()

        if self.current_row == -1 or self.current_col == -1:
            self.current_row = self.row_num - 1
            self.current_col = self.col_num - 1

    def jump_to_next_specific(self, layer_1_index, layer_2_index, layer_1_range, layer_2_range):
        layer_2_index += 1
        if layer_2_range <= layer_2_index:
            layer_2_index = 0
            layer_1_index += 1

        return layer_1_index, layer_2_index

    def jump_to_last_specific(self, layer_1_index, layer_2_index, layer_1_range, layer_2_range):
        layer_2_index -= 1
        if layer_2_range < 0:
            layer_2_index = layer_2_range - 1
            layer_1_index -= 1
        return layer_1_index, layer_2_index

    def jump_to_random(self):
        row, col = self.random_indices[self.exp_index]
        return row, col


    def capture_video(self):
        if configs.bool_capture_video:
            file_path = f"output/subject_{configs.subject_num}/{configs.mode}/col_{self.current_col}-row_{self.current_row}"
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            self.scene.removeItem(self.point)
            self.point.setBrush(QBrush(QColor(255, 0, 0)))
            self.scene.addItem(self.point)
            QCoreApplication.processEvents()

            for i in range(10):
                ret, frame = self.cap.read()
                if ret:
                    filename = f"{file_path}/capture_{i}.jpg"
                    cv2.imwrite(filename, frame)


def exp_main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

