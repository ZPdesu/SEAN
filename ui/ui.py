from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from .util import number_color
from functools import partial
import glob
from ui.util import number_object
from ui.mouse_event import ReferenceDialog, SnapshotDialog
import copy


Lb_width = 100
Lb_height = 40
Lb_row_shift = 25
Lb_col_shift = 5
Lb_x = 100
Lb_y = 690


Tb_width = 100
Tb_height = 40
Tb_row_shift = 50
Tb_col_shift = 5
Tb_x = 100
Tb_y = 60



_translate = QtCore.QCoreApplication.translate



class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1920, 1080)

        # Form.resize(1980, 1100)


        # Label Buttons to change the semantic meanings of the Brush
        # First Row
        self.add_brush_widgets(Form)
        self.add_top_buttons(Form)
        self.add_label_buttons(Form)
        self.add_tool_buttons(Form)
        self.add_checkbox_widgets(Form)
        self.add_input_img_button(Form)




        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setGeometry(QtCore.QRect(652, 140, 518, 518))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_2.setGeometry(QtCore.QRect(1204, 140, 518, 518))
        self.graphicsView_2.setObjectName("graphicsView_2")

        self.graphicsView_GT = QtWidgets.QGraphicsView(Form)
        self.graphicsView_GT.setGeometry(QtCore.QRect(100, 140, 518, 518))
        self.graphicsView_GT.setObjectName("graphicsView_GT")


        self.referDialog = ReferenceDialog(self)
        self.referDialog.setObjectName('Reference Dialog')
        # self.referDialog.setWindowTitle('Reference Image:')
        self.referDialog.setWindowTitle('Style Image')
        self.referDialogImage = QtWidgets.QLabel(self.referDialog)
        self.referDialogImage.setFixedSize(512, 512)
        # self.referDialog.show()


        self.snapshotDialog = SnapshotDialog(self)
        self.snapshotDialog.setObjectName('Snapshot Dialog')
        self.snapshotDialog.setWindowTitle('Reference Image:')
        self.snapshotDialogImage = QtWidgets.QLabel(self.snapshotDialog)
        self.snapshotDialogImage.setFixedSize(512, 512)

        self.add_intermediate_results_button(Form)
        self.add_alpha_bar(Form)



        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        # Form.setWindowTitle(_translate("Form", "Let's Party Face Manipulation v0.2"))
        Form.setWindowTitle(_translate("Form", "Let's Party Face Manipulation"))
        self.pushButton.setText(_translate("Form", "Open Image"))
        self.pushButton_2.setText(_translate("Form", "Mask"))
        self.pushButton_3.setText(_translate("Form", "Sketches"))
        self.pushButton_4.setText(_translate("Form", "Color"))

        self.saveImg.setText(_translate("Form", "Save Img"))

    def add_alpha_bar(self, Form):


        self.alphaLabel = QtWidgets.QLabel(Form)
        self.alphaLabel.setObjectName("alphaLabel")
        self.alphaLabel.setGeometry(QtCore.QRect(Lb_x + 10 * Lb_row_shift + 10 * Lb_width + 40, Lb_y, 150, 20))
        self.alphaLabel.setText('Alpha: 1.0')
        font = self.brushsizeLabel.font()
        font.setPointSize(10)
        font.setBold(True)
        self.alphaLabel.setFont(font)

        self.alphaSlider = QtWidgets.QSlider(Form)
        self.alphaSlider.setOrientation(QtCore.Qt.Horizontal)
        self.alphaSlider.setGeometry(QtCore.QRect(Lb_x + 10 * Lb_row_shift + 10 * Lb_width + 150, Lb_y, 225, 10))
        self.alphaSlider.setObjectName("alphaSlider")
        self.alphaSlider.setMinimum(0)
        self.alphaSlider.setMaximum(20)
        self.alphaSlider.setValue(20)
        self.alphaSlider.valueChanged.connect(Form.change_alpha_value)



    def add_intermediate_results_button(self, Form):

        self.snap_scrollArea = QtWidgets.QScrollArea(Form)
        self.snap_scrollArea.setGeometry(QtCore.QRect(100, Lb_y + Lb_height + Lb_col_shift + Lb_height + 30, 1622, 250))
        self.snap_scrollArea.setWidgetResizable(True)
        self.snap_scrollArea.setObjectName("snap_scrollArea")
        self.snap_scrollArea.setAlignment(Qt.AlignCenter)
        #self.snap_scrollArea.setStyleSheet("border-color: transparent")
        self.snap_scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


        self.snap_scrollAreaWidgetContents = QtWidgets.QWidget()
        self.snap_scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1622, 250))
        self.snap_scrollAreaWidgetContents.setObjectName("snap_scrollAreaWidgetContents")

        self.snap_gridlLayout = QtWidgets.QGridLayout(self.snap_scrollAreaWidgetContents)
        # # snap_horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.snap_gridlLayout.setSpacing(20)
        self.snap_gridlLayout.setAlignment(Qt.AlignLeft)

        self.snap_style_button_list = []
        self.mask_snap_style_button_list = []

        for i in range(15):
            snap_style_button = QtWidgets.QPushButton()
            snap_style_button.setFixedSize(100, 100)
            snap_style_button.setStyleSheet("background-color: transparent")
            snap_style_button.setIcon(QIcon())
            snap_style_button.setIconSize(QSize(100, 100))
            snap_style_button.clicked.connect(partial(self.open_snapshot_dialog, i))
            # snap_style_button.snap_shot_name = None
            self.snap_style_button_list.append(snap_style_button)
            # style_button.hide()
            self.snap_gridlLayout.addWidget(snap_style_button, 1, i)


            mask_snap_style_button = QtWidgets.QPushButton()
            mask_snap_style_button.setFixedSize(100, 100)
            mask_snap_style_button.setStyleSheet("background-color: transparent")
            mask_snap_style_button.setIcon(QIcon())
            mask_snap_style_button.setIconSize(QSize(100, 100))
            self.mask_snap_style_button_list.append(mask_snap_style_button)
            # mask_snap_style_button.hide()
            self.snap_gridlLayout.addWidget(mask_snap_style_button, 0, i)


        self.snap_scrollArea.setWidget(self.snap_scrollAreaWidgetContents)






    def add_input_img_button(self, Form):
        self.input_img_button = QtWidgets.QPushButton(Form)
        self.input_img_button.setGeometry(QtCore.QRect(1770, 15, 100, 100))
        self.input_img_button.setStyleSheet("background-color: transparent")
        self.input_img_button.setFixedSize(100, 100)
        self.input_img_button.setIcon(QIcon(None))
        self.input_img_button.setIconSize(QSize(100, 100))
        self.input_img_button.clicked.connect(partial(Form.update_entire_feature, 0))


    def add_checkbox_widgets(self, Form):
        self.checkBoxGroupBox = QtWidgets.QGroupBox("Replace Style of Components", Form)
        self.checkBoxGroupBox.setGeometry(QtCore.QRect(920, 10, 800, 100))

        layout = QtWidgets.QGridLayout()
        self.checkBoxGroup = QtWidgets.QButtonGroup(Form)
        self.checkBoxGroup.setExclusive(False)
        for i, j in enumerate(number_object):
            cb = QtWidgets.QCheckBox(number_object[j])
            self.checkBoxGroup.addButton(cb, i)
            layout.addWidget(cb, i//10, i%10)

        cb = QtWidgets.QCheckBox('ALL')
        self.checkBoxGroup.addButton(cb, )
        layout.addWidget(cb, (i+1)//10, (i+1)%10)

        self.checkBoxGroupBox.setLayout(layout)

        for i in range(19):
            self.checkBoxGroup.button(i).setChecked(True)

        checkbox_status = [cb.isChecked() for cb in self.checkBoxGroup.buttons()]
        checkbox_status = checkbox_status[:19]
        self.checkbox_status = checkbox_status
        self.checkBoxGroup.buttonToggled.connect(self.cb_event)






    def add_brush_widgets(self, Form):
        KaustLogo = QtWidgets.QLabel(self)
        # KaustLogo.setPixmap(QPixmap('icons/kaust_logo.svg').scaled(60, 60))
        KaustLogo.setPixmap(QPixmap('icons/1999780_200.png').scaled(60, 60))
        KaustLogo.setGeometry(QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60), 25, 80, 80))


        self.add_style_imgs_buttons(Form)

        self.brushsizeLabel = QtWidgets.QLabel(Form)
        self.brushsizeLabel.setObjectName("brushsizeLabel")
        self.brushsizeLabel.setGeometry(QtCore.QRect(Tb_x, 25, 150, 20))
        self.brushsizeLabel.setText('Brush size: 6')
        font = self.brushsizeLabel.font()
        font.setPointSize(10)
        font.setBold(True)
        self.brushsizeLabel.setFont(font)

        self.brushSlider = QtWidgets.QSlider(Form)
        self.brushSlider.setOrientation(QtCore.Qt.Horizontal)
        self.brushSlider.setGeometry(QtCore.QRect(Tb_x + 150, 25, 600, 10))
        self.brushSlider.setObjectName("brushSlider")
        self.brushSlider.setMinimum(1)
        self.brushSlider.setMaximum(100)
        self.brushSlider.setValue(8)
        self.brushSlider.valueChanged.connect(Form.change_brush_size)





    def add_top_buttons(self, Form):
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(Tb_x, Tb_y, Tb_width, Tb_height))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(Form.open)

        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(Tb_x + 1 * Tb_row_shift + 1 * Tb_width, Tb_y, Tb_width, Tb_height))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_2.clicked.connect(Form.style_linear_interpolation)




        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(Tb_x + 2 * Tb_row_shift + 2 * Tb_width, Tb_y, Tb_width, Tb_height))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(Tb_x + 3 * Tb_row_shift + 3 * Tb_width, Tb_y, Tb_width, Tb_height))
        self.pushButton_4.setObjectName("pushButton_4")

        self.saveImg = QtWidgets.QPushButton(Form)
        self.saveImg.setGeometry(QtCore.QRect(Tb_x + 4 * Tb_row_shift + 4 * Tb_width, Tb_y, Tb_width, Tb_height))
        self.saveImg.setObjectName("saveImg")
        self.saveImg.clicked.connect(Form.save_img)
        self.retranslateUi(Form)




    def add_tool_buttons(self, Form):
        self.newButton = QtWidgets.QPushButton(Form)
        self.newButton.setGeometry(QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60), 140, 60, 60))
        self.newButton.setObjectName("openButton")
        self.newButton.setIcon(QIcon('icons/add_new_document.png'))
        self.newButton.setIconSize(QSize(60, 60))
        self.newButton.clicked.connect(Form.init_screen)

        self.openButton = QtWidgets.QPushButton(Form)
        self.openButton.setGeometry(QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60), 140 + 60*1 + 10*1, 60, 60))
        self.openButton.setObjectName("openButton")
        self.openButton.setIcon(QIcon('icons/open.png'))
        self.openButton.setIconSize(QSize(60, 60))
        self.openButton.clicked.connect(Form.open)



        self.fillButton = QtWidgets.QPushButton(Form)
        self.fillButton.setGeometry(QtCore.QRect(int(Lb_x - 1*Lb_row_shift - 60), 140 + 60*2 + 10*2, 60, 60))
        self.fillButton.setObjectName("fillButton")
        self.fillButton.setIcon(QIcon('icons/paint_can.png'))
        self.fillButton.setIconSize(QSize(60, 60))
        self.fillButton.clicked.connect(partial(Form.mode_select, 2))


        self.brushButton = QtWidgets.QPushButton(Form)
        self.brushButton.setGeometry(QtCore.QRect(int(Lb_x - 1*Lb_row_shift - 60), 140 + 60*3 + 10*3, 60, 60))
        self.brushButton.setObjectName("brushButton")
        self.brushButton.setIcon(QIcon('icons/paint_brush.png'))
        self.brushButton.setIconSize(QSize(60, 60))
        self.brushButton.setStyleSheet("background-color: #85adad")
        #self.brushButton.setStyleSheet("background-color:")
        self.brushButton.clicked.connect(partial(Form.mode_select, 0))


        self.recButton = QtWidgets.QPushButton(Form)
        self.recButton.setGeometry(QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60), 140 + 60 * 4 + 10 * 4, 60, 60))
        self.recButton.setObjectName("undolButton")
        self.recButton.setIcon(QIcon('icons/brush_square.png'))
        self.recButton.setIconSize(QSize(60, 60))
        self.recButton.clicked.connect(partial(Form.mode_select, 1))



        self.undoButton = QtWidgets.QPushButton(Form)
        self.undoButton.setGeometry(QtCore.QRect(int(Lb_x - 1*Lb_row_shift - 60), 140 + 60*5 + 10*5, 60, 60))
        self.undoButton.setObjectName("undolButton")
        self.undoButton.setIcon(QIcon('icons/undo.png'))
        self.undoButton.setIconSize(QSize(60, 60))
        self.undoButton.clicked.connect(Form.undo)

        self.saveButton = QtWidgets.QPushButton(Form)
        self.saveButton.setGeometry(QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60), 140 + 60 * 6 + 10 * 6, 60, 60))
        self.saveButton.setObjectName("saveButton")
        self.saveButton.setIcon(QIcon('icons/save.png'))
        self.saveButton.setIconSize(QSize(60, 60))
        self.saveButton.clicked.connect(Form.save_img)


    def add_style_imgs_buttons(self, Form):

        self.scrollArea = QtWidgets.QScrollArea(Form)
        self.scrollArea.setGeometry(QtCore.QRect(1756, 140, 140, 512))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        self.scrollArea.setAlignment(Qt.AlignCenter)
        # self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 140, 512))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")


        verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        verticalLayout.setContentsMargins(11, 11, 11, 11)
        verticalLayout.setSpacing(6)


        img_path_list = glob.glob('imgs/style_imgs_test/*.jpg')
        img_path_list.sort()

        style_button = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        style_button.setFixedSize(100, 100)
        style_button.setIcon(QIcon('icons/random.png'))
        style_button.setIconSize(QSize(100, 100))
        style_button.clicked.connect(Form.load_partial_average_feature)
        verticalLayout.addWidget(style_button)

        for img_path in img_path_list:
            style_button = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
            style_button.setFixedSize(100, 100)
            style_button.setIcon(QIcon(img_path))
            style_button.setIconSize(QSize(100, 100))
            style_button.clicked.connect(partial(Form.update_entire_feature, img_path))
            verticalLayout.addWidget(style_button)


        verticalLayout.addWidget(style_button)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)



    def add_label_buttons(self, Form):

        self.color_Button = QtWidgets.QPushButton(Form)
        self.color_Button.setGeometry(QtCore.QRect(int(Lb_x - 1*Lb_row_shift - 60), Lb_y, 60, 60))
        self.color_Button.setObjectName("labelButton_0")
        self.color_Button.setStyleSheet("background-color: %s;" % number_color[1])



        self.labelButton_0 = QtWidgets.QPushButton(Form)
        self.labelButton_0.setGeometry(QtCore.QRect(Lb_x, Lb_y, Lb_width, Lb_height))
        self.labelButton_0.setObjectName("labelButton_0")
        self.labelButton_0.setText(_translate("Form", "background"))
        self.labelButton_0.setStyleSheet("background-color: %s;" % number_color[0]+ " color: black")
        self.labelButton_0.clicked.connect(partial(Form.switch_labels, 0))



        self.labelButton_1 = QtWidgets.QPushButton(Form)
        self.labelButton_1.setGeometry(QtCore.QRect(Lb_x + 1*Lb_row_shift + 1*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_1.setObjectName("labelButton_1")
        self.labelButton_1.setText(_translate("Form", "skin"))
        self.labelButton_1.setStyleSheet("background-color: %s;" % number_color[1] + " color: black")
        self.labelButton_1.clicked.connect(partial(Form.switch_labels, 1))


        self.labelButton_2 = QtWidgets.QPushButton(Form)
        self.labelButton_2.setGeometry(QtCore.QRect(Lb_x + 2*Lb_row_shift + 2*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_2.setObjectName("labelButton_2")
        self.labelButton_2.setText(_translate("Form", "nose"))
        self.labelButton_2.setStyleSheet("background-color: %s;" % number_color[2] + " color: black")
        self.labelButton_2.clicked.connect(partial(Form.switch_labels, 2))


        self.labelButton_3 = QtWidgets.QPushButton(Form)
        self.labelButton_3.setGeometry(QtCore.QRect(Lb_x + 3*Lb_row_shift + 3*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_3.setObjectName("labelButton_3")
        self.labelButton_3.setText(_translate("Form", "eye_g"))
        self.labelButton_3.setStyleSheet("background-color: %s;" % number_color[3] + " color: black")
        self.labelButton_3.clicked.connect(partial(Form.switch_labels, 3))


        self.labelButton_4 = QtWidgets.QPushButton(Form)
        self.labelButton_4.setGeometry(QtCore.QRect(Lb_x + 4*Lb_row_shift + 4*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_4.setObjectName("labelButton_4")
        self.labelButton_4.setText(_translate("Form", "l_eye"))
        self.labelButton_4.setStyleSheet("background-color: %s;" % number_color[4] + " color: black")
        self.labelButton_4.clicked.connect(partial(Form.switch_labels, 4))


        self.labelButton_5 = QtWidgets.QPushButton(Form)
        self.labelButton_5.setGeometry(QtCore.QRect(Lb_x + 5*Lb_row_shift + 5*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_5.setObjectName("labelButton_5")
        self.labelButton_5.setText(_translate("Form", "r_eye"))
        self.labelButton_5.setStyleSheet("background-color: %s;" % number_color[5] + " color: black")
        self.labelButton_5.clicked.connect(partial(Form.switch_labels, 5))


        self.labelButton_6 = QtWidgets.QPushButton(Form)
        self.labelButton_6.setGeometry(QtCore.QRect(Lb_x + 6*Lb_row_shift + 6*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_6.setObjectName("labelButton_6")
        self.labelButton_6.setText(_translate("Form", "l_brow"))
        self.labelButton_6.setStyleSheet("background-color: %s;" % number_color[6] + " color: black")
        self.labelButton_6.clicked.connect(partial(Form.switch_labels, 6))


        self.labelButton_7 = QtWidgets.QPushButton(Form)
        self.labelButton_7.setGeometry(QtCore.QRect(Lb_x + 7*Lb_row_shift + 7*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_7.setObjectName("labelButton_7")
        self.labelButton_7.setText(_translate("Form", "r_brow"))
        self.labelButton_7.setStyleSheet("background-color: %s;" % number_color[7] + " color: black")
        self.labelButton_7.clicked.connect(partial(Form.switch_labels, 7))


        self.labelButton_8 = QtWidgets.QPushButton(Form)
        self.labelButton_8.setGeometry(QtCore.QRect(Lb_x + 8*Lb_row_shift + 8*Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_8.setObjectName("labelButton_8")
        self.labelButton_8.setText(_translate("Form", "l_ear"))
        self.labelButton_8.setStyleSheet("background-color: %s;" % number_color[8] + " color: black")
        self.labelButton_8.clicked.connect(partial(Form.switch_labels, 8))

        self.labelButton_9 = QtWidgets.QPushButton(Form)
        self.labelButton_9.setGeometry(QtCore.QRect(Lb_x + 9 * Lb_row_shift + 9 * Lb_width, Lb_y, Lb_width, Lb_height))
        self.labelButton_9.setObjectName("labelButton_9")
        self.labelButton_9.setText(_translate("Form", "r_ear"))
        self.labelButton_9.setStyleSheet("background-color: %s;" % number_color[9] + " color: black")
        self.labelButton_9.clicked.connect(partial(Form.switch_labels, 9))


        # Second Row


        self.labelButton_10 = QtWidgets.QPushButton(Form)
        self.labelButton_10.setGeometry(QtCore.QRect(Lb_x,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_10.setObjectName("labelButton_10")
        self.labelButton_10.setText(_translate("Form", "mouth"))
        self.labelButton_10.setStyleSheet("background-color: %s;" % number_color[10] + " color: black")
        self.labelButton_10.clicked.connect(partial(Form.switch_labels, 10))


        self.labelButton_11 = QtWidgets.QPushButton(Form)
        self.labelButton_11.setGeometry(QtCore.QRect(Lb_x + 1*Lb_row_shift + 1*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_11.setObjectName("labelButton_11")
        self.labelButton_11.setText(_translate("Form", "u_lip"))
        self.labelButton_11.setStyleSheet("background-color: %s;" % number_color[11] + " color: black")
        self.labelButton_11.clicked.connect(partial(Form.switch_labels, 11))


        self.labelButton_12 = QtWidgets.QPushButton(Form)
        self.labelButton_12.setGeometry(QtCore.QRect(Lb_x + 2*Lb_row_shift + 2*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_12.setObjectName("labelButton_12")
        self.labelButton_12.setText(_translate("Form", "l_lip"))
        self.labelButton_12.setStyleSheet("background-color: %s;" % number_color[12] + " color: black")
        self.labelButton_12.clicked.connect(partial(Form.switch_labels, 12))


        self.labelButton_13 = QtWidgets.QPushButton(Form)
        self.labelButton_13.setGeometry(QtCore.QRect(Lb_x + 3*Lb_row_shift + 3*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_13.setObjectName("labelButton_13")
        self.labelButton_13.setText(_translate("Form", "hair"))
        self.labelButton_13.setStyleSheet("background-color: %s;" % number_color[13] + " color: black")
        self.labelButton_13.clicked.connect(partial(Form.switch_labels, 13))


        self.labelButton_14 = QtWidgets.QPushButton(Form)
        self.labelButton_14.setGeometry(QtCore.QRect(Lb_x + 4*Lb_row_shift + 4*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_14.setObjectName("labelButton_14")
        self.labelButton_14.setText(_translate("Form", "hat"))
        self.labelButton_14.setStyleSheet("background-color: %s;" % number_color[14] + " color: black")
        self.labelButton_14.clicked.connect(partial(Form.switch_labels, 14))


        self.labelButton_15 = QtWidgets.QPushButton(Form)
        self.labelButton_15.setGeometry(QtCore.QRect(Lb_x + 5*Lb_row_shift + 5*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_15.setObjectName("labelButton_15")
        self.labelButton_15.setText(_translate("Form", "ear_r"))
        self.labelButton_15.setStyleSheet("background-color: %s;" % number_color[15] + " color: black")
        self.labelButton_15.clicked.connect(partial(Form.switch_labels, 15))


        self.labelButton_16 = QtWidgets.QPushButton(Form)
        self.labelButton_16.setGeometry(QtCore.QRect(Lb_x + 6*Lb_row_shift + 6*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_16.setObjectName("labelButton_16")
        self.labelButton_16.setText(_translate("Form", "neck_l"))
        self.labelButton_16.setStyleSheet("background-color: %s;" % number_color[16] + " color: black")
        self.labelButton_16.clicked.connect(partial(Form.switch_labels, 16))


        self.labelButton_17 = QtWidgets.QPushButton(Form)
        self.labelButton_17.setGeometry(QtCore.QRect(Lb_x + 7*Lb_row_shift + 7*Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_17.setObjectName("labelButton_17")
        self.labelButton_17.setText(_translate("Form", "neck"))
        self.labelButton_17.setStyleSheet("background-color: %s;" % number_color[17] + " color: black")
        self.labelButton_17.clicked.connect(partial(Form.switch_labels, 17))

        self.labelButton_18 = QtWidgets.QPushButton(Form)
        self.labelButton_18.setGeometry(QtCore.QRect(Lb_x + 8 * Lb_row_shift + 8 * Lb_width,
                                                     Lb_y + Lb_height + Lb_col_shift, Lb_width, Lb_height))
        self.labelButton_18.setObjectName("labelButton_18")
        self.labelButton_18.setText(_translate("Form", "cloth"))
        self.labelButton_18.setStyleSheet("background-color: %s;" % number_color[18] + " color: black")
        self.labelButton_18.clicked.connect(partial(Form.switch_labels, 18))




    def cb_event(self, id, ifchecked):

        if id.text() == 'ALL':
            if ifchecked:
                for cb in self.checkBoxGroup.buttons():
                    cb.setChecked(True)
            else:
                for cb in self.checkBoxGroup.buttons():
                    cb.setChecked(False)
        self.change_cb_state()


    def change_cb_state(self):
        checkbox_status = [cb.isChecked() for cb in self.checkBoxGroup.buttons()]
        checkbox_status = checkbox_status[:19]
        #self.obj_dic_back = copy.deepcopy(self.obj_dic)
        self.checkbox_status = checkbox_status



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
