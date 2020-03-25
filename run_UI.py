from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel


import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from ui.ui import Ui_Form
from ui.mouse_event import GraphicsScene
import cv2
import skimage.io
from ui.util import number_color, color_pred
import qdarkstyle
import qdarkgraystyle
import os
import numpy as np
from PyQt5 import QtGui
import datetime
import skimage.io


from data.base_dataset import get_params, get_transform
from PIL import Image
import os
import torch
from util.util import tensor2im
from glob import glob
import copy

class ExWindow(QMainWindow):
    def __init__(self, opt):
        super().__init__()
        self.EX = Ex(opt)
        # self.setWindowIcon(QtGui.QIcon('icons/kaust_logo.svg'))


class Ex(QWidget, Ui_Form):
    def __init__(self, opt):

        super().__init__()
        self.init_deep_model(opt)

        self.setupUi(self)
        self.show()

        self.modes = 0
        self.alpha = 1

        self.mouse_clicked = False
        self.scene = GraphicsScene(self.modes, self)
        self.scene.setSceneRect(0, 0, 512, 512)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignCenter)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.result_scene = QGraphicsScene()
        self.graphicsView_2.setScene(self.result_scene)
        self.graphicsView_2.setAlignment(Qt.AlignCenter)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.GT_scene = QGraphicsScene()
        self.graphicsView_GT.setScene(self.GT_scene)
        self.graphicsView_GT.setAlignment(Qt.AlignCenter)
        self.graphicsView_GT.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_GT.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


        self.dlg = QColorDialog(self.graphicsView)

        self.init_screen()

    def init_screen(self):
        #self.image = QPixmap(self.graphicsView.size())
        self.image = QPixmap(QSize(512, 512))
        self.image.fill(QColor('#000000'))
        self.mat_img = np.zeros([512, 512, 3], np.uint8)


        self.mat_img_org = self.mat_img.copy()

        self.GT_img_path = None
        GT_img = self.mat_img.copy()
        self.GT_img = Image.fromarray(GT_img)
        self.GT_img = self.GT_img.convert('RGB')

        #################### add GT image
        self.update_GT_image(GT_img)

        #####################


        self.scene.reset()
        if len(self.scene.items()) > 0:
            self.scene.reset_items()
        self.scene.addPixmap(self.image)

        ###############

        ############### load average features

        self.load_average_feature()
        self.run_deep_model()
        self.recorded_img_names = []



    def init_deep_model(self, opt):
        self.opt = opt
        self.model = Pix2PixModel(self.opt)
        self.model.eval()

    def run_deep_model(self):
        torch.manual_seed(0)

        data_i = self.get_single_input()

        if self.obj_dic is not None:
            data_i['obj_dic'] = self.obj_dic


        generated = self.model(data_i, mode='UI_mode')
        generated_img = self.convert_output_image(generated)
        qim = QImage(generated_img.data, generated_img.shape[1], generated_img.shape[0], QImage.Format_RGB888)


        if len(self.result_scene.items()) > 0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(QPixmap.fromImage(qim).scaled(QSize(512,512),transformMode=Qt.SmoothTransformation))
        self.generated_img = generated_img


    @pyqtSlot()
    def open(self):

        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                                                  QDir.currentPath() + '/imgs/colormaps')
        if fileName:
            image = QPixmap(fileName)
            self.mat_img_path = os.path.join(self.opt.label_dir, os.path.basename(fileName))

            # USE CV2 read images, because of using gray scale images, no matter the RGB orders

            mat_img = cv2.imread(self.mat_img_path)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return
            # self.image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            self.image = image.scaled(QSize(512, 512), Qt.IgnoreAspectRatio)

            self.mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_NEAREST)
            self.mat_img_org = self.mat_img.copy()

            self.GT_img_path = os.path.join(self.opt.image_dir, os.path.basename(fileName)[:-4] + '.jpg')
            GT_img = skimage.io.imread(self.GT_img_path)
            self.GT_img = Image.fromarray(GT_img)
            self.GT_img = self.GT_img.convert('RGB')

            self.input_img_button.setIcon(QIcon(self.GT_img_path))


            #################### add GT image
            self.update_GT_image(GT_img)

            #####################


            self.scene.reset()
            if len(self.scene.items()) > 0:
                self.scene.reset_items()
            self.scene.addPixmap(self.image)

            self.load_input_feature()
            self.run_deep_model()

    @pyqtSlot()
    def change_brush_size(self):
        self.scene.brush_size = self.brushSlider.value()
        self.brushsizeLabel.setText('Brush size: %d' % self.scene.brush_size)

    @pyqtSlot()
    def change_alpha_value(self):
        self.alpha = self.alphaSlider.value() / 20
        self.alphaLabel.setText('Alpha: %.2f' % self.alpha)


    @pyqtSlot()
    def mode_select(self, mode):
        self.modes = mode
        self.scene.modes = mode

        if mode == 0:
            self.brushButton.setStyleSheet("background-color: #85adad")
            self.recButton.setStyleSheet("background-color:")
            self.fillButton.setStyleSheet("background-color:")
            QApplication.setOverrideCursor(Qt.ArrowCursor)
        elif mode == 1:
            self.recButton.setStyleSheet("background-color: #85adad")
            self.brushButton.setStyleSheet("background-color:")
            self.fillButton.setStyleSheet("background-color:")
            QApplication.setOverrideCursor(Qt.ArrowCursor)
        elif mode == 2:
            self.fillButton.setStyleSheet("background-color: #85adad")
            self.brushButton.setStyleSheet("background-color:")
            self.recButton.setStyleSheet("background-color:")
            QApplication.setOverrideCursor(Qt.PointingHandCursor)




    @pyqtSlot()
    def save_img(self):


        current_time = datetime.datetime.now()
        ui_result_folder = 'ui_results'

        if not os.path.exists(ui_result_folder):
            os.mkdir(ui_result_folder)

        skimage.io.imsave(os.path.join(ui_result_folder, str(current_time) +'_G_img.png'), self.generated_img)
        skimage.io.imsave(os.path.join(ui_result_folder, str(current_time) +'_I.png'), self.mat_img[:, :, 0])
        skimage.io.imsave(os.path.join(ui_result_folder, str(current_time) +'_ColorI.png'), color_pred(self.mat_img[:, :, 0]))


    @pyqtSlot()
    def switch_labels(self, label):
        self.scene.label = label
        self.scene.color = number_color[label]
        self.color_Button.setStyleSheet("background-color: %s;" % self.scene.color)


    @pyqtSlot()
    def undo(self):
        self.scene.undo()



    # get input images and labels
    def get_single_input(self):

        image_path = self.GT_img_path
        image = self.GT_img
        label_img = self.mat_img[:, :, 0]


        label = Image.fromarray(label_img)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        label_tensor.unsqueeze_(0)

        image_tensor = torch.zeros([1, 3, 256, 256])


        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = torch.Tensor([0])

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }



        return input_dict


    def convert_output_image(self,generated):
        tile = self.opt.batchSize > 8
        t = tensor2im(generated, tile=tile)[0]
        return t

    def update_GT_image(self, GT_img):
        qim = QImage(GT_img.data, GT_img.shape[1], GT_img.shape[0], GT_img.strides[0],
                     QImage.Format_RGB888)
        qim = qim.scaled(QSize(256, 256), Qt.IgnoreAspectRatio, transformMode=Qt.SmoothTransformation)
        if len(self.GT_scene.items()) > 0:
            self.GT_scene.removeItem(self.GT_scene.items()[-1])
        self.GT_scene.addPixmap(QPixmap.fromImage(qim).scaled(QSize(512, 512),transformMode=Qt.SmoothTransformation))

    def load_average_feature(self):

        ############### load average features

        average_style_code_folder = 'styles_test/mean_style_code/mean/'
        input_style_dic = {}

        ############### hard coding for categories

        for i in range(19):
            input_style_dic[str(i)] = {}

            average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
            average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in
                                     average_category_folder_list]

            for style_code_path in average_category_list:
                    input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                        np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()

        self.obj_dic = input_style_dic
        # self.obj_dic_back = copy.deepcopy(self.obj_dic)

    def load_partial_average_feature(self):

        average_style_code_folder = 'styles_test/mean_style_code/mean/'


        for i, cb_status in enumerate(self.checkbox_status):
            if cb_status:

                average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
                average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in
                                         average_category_folder_list]

                for style_code_path in average_category_list:
                    self.obj_dic[str(i)][style_code_path] = torch.from_numpy(
                        np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()
                if str(i) in self.style_img_mask_dic:
                    del self.style_img_mask_dic[str(i)]

        self.run_deep_model()
        self.update_snapshots()




    def load_input_feature(self):

        ############### load average features

        average_style_code_folder = 'styles_test/mean_style_code/mean/'
        input_style_code_folder = 'styles_test/style_codes/' + os.path.basename(self.GT_img_path)
        input_style_dic = {}
        self.label_count = []


        self.style_img_mask_dic = {}

        for i in range(19):
            input_style_dic[str(i)] = {}

            input_category_folder_list = glob(os.path.join(input_style_code_folder, str(i), '*.npy'))
            input_category_list = [os.path.splitext(os.path.basename(name))[0] for name in input_category_folder_list]

            average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
            average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in average_category_folder_list]

            for style_code_path in average_category_list:
                if style_code_path in input_category_list:
                    input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                        np.load(os.path.join(input_style_code_folder, str(i), style_code_path+'.npy'))).cuda()

                    if style_code_path == 'ACE':
                        self.style_img_mask_dic[str(i)] = self.GT_img_path
                        self.label_count.append(i)

                else:
                    input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                        np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()

        self.obj_dic = input_style_dic
        #self.obj_dic_back = copy.deepcopy(self.obj_dic)
        self.obj_dic_GT = copy.deepcopy(self.obj_dic)

        self.update_snapshots()


    def style_linear_interpolation(self):

        ui_result_folder = 'style_interpolation'


        img_list = glob('imgs/style_imgs_test/*.jpg')
        img_list.sort()

        for style_count,_ in enumerate(img_list):
            if style_count == len(img_list) - 1:
                break
            style_path_1 = img_list[style_count]
            style_path_2 = img_list[style_count + 1]



            style_path_1_folder = 'styles_test/style_codes/' + os.path.basename(style_path_1)
            style_path_2_folder = 'styles_test/style_codes/' + os.path.basename(style_path_2)





            for count_num in range(1, 21):
                alpha = count_num * 0.05


                for i, cb_status in enumerate(self.checkbox_status):


                    if cb_status and i in self.label_count:
                        input_category_folder_list_1 = glob(os.path.join(style_path_1_folder, str(i), '*.npy'))
                        input_category_list_1 = [os.path.splitext(os.path.basename(name))[0] for name in input_category_folder_list_1]

                        input_category_folder_list_2 = glob(os.path.join(style_path_2_folder, str(i), '*.npy'))
                        input_category_list_2 = [os.path.splitext(os.path.basename(name))[0] for name in input_category_folder_list_2]

                        if 'ACE' in input_category_list_1:
                            style_code1 = torch.from_numpy(np.load(os.path.join(style_path_1_folder, str(i), 'ACE.npy'))).cuda()
                        else:
                            style_code1 = self.obj_dic_GT[str(i)]['ACE']


                        if 'ACE' in input_category_list_2:
                            style_code2 = torch.from_numpy(np.load(os.path.join(style_path_2_folder, str(i), 'ACE.npy'))).cuda()
                        else:
                            style_code2 = self.obj_dic_GT[str(i)]['ACE']

                        self.obj_dic[str(i)]['ACE'] = (1 - alpha) * style_code1 + alpha * style_code2

                self.run_deep_model()

                if count_num < 20:
                    skimage.io.imsave(os.path.join(ui_result_folder, os.path.basename(style_path_1)[:-4] + '_' + os.path.basename(style_path_2)[:-4] + '_' + str(count_num) + '.png'), self.generated_img)
                else:
                    skimage.io.imsave(os.path.join(ui_result_folder, os.path.basename(style_path_2)[:-4] + '.png'), self.generated_img)







    def update_entire_feature(self, style_img_path):


        if style_img_path == 0:
            style_img_path = self.GT_img_path
            if style_img_path == None:
                return
            input_style_code_folder = 'styles_test/style_codes/' + os.path.basename(style_img_path)
        else:
            input_style_code_folder = 'styles_test/style_codes/' + os.path.basename(style_img_path)



        for i, cb_status in enumerate(self.checkbox_status):


            if cb_status and i in self.label_count:
                input_category_folder_list = glob(os.path.join(input_style_code_folder, str(i), '*.npy'))
                input_category_list = [os.path.splitext(os.path.basename(name))[0] for name in
                                       input_category_folder_list]

                style_code_path = 'ACE'
                if style_code_path in input_category_list:

                    if self.alpha == 1:
                        self.obj_dic[str(i)][style_code_path] = torch.from_numpy(
                            np.load(os.path.join(input_style_code_folder, str(i), style_code_path + '.npy'))).cuda()
                    else:
                        ##################### some problems here. using the same list dic
                        self.obj_dic[str(i)][style_code_path] = self.alpha * torch.from_numpy(
                            np.load(os.path.join(input_style_code_folder, str(i), style_code_path + '.npy'))).cuda() + (1- self.alpha) * self.obj_dic_GT[str(i)][style_code_path]


                    if style_code_path == 'ACE':
                        self.style_img_mask_dic[str(i)] = style_img_path

                elif os.path.exists(os.path.join('styles_test/style_codes', os.path.basename(self.GT_img_path),str(i),style_code_path + '.npy')):
                    if self.alpha == 1:
                        self.obj_dic[str(i)][style_code_path] = torch.from_numpy(
                            np.load(os.path.join('styles_test/style_codes', os.path.basename(self.GT_img_path), str(i), style_code_path + '.npy'))).cuda()
                    else:
                        self.obj_dic[str(i)][style_code_path] = self.alpha * torch.from_numpy(
                            np.load(os.path.join('styles_test/style_codes', os.path.basename(self.GT_img_path), str(i), style_code_path + '.npy'))).cuda() + (1- self.alpha) * self.obj_dic_GT[str(i)][style_code_path]


                    if style_code_path == 'ACE':
                        self.style_img_mask_dic[str(i)] = self.GT_img_path

        self.run_deep_model()
        self.update_snapshots()
        self.show_reference_image(style_img_path)


    def show_reference_image(self, im_name):

        qim = QImage(im_name).scaled(QSize(256, 256),transformMode=Qt.SmoothTransformation)
        # self.referDialogImage.setPixmap(QPixmap.fromImage(qim).scaled(QSize(512, 512), transformMode=Qt.SmoothTransformation))
        # # self.referDialog.setWindowTitle('Input:' + os.path.basename(self.GT_img_path) + '\t \t Reference:' + os.path.basename(im_name))
        # self.referDialog.show()

        self.GT_scene.addPixmap(QPixmap.fromImage(qim).scaled(QSize(512, 512), transformMode=Qt.SmoothTransformation))


    def update_snapshots(self):
        self.clean_snapshots()
        self.recorded_img_names = np.unique(list(self.style_img_mask_dic.values()))
        self.recorded_mask_dic = {}

        tmp_count = 0


        for i, name in enumerate(self.recorded_img_names):
            self.recorded_mask_dic[name] = [int(num) for num in self.style_img_mask_dic if self.style_img_mask_dic[num]==name]


            ########## show mask option 1: masks of the style image
            rgb_mask = skimage.io.imread(os.path.join(os.path.dirname(self.opt.label_dir), 'vis', os.path.basename(name)[:-4] + '.png'))
            gray_mask = skimage.io.imread(os.path.join(self.opt.label_dir, os.path.basename(name)[:-4] + '.png'))


            mask_snap = np.where(np.isin(np.repeat(np.expand_dims(gray_mask,2),3, axis=2), self.recorded_mask_dic[name]), rgb_mask, 255)


            if not (mask_snap==255).all():
                self.mask_snap_style_button_list[tmp_count].setIcon(QIcon(QPixmap.fromImage(QImage(mask_snap.data, mask_snap.shape[1], mask_snap.shape[0], mask_snap.strides[0],
                         QImage.Format_RGB888))))


                self.snap_style_button_list[tmp_count].setIcon(QIcon(name))
                tmp_count += 1




    def clean_snapshots(self):
        for snap_style_button in self.snap_style_button_list:
            snap_style_button.setIcon(QIcon())
        for mask_snap_style_button in self.mask_snap_style_button_list:
            mask_snap_style_button.setIcon(QIcon())


    def open_snapshot_dialog(self, i):
        if i < len(self.recorded_img_names):
            im_name = self.recorded_img_names[i]
            qim = QImage(im_name).scaled(QSize(256, 256), transformMode=Qt.SmoothTransformation)
            self.snapshotDialogImage.setPixmap(
                QPixmap.fromImage(qim).scaled(QSize(512, 512), transformMode=Qt.SmoothTransformation))
            self.snapshotDialog.setWindowTitle('Reference:' + os.path.basename(im_name))
            self.snapshotDialog.show()
            self.snapshotDialog.count = i
        else:
            self.snapshotDialog.setWindowTitle('Reference:')
            self.snapshotDialogImage.setPixmap(QPixmap())
            self.snapshotDialog.show()
            self.snapshotDialog.count = i




if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.status = 'UI_mode'

    app = QApplication(sys.argv)
    #app.setStyleSheet(qdarkgraystyle.load_stylesheet())
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ex = ExWindow(opt)
    # ex = Ex(opt)
    sys.exit(app.exec_())