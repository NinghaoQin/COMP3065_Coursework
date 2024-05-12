from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5 import uic
import torch
import argparse
import os
import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from utils.Utils import ResizePadding
from utils.CameraLoader import CamLoader, CamLoader_Q
from yolov5.yolov5_detector import Detector
from deep_sort import DeepSort

inp_dets = 0


def preproc(image):
    """preprocess function for CameraLoader.
    """
    resize_fn = ResizePadding(inp_dets, inp_dets)
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


class Ui_MainWindow(object):
    def __init__(self):

        self.ui = uic.loadUi('./ui/main_ui_1280.ui')
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        size = self.ui.geometry()
        self.ui.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2 - 35)
        self.ui.setWindowIcon(QIcon("./ui/image/logo.ico"))

        # ----------------------------------------
        # 变量初始化
        # ----------------------------------------
        self.tag = 1
        self.video_path = None
        self.timer_camera = QtCore.QTimer()
        self.timer_camera_load = QtCore.QTimer()
        self.timer_video = QtCore.QTimer()
        self.flag_timer = ""
        self.CAM_NUM = 0
        self.cap = cv2.VideoCapture(self.CAM_NUM)
        self.cap_video = None
        self.stop = False
        self.count = 0
        self.start_time = 0
        self.ui.toolButton_video.clicked.connect(self.select_video)
        self.ui.toolButton_camera.clicked.connect(self.open_camera)
        self.ui.start.clicked.connect(self.start)
        self.ui.cancel.clicked.connect(self.stop_start)

    def select_video(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self.ui.centralwidget, "Chose File",
                                                                self.video_path,
                                                                "movie(*.mp4;*.avi)")
        self.video_path = fileName_choose
        if fileName_choose != '':
            self.flag_timer = "video"
            if len(fileName_choose) > 10:
                fileName_choose = fileName_choose[:5] + '...' + fileName_choose[-5:]
            self.ui.toolButton_video.setText("Ready")
            self.ui.toolButton_camera.setText("Camera")

            QtWidgets.QApplication.processEvents()
            try:
                self.cap_video = cv2.VideoCapture(fileName_choose)
            except:
                print("[INFO] could not determine # of frames in video")

    def open_camera(self):
        self.video_path = '0'
        self.ui.toolButton_camera.setText("Ready")
        self.ui.toolButton_video.setText("Upload")

    def start(self):
        source = self.video_path
        self.stop = False
        self.ui.start.setText("RUNNING...")
        par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
        par.add_argument('-C', '--camera', default=source,
                         # required=True,  # default=2,
                         help='Source of camera or video file path.')
        par.add_argument('--detection_input_size', type=int, default=384,
                         help='Size of input in detection model in square must be divisible by 32 (int).')
        args = par.parse_args()

        # DETECTION MODEL.
        global inp_dets
        inp_dets = args.detection_input_size

        detect_model = Detector()

        # Tracker.
        tracker = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")

        cam_source = args.camera
        if type(cam_source) is str and os.path.isfile(cam_source):
            # Use loader thread with Q for video file.
            cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
        else:
            # Use normal thread loader for webcam.
            cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                            preprocess=preproc).start()

        color_palette = np.linspace(0, 255, 20, dtype=np.uint8)
        color_dict = {}
        fps_time = 0
        f = 0
        while cam.grabbed():
            f += 1
            frame = cam.getitem()
            image = frame.copy()

            # Detect humans bbox in the frame with detector model.
            _, detected =  detect_model.detect(frame)

            outputs = []
            if len(detected) != 0:
                bbox_xywh = [[(bb[0]+bb[2])/2, (bb[1]+bb[3])/2, bb[2]-bb[0], bb[3]-bb[1]] for bb in detected]
                outputs = tracker.update((torch.Tensor(bbox_xywh)), ([bb[4] for bb in detected]), frame)

            for i, (track_id, bbox, trace) in enumerate(outputs):
                if track_id not in color_dict:
                    color_dict[track_id] = cv2.cvtColor(np.uint8([[[color_palette[len(color_dict)%20], 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
                color = color_dict[track_id]
                color = (int(color[0]),int(color[1]),int(color[2]))
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (int(color[0]),int(color[1]),int(color[2])), thickness=1)
                frame = cv2.putText(frame, str(track_id), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_COMPLEX, 0.4,
                                    (255,0,0), 2)
                for point in trace:
                    frame = cv2.circle(frame, (int(point[0]), int(point[1])), 3, color=color,thickness=-1)

            # Show Frame.
            frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
            frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            frame = frame[:, :, ::-1]
            fps_time = time.time()

            cv2.waitKey(1)
            if self.stop:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            qt_image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.ui.label_display.setPixmap(pixmap)

        # Clear resource.
        cam.stop()

        cv2.destroyAllWindows()

    def stop_start(self):
        self.stop = True
        self.ui.toolButton_video.setText("Upload")
        self.ui.toolButton_camera.setText("Camera")
        self.ui.start.setText("RUN")
        self.ui.label_display.clear()
        self.ui.label_display.setPixmap(QtGui.QPixmap("./ui/image/t.png"))


if __name__ == '__main__':
    detect_app = QApplication([])
    detect_ui = Ui_MainWindow()
    detect_ui.ui.show()
    detect_app.exec_()
