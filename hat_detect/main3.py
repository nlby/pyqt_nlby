import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from PyQt5 import QtCore, QtGui, QtWidgets
# from custom.stackedWidget import StackedWidget
from custom.treeView import FileSystemTreeView
from custom.modelwidgets import ReListWidget, FuncListWidget
from custom.customwidgets import MyLed

import os
import time, datetime
import pymysql

import colorsys
import os
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
import os
from gluoncv import model_zoo, data, utils
#from matplotlib import pyplot as plt
import mxnet as mx
import cv2
import math
import argparse
import datetime
from mxnet import nd
from gluoncv.data.transforms.presets.yolo import transform_test

from custom.graphicsView import GraphicsView
from playsound import playsound
from threading import Thread
import random

conn = pymysql.connect(host="cdb-haesuty8.bj.tencentcdb.com", port=10137, user="root",password="NLDby677",database="deep_learning",charset="utf8")

video_time_list = []
video_all_list = []
video_no_list = []
video_yes_list = []
video_table_name = ""

camera_time_list = []
camera_all_list = []
camera_no_list = []
camera_yes_list = []
camera_table_name = ""


ctx = mx.cpu()

net = model_zoo.get_model("yolo3_darknet53_voc", pretrained=False)
classes = ['hat', 'person']
for param in net.collect_params().values():
     if param._data is not None:
         continue
     param.initialize()
net.reset_class(classes)
net.collect_params().reset_ctx(ctx)
net.load_parameters('yolo3_darknet53_voc_best.params', ctx=ctx)
print('use darknet to extract feature')


class_names = ['hat', 'person']
trackerTypes = [
    'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'
]

def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


def sel_track(img, bboxes, scores, labels, trackerType="CSRT", colors=None, scale=1.0):
    from matplotlib import pyplot as plt
    from gluoncv.utils.filesystem import try_import_cv2
    for i, bbox in enumerate(bboxes):
        if labels is not None and not len(bboxes) == len(labels):
            raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()
    if len(bboxes) < 1:
        return img

    # use random colors if None is provided
    colors = dict()
    colors[0] = plt.get_cmap('hsv')(0 / len(class_names))
    colors[1] = plt.get_cmap('hsv')(1 / len(class_names))
    hat_num, person_num = 0, 0
    hat = cv2.MultiTracker_create()
    person = cv2.MultiTracker_create()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < 0.4:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        # if cls_id not in colors:
        #     if class_names is not None:
        #         colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
        #     else:
        #         colors[cls_id] = (random.random(), random.random(), random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        bcolor = [x * 255 for x in colors[cls_id]]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bcolor, 2)

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:d}%'.format(int(scores.flat[i]*100)) if scores is not None else ''
        if class_name or score:
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            cv2.putText(img, '{:s}'.format(class_name),
                        (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, min(scale/2, 2),
                        bcolor, min(int(scale), 5), lineType=cv2.LINE_AA)
        if class_name == 'hat':
            hat_num += 1
            rect = (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
            hat.add(createTrackerByName(trackerType), img, rect)
        elif class_name == 'person':
            person_num += 1
            rect = (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
            person.add(createTrackerByName(trackerType), img, rect)
    img = np.asarray(img)
    return img, hat, person, colors, hat_num, person_num


def detect_video2(video_path, output_path, thread):
    global video_time_list
    global video_all_list
    global video_no_list
    global video_yes_list
    import cv2
    # Create a video capture object to read videos
    cap = cv2.VideoCapture(video_path)

    # Read first frame
    success, frame = cap.read()
    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
        sys.exit(1)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    img = nd.array(frame, dtype='uint8')
    x, img = transform_test(img, short=416)
    x = x.as_in_context(ctx)
    box_ids, scores, bboxes = net(x)
    video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (img.shape[1], img.shape[0])
    out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    frame, hat, person, colors, hat_num, person_num = sel_track(img, bboxes[0], scores[0], box_ids[0],
                                                                trackerType="MEDIANFLOW")
    curr_time = timer()
    exec_time = curr_time - prev_time
    prev_time = curr_time
    accum_time = accum_time + exec_time
    curr_fps = curr_fps + 1
    if accum_time > 1:
        accum_time = accum_time - 1
        fps = "FPS: " + str(curr_fps)
        curr_fps = 0
    cv2.putText(frame,
                text=fps,
                org=(3, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.50,
                color=(255, 0, 0),
                thickness=2)
    cv2.imshow('MultiTracker', frame)
    ## 输出人数，戴安全帽的人数，未佩戴安全帽的人数
    print("person:{}, hat: {}, person without hat: {}".format(hat_num + person_num, hat_num, person_num))


    all = hat_num + person_num
    yes = hat_num
    no = person_num

    num_str = "person:{}, hat: {}, person without hat: {}".format(hat_num + person_num, hat_num, person_num)


    thread._signal.emit(num_str, frame, all, yes, no)

    ## 系统时间
    theTime = datetime.datetime.now()
    ## 输出系统时间
    print("system time: ", theTime)

    # cursor = conn.cursor()
    # insert_data_sql = 'insert into `%s`(detect_time,allp,yesp,nop)' % thread.table_name + ' values (%s,%s,%s,%s)'
    # cursor.execute(insert_data_sql, [str(theTime)[0:str(theTime).rfind(".")], all, yes, no])
    # conn.commit()
    # st = SqlThread(insert_data_sql, theTime, all, yes, no)
    # st.start()
    video_time_list.append(str(theTime)[0:str(theTime).rfind(".")])
    video_all_list.append(all)
    video_yes_list.append(yes)
    video_no_list.append(no)


    milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)

    seconds = milliseconds // 1000
    milliseconds = milliseconds % 1000
    minutes = 0
    hours = 0
    if seconds >= 60:
        minutes = seconds // 60
        seconds = seconds % 60

    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
    ## 小时 分钟 秒 毫秒
    print("{}:{}:{}:{}".format(int(hours), int(minutes), int(seconds), int(milliseconds)))
    # quit on ESC button
    out.write(frame)
    skip = 5
    cnt = 0
    while start_video and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        cnt += 1
        if cnt % skip == 0:
            img = nd.array(frame, dtype='uint8')
            x, img = transform_test(img, short=416)
            x = x.as_in_context(ctx)
            box_ids, scores, bboxes = net(x)
            frame, hat, person, colors, hat_num, person_num = sel_track(img, bboxes[0], scores[0], box_ids[0],
                                                                        trackerType="MEDIANFLOW")
        else:
            hat_num, person_num = 0, 0
            # get updated location of objects in subsequent frames
            img = Image.fromarray(frame)
            frame = img.resize(video_size)
            frame = np.array(frame)
            success, boxes = hat.update(frame)

            # draw tracked objects
            bcolor = [x * 255 for x in colors[0]]
            for i, newbox in enumerate(boxes):
                hat_num += 1
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, bcolor, 2)
                y = p1[1] - 15 if p1[1] - 15 > 15 else p1[1] + 15
                cv2.putText(frame, 'hat',
                            (p1[0], y), cv2.FONT_HERSHEY_SIMPLEX, 1 / 2,
                            bcolor, 1, lineType=cv2.LINE_AA)

            success, boxes = person.update(frame)

            # draw tracked objects
            bcolor = [x * 255 for x in colors[1]]
            for i, newbox in enumerate(boxes):
                person_num += 1
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, bcolor, 2)
                y = p1[1] - 15 if p1[1] - 15 > 15 else p1[1] + 15
                cv2.putText(frame, 'person',
                            (p1[0], y), cv2.FONT_HERSHEY_SIMPLEX, 1 / 2,
                            bcolor, 1, lineType=cv2.LINE_AA)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(frame,
                    text=fps,
                    org=(3, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50,
                    color=(255, 0, 0),
                    thickness=2)
        # show frame
        cv2.imshow('MultiTracker', frame)
        ## 输出人数，戴安全帽的人数，未佩戴安全帽的人数
        print("person:{}, hat: {}, person without hat: {}".format(hat_num + person_num, hat_num, person_num))


        all = hat_num + person_num
        yes = hat_num
        no = person_num

        num_str = "person:{}, hat: {}, person without hat: {}".format(hat_num + person_num, hat_num, person_num)
        if all > 0 and no > 0:
            t = AlertThread('报警')
            t.start()

        thread._signal.emit(num_str, frame, all, yes, no)


        ## 系统时间
        theTime = datetime.datetime.now()
        ## 输出系统时间
        print("system time: ", theTime)

        # cursor = conn.cursor()
        # insert_data_sql = 'insert into `%s`(detect_time,allp,yesp,nop)' % thread.table_name + ' values (%s,%s,%s,%s)'
        # cursor.execute(insert_data_sql, [str(theTime)[0:str(theTime).rfind(".")], all, yes, no])
        # conn.commit()

        # st = SqlThread(insert_data_sql, theTime, all, yes, no)
        # st.start()
        video_time_list.append(str(theTime)[0:str(theTime).rfind(".")])
        video_all_list.append(all)
        video_yes_list.append(yes)
        video_no_list.append(no)


        milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)

        seconds = milliseconds // 1000
        milliseconds = milliseconds % 1000
        minutes = 0
        hours = 0
        if seconds >= 60:
            minutes = seconds // 60
            seconds = seconds % 60

        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60
        ## 小时 分钟 秒 毫秒
        print("{}:{}:{}:{}".format(int(hours), int(minutes), int(seconds), int(milliseconds)))
        out.write(frame)
        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break


def detect_camera2(camera_id, output_path, thread):
    global camera_time_list
    global camera_all_list
    global camera_no_list
    global camera_yes_list
    import cv2
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    success, frame = cap.read()
    if not success:
        print('Failed to read video')
        sys.exit(1)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    img = nd.array(frame, dtype='uint8')
    x, img = transform_test(img, short=416)
    x = x.as_in_context(ctx)
    box_ids, scores, bboxes = net(x)
    video_size = (img.shape[1], img.shape[0])

    frame, hat, person, colors, hat_num, person_num = sel_track(img, bboxes[0], scores[0], box_ids[0],
                                                                trackerType="MEDIANFLOW")
    curr_time = timer()
    exec_time = curr_time - prev_time
    prev_time = curr_time
    accum_time = accum_time + exec_time
    curr_fps = curr_fps + 1
    if accum_time > 1:
        accum_time = accum_time - 1
        fps = "FPS: " + str(curr_fps)
        curr_fps = 0
    cv2.putText(frame,
                text=fps,
                org=(3, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.50,
                color=(255, 0, 0),
                thickness=2)
    cv2.imshow('MultiTracker', frame)
    ## 输出人数，戴安全帽的人数，未佩戴安全帽的人数
    print("person:{}, hat: {}, person without hat: {}".format(hat_num + person_num, hat_num, person_num))

    all = hat_num + person_num
    yes = hat_num
    no = person_num

    num_str = "person:{}, hat: {}, person without hat: {}".format(hat_num + person_num, hat_num, person_num)


    thread._signal.emit(num_str, frame, all, yes, no)


    ## 系统时间
    theTime = datetime.datetime.now()
    ## 输出系统时间
    print("system time: ", theTime)

    # cursor = conn.cursor()
    # insert_data_sql = 'insert into `%s`(detect_time,allp,yesp,nop)' % thread.table_name + ' values (%s,%s,%s,%s)'
    # cursor.execute(insert_data_sql, [str(theTime)[0:str(theTime).rfind(".")], all, yes, no])
    # conn.commit()
    camera_time_list.append(str(theTime)[0:str(theTime).rfind(".")])
    camera_all_list.append(all)
    camera_yes_list.append(yes)
    camera_no_list.append(no)

    milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)

    seconds = milliseconds // 1000
    milliseconds = milliseconds % 1000
    minutes = 0
    hours = 0
    if seconds >= 60:
        minutes = seconds // 60
        seconds = seconds % 60

    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
    ## 小时 分钟 秒 毫秒
    print("{}:{}:{}:{}".format(int(hours), int(minutes), int(seconds), int(milliseconds)))
    # quit on ESC button
    skip = 5
    cnt = 0
    while start_camera and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        cnt += 1
        if cnt % skip == 0:
            img = nd.array(frame, dtype='uint8')
            x, img = transform_test(img, short=416)
            x = x.as_in_context(ctx)
            box_ids, scores, bboxes = net(x)
            frame, hat, person, colors, hat_num, person_num = sel_track(img, bboxes[0], scores[0], box_ids[0],
                                                                        trackerType="MEDIANFLOW")
        else:
            hat_num, person_num = 0, 0
            # get updated location of objects in subsequent frames
            img = Image.fromarray(frame)
            frame = img.resize(video_size)
            frame = np.array(frame)
            success, boxes = hat.update(frame)

            # draw tracked objects
            bcolor = [x * 255 for x in colors[0]]
            for i, newbox in enumerate(boxes):
                hat_num += 1
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, bcolor, 2)
                y = p1[1] - 15 if p1[1] - 15 > 15 else p1[1] + 15
                cv2.putText(frame, 'hat',
                            (p1[0], y), cv2.FONT_HERSHEY_SIMPLEX, 1 / 2,
                            bcolor, 1, lineType=cv2.LINE_AA)

            success, boxes = person.update(frame)

            # draw tracked objects
            bcolor = [x * 255 for x in colors[1]]
            for i, newbox in enumerate(boxes):
                person_num += 1
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, bcolor, 2)
                y = p1[1] - 15 if p1[1] - 15 > 15 else p1[1] + 15
                cv2.putText(frame, 'person',
                            (p1[0], y), cv2.FONT_HERSHEY_SIMPLEX, 1 / 2,
                            bcolor, 1, lineType=cv2.LINE_AA)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(frame,
                    text=fps,
                    org=(3, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50,
                    color=(255, 0, 0),
                    thickness=2)
        # show frame
        cv2.imshow('MultiTracker', frame)
        ## 输出人数，戴安全帽的人数，未佩戴安全帽的人数
        print("person:{}, hat: {}, person without hat: {}".format(hat_num + person_num, hat_num, person_num))


        all = hat_num + person_num
        yes = hat_num
        no = person_num

        num_str = "person:{}, hat: {}, person without hat: {}".format(hat_num + person_num, hat_num, person_num)
        if all > 0 and no > 0:
            t = AlertThread('报警')
            t.start()

        thread._signal.emit(num_str, frame, all, yes, no)


        ## 系统时间
        theTime = datetime.datetime.now()
        ## 输出系统时间
        print("system time: ", theTime)

        # cursor = conn.cursor()
        # insert_data_sql = 'insert into `%s`(detect_time,allp,yesp,nop)' % thread.table_name + ' values (%s,%s,%s,%s)'
        # cursor.execute(insert_data_sql, [str(theTime)[0:str(theTime).rfind(".")], all, yes, no])
        # conn.commit()
        camera_time_list.append(str(theTime)[0:str(theTime).rfind(".")])
        camera_all_list.append(all)
        camera_yes_list.append(yes)
        camera_no_list.append(no)


        milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)

        seconds = milliseconds // 1000
        milliseconds = milliseconds % 1000
        minutes = 0
        hours = 0
        if seconds >= 60:
            minutes = seconds // 60
            seconds = seconds % 60

        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60
        ## 小时 分钟 秒 毫秒
        print("{}:{}:{}:{}".format(int(hours), int(minutes), int(seconds), int(milliseconds)))
        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break


class AlertThread(Thread):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def run(self):  # 固定名字run ！！！必须用固定名
        playsound("alert.mp3")


result_signal = pyqtSignal(str)
# 继承QThread
class Runthread(QtCore.QThread):
    # python3,pyqt5与之前的版本有些不一样
    #  通过类成员对象定义信号对象
    # result_signal = pyqtSignal(str)
    _signal = pyqtSignal(str, object, int, int, int)
    def __init__(self, app, parent=None):
        super(Runthread, self).__init__()
        self.app = app

    def __del__(self):
        self.wait()

    def run(self):
        cursor = conn.cursor()
        # 获取当地的时间，并转换成固定形式
        now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        table_name = self.app.input_file_name + "_" + now  # 加上``有这个'""'
        insert_video_sql = "INSERT INTO video(video_name) VALUES (%s);"
        cursor.execute(insert_video_sql, [table_name])
        conn.commit()
        #  创建数据表的sql 语句  并设置name_id 为主键自增长不为空

        sql_createTb = """CREATE TABLE  `%s`(
                         video_id INT NOT NULL AUTO_INCREMENT,
                         detect_time VARCHAR(100),
                         allp INT,yesp INT,nop INT,
                         PRIMARY KEY(video_id))
                         """ % (table_name)
        # % pymysql.escape_string(table_name)有\\






        # % pymysql.escape_string(table_name)有\\
        cursor.execute(sql_createTb)

        self.table_name = table_name
        global video_table_name
        video_table_name = table_name
        detect_video2(self.app.input_video, self.app.output_video,self)

    def callback(self, msg):
        pass

class CameraThread(QtCore.QThread):
    # python3,pyqt5与之前的版本有些不一样
    #  通过类成员对象定义信号对象
    # result_signal = pyqtSignal(str)
    _signal = pyqtSignal(str, object, int, int, int)
    def __init__(self, app, parent=None):
        super(CameraThread, self).__init__()
        self.app = app

    def __del__(self):
        self.wait()

    def run(self):
        cursor = conn.cursor()
        # 获取当地的时间，并转换成固定形式
        now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        table_name =  "camera_" + now  # 加上``有这个'""'
        insert_video_sql = "INSERT INTO camera(camera_name) VALUES (%s);"
        cursor.execute(insert_video_sql, [table_name])
        conn.commit()
        #  创建数据表的sql 语句  并设置name_id 为主键自增长不为空

        sql_createTb = """CREATE TABLE  `%s`(
                         camera_id INT NOT NULL AUTO_INCREMENT,
                         detect_time VARCHAR(100),
                         allp INT,yesp INT,nop INT,
                         PRIMARY KEY(camera_id))
                         """ % (table_name)
        # % pymysql.escape_string(table_name)有\\


        # % pymysql.escape_string(table_name)有\\
        cursor.execute(sql_createTb)

        self.table_name = table_name
        global camera_table_name
        camera_table_name = table_name
        detect_camera2(0, self.app.out_camera_path, self)

    def callback(self, msg):
        pass


class SqlThread(QtCore.QThread):
    # python3,pyqt5与之前的版本有些不一样
    #  通过类成员对象定义信号对象
    # result_signal = pyqtSignal(str)
    _signal = pyqtSignal(str, object, int, int, int)
    def __init__(self, sql, theTime, all, yes, no, parent=None):
        super(SqlThread, self).__init__()
        self.sql = sql
        self.theTime = theTime
        self.all = all
        self.yes = yes
        self.no = no

    def __del__(self):
        self.wait()

    def run(self):
        cursor = conn.cursor()
        insert_data_sql = self.sql
        cursor.execute(insert_data_sql, [str(self.theTime)[0:str(self.theTime).rfind(".")], self.all, self.yes, self.no])
        conn.commit()


    def callback(self, msg):
        pass


all = 0
yes = 0
no = 0
start_video = True
is_appear = 0
start_camera = True
class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()

        '以下是对主布局各部分分别实例化并设置属性'
        # self.funcListWidget = FuncListWidget(self)   # 表示功能菜单
        # self.fileSystemTreeView = FileSystemTreeView(self)  # 表示文件目录






        '主页面布局'

        '检测功能区'
        self.widget = QtWidgets.QWidget(self)
        self.widget.setMinimumHeight(130)
        self.widget.setMinimumWidth(200)
        self.widget.setGeometry(QtCore.QRect(10, 10, 171, 271))
        self.widget.setObjectName("widget")
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setGeometry(QtCore.QRect(10, 10, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.widget)
        self.pushButton_2.setGeometry(QtCore.QRect(90, 10, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.widget)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 40, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.widget)
        self.pushButton_4.setGeometry(QtCore.QRect(10, 100, 75, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.widget)
        self.pushButton_5.setGeometry(QtCore.QRect(90, 100, 75, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.widget)
        self.pushButton_6.setGeometry(QtCore.QRect(10, 130, 75, 23))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.widget)
        self.pushButton_7.setGeometry(QtCore.QRect(90, 130, 75, 23))
        self.pushButton_7.setObjectName("pushButton_7")

        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setGeometry(QtCore.QRect(10, 70, 141, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_2.setGeometry(QtCore.QRect(10, 160, 141, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_3.setGeometry(QtCore.QRect(10, 190, 141, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")


        self.lineEdit.setText("摄像头结果保存路径")
        self.pushButton.setText("摄像头检测")
        self.pushButton.clicked.connect(self.detectc)
        self.pushButton_2.setText("停止检测")
        self.pushButton_3.setText("选择保存路径")
        self.pushButton_3.clicked.connect(self.doCameraOutput)
        self.pushButton_4.setText("选择视频文件")
        self.pushButton_4.clicked.connect(self.doInputVideo)
        self.pushButton_5.setText("选择保存路径")
        self.pushButton_5.clicked.connect(self.doOutputVideo)
        self.pushButton_6.setText("开始检测")
        self.pushButton_6.clicked.connect(self.detects)
        self.pushButton_7.setText("停止检测")
        self.pushButton_7.clicked.connect(self.close_video)
        self.pushButton_2.clicked.connect(self.close_camera)
        # self.pushButton_3.setText("普通检测")
        # self.pushButton_4.setText("特殊检测")


        self.dock_group = QDockWidget(self)
        self.dock_group.setWidget(self.widget)
        self.dock_group.setTitleBarWidget(QLabel('检测功能区'))
        self.dock_group.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_group)

        '指示灯'
        self.led = MyLed()
        self.dock_led = QDockWidget(self)
        self.dock_led.setWidget(self.led)
        self.dock_led.setTitleBarWidget(QLabel('指示灯'))
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_led)

        '分析结果'
        self.groupBox = QtWidgets.QWidget(self)
        self.groupBox.setMinimumHeight(130)
        self.groupBox.setMinimumWidth(200)
        self.groupBox.setGeometry(QtCore.QRect(160, 90, 181, 201))
        self.label_11 = QtWidgets.QLabel(self.groupBox)
        self.label_11.setGeometry(QtCore.QRect(10, 40, 54, 32))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.groupBox)
        self.label_12.setGeometry(QtCore.QRect(120, 40, 54, 32))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.groupBox)
        self.label_13.setGeometry(QtCore.QRect(10, 80, 54, 32))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.groupBox)
        self.label_14.setGeometry(QtCore.QRect(120, 80, 54, 32))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.groupBox)
        self.label_15.setGeometry(QtCore.QRect(10, 120, 54, 32))
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.groupBox)
        self.label_16.setGeometry(QtCore.QRect(120, 120, 54, 32))
        self.label_11.setStyleSheet(
            "color:rgb(10,10,10,255);font-size:12px;font-weight:bold;font-family:Roman times; border:0px solid #000000;")
        self.label_12.setStyleSheet(
            "color:rgb(10,10,10,255);font-size:12px;font-weight:bold;font-family:Roman times; border:0px solid #000000;")
        self.label_13.setStyleSheet(
            "color:rgb(10,10,10,255);font-size:12px;font-weight:bold;font-family:Roman times; border:0px solid #000000;")
        self.label_14.setStyleSheet(
            "color:rgb(10,10,10,255);font-size:12px;font-weight:bold;font-family:Roman times; border:0px solid #000000;")
        self.label_15.setStyleSheet(
            "color:#ff0000;font-size:12px;font-weight:bold;font-family:Roman times; border:0px solid #000000;")
        self.label_16.setStyleSheet(
            "color:#ff0000;font-size:12px;font-weight:bold;font-family:Roman times; border:0px solid #000000;")
        self.label_16.setObjectName("label_16")
        self.label_11.setText("总人数")
        self.label_12.setText("2")
        self.label_13.setText("戴帽人数")
        self.label_14.setText("1")
        self.label_15.setText("未戴人数")
        self.label_16.setText("1")
        self.dock_num = QDockWidget(self)
        self.dock_num.setWidget(self.groupBox)
        self.dock_num.setTitleBarWidget(QLabel('分析结果'))
        self.dock_num.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_num)


        '中心区域'
        self.center = QLabel()
        self.center.setText("占位")
        self.center.setGeometry(QtCore.QRect(0, 0, 900, 700))
        self.setCentralWidget(self.center)
        path = 'icons/j.jpg'
        pic = QtGui.QPixmap(path).scaled(self.center.width(), self.center.height())
        self.center.setPixmap(pic)

        self.graphicsView = GraphicsView(self)
        self.graphicsView.hide()







        '设置标题栏样式'
        self.setWindowTitle('工地安全帽检测演示客户端')
        self.setWindowIcon(QIcon('icons/main.png'))



        reply = QMessageBox.information(self,
                                    "工地安全帽检测演示客户端",
                                    "欢迎使用！")



        '查询结果'
        self.resultlist1 = ReListWidget(self)
        self.resultlist1.itemClicked.connect(self.select_video)
        self.dock_result1 = QDockWidget(self)
        self.dock_result1.setWidget(self.resultlist1)
        self.dock_result1.setTitleBarWidget(QLabel('查询结果'))
        self.dock_result1.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_result1)

        '查询功能区'
        self.widget2 = QtWidgets.QWidget(self)
        self.widget2.setMinimumHeight(250)
        self.widget2.setMinimumWidth(200)
        self.widget2.setGeometry(QtCore.QRect(10, 10, 171, 271))
        self.widget2.setObjectName("widget2")
        self.label_23 = QtWidgets.QLabel(self.widget2)
        self.label_23.setGeometry(QtCore.QRect(10, 70, 60, 23))
        self.label_23.setObjectName("label_23")
        self.label_23.setText("选定视频")
        self.label_23.setStyleSheet("color:rgb(10,10,10,255);font-size:12px;font-weight:bold;font-family:Roman times; border:0px solid #000000;")
        self.label_24 = QtWidgets.QLabel(self.widget2)
        self.label_24.setGeometry(QtCore.QRect(90, 70, 200, 23))
        self.label_24.setObjectName("label_23")
        self.label_24.setText("选定视频")
        self.label_24.setStyleSheet("color:rgb(10,10,10,255);font-size:12px;font-weight:bold;font-family:Roman times; border:0px solid #000000;")
        self.pushButton_11 = QtWidgets.QPushButton(self.widget2)
        self.pushButton_11.setGeometry(QtCore.QRect(10, 100, 75, 23))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_11.setText("查询视频列表")
        self.pushButton_11.clicked.connect(self.select_all_video)
        self.pushButton_12 = QtWidgets.QPushButton(self.widget2)
        self.pushButton_12.setGeometry(QtCore.QRect(90, 100, 75, 23))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_12.setText("查询视频结果")
        self.pushButton_12.clicked.connect(self.select_video_result)
        self.pushButton_13 = QtWidgets.QPushButton(self.widget2)
        self.pushButton_13.setGeometry(QtCore.QRect(10, 130, 75, 23))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_13.setText("选择导出路径")
        self.pushButton_13.clicked.connect(self.generate_csv_path)
        self.pushButton_14 = QtWidgets.QPushButton(self.widget2)
        self.pushButton_14.setGeometry(QtCore.QRect(90, 130, 75, 23))
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_14.setText("导出查询结果")
        self.pushButton_14.clicked.connect(self.generate_csv)
        self.lineEdit_11 = QtWidgets.QLineEdit(self.widget2)
        self.lineEdit_11.setGeometry(QtCore.QRect(10, 160, 141, 20))
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.lineEdit_11.setText("导出数据路径")
        self.pushButton_15 = QtWidgets.QPushButton(self.widget2)
        self.pushButton_15.setGeometry(QtCore.QRect(10, 190, 75, 23))
        self.pushButton_15.setObjectName("pushButton_15")
        self.pushButton_15.setText("查询摄像列表")
        self.pushButton_15.clicked.connect(self.select_all_camera)
        self.pushButton_16 = QtWidgets.QPushButton(self.widget2)
        self.pushButton_16.setGeometry(QtCore.QRect(90, 190, 75, 23))
        self.pushButton_16.setObjectName("pushButton_16")
        self.pushButton_16.setText("查询摄像结果")
        self.pushButton_16.clicked.connect(self.select_video_result)
        self.pushButton_17 = QtWidgets.QPushButton(self.widget2)
        self.pushButton_17.setGeometry(QtCore.QRect(10, 220, 75, 23))
        self.pushButton_17.setObjectName("pushButton_17")
        self.pushButton_17.setText("退出系统")
        self.pushButton_17.clicked.connect(self.quit)
        self.dock_group2 = QDockWidget(self)
        self.dock_group2.setWidget(self.widget2)
        self.dock_group2.setTitleBarWidget(QLabel('查询功能区'))
        self.dock_group2.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_group2)






    def doInputVideo(self):
        self.cwd = os.getcwd()
        self.input_video, filetype = QFileDialog.getOpenFileName()  # 设置文件扩展名过滤,用双分号间隔
        if self.input_video == "":
            print("\n取消选择")
            return
        print("\n你选择的文件为:")
        print(self.input_video)
        print("文件筛选器类型: ", filetype)
        self.lineEdit_2.setText(self.input_video)
        self.input_file_name = self.input_video[self.input_video.rfind("/")+1: len(self.input_video)]
        self.input_file_name = self.input_file_name.replace(".", "_")
        print(self.input_file_name)

    def doOutputVideo(self):
        dir_choose = QFileDialog.getExistingDirectory(self,
                                                      "选取文件夹",
                                                      self.cwd)  # 起始路径
        if dir_choose == "":
            print("\n取消选择")
            return
        print("\n你选择的文件夹为:")
        print(dir_choose)
        out_file = self.input_file_name.split("_")
        self.out_file_name = out_file[0] + "_result." + out_file[1]
        print(self.out_file_name)
        f = dir_choose + "/" + self.out_file_name
        print(f)
        with open(f, "a+") as fp:
            pass
        self.output_video = f
        self.lineEdit_3.setText(self.output_video)


    def detects(self):

        # 清空
        global video_time_list
        global video_all_list
        global video_no_list
        global video_yes_list
        global video_table_name

        video_time_list = []
        video_all_list = []
        video_no_list = []
        video_yes_list = []
        video_table_name = ""

        global start_video
        global start_camera

        start_video = True
        start_camera = True
        if not hasattr(self, "input_video") or not hasattr(self, "output_video"):
            reply = QMessageBox.information(self,
                                            "工地安全帽检测演示客户端",
                                            "请先选择结果保存路径！")
        else:
            self.center.hide()
            self.graphicsView = GraphicsView(self)
            self.setCentralWidget(self.graphicsView)
            self.graphicsView.show()
            # 创建线程
            self.thread = Runthread(self)
            # 连接信号
            self.thread._signal.connect(self.callbacklog)
            # 开始线程
            self.thread.start()

    def detectc(self):
        global camera_time_list
        global camera_all_list
        global camera_no_list
        global camera_yes_list
        global camera_table_name

        camera_time_list = []
        camera_all_list = []
        camera_no_list = []
        camera_yes_list = []
        camera_table_name = ""

        global start_camera
        global start_video
        start_camera = True
        start_video = True
        if not hasattr(self, "out_camera_path"):
            reply = QMessageBox.information(self,
                                            "工地安全帽检测演示客户端",
                                            "请先选择结果保存路径！")
        else:
            self.center.hide()
            self.graphicsView = GraphicsView(self)
            self.setCentralWidget(self.graphicsView)
            self.graphicsView.show()
            # 创建线程
            self.thread = CameraThread(self)
            # 连接信号
            self.thread._signal.connect(self.callbacklog)
            # 开始线程
            self.thread.start()




    def callbacklog(self, data, data2, all, yes, no):
        global is_appear
        global start_video
        global start_camera
        if start_video and start_camera:
            self.resultlist1.addItem(data)
            self.graphicsView.change_image(data2)
            self.label_12.setText(str(all))
            self.label_14.setText(str(yes))
            self.label_16.setText(str(no))
            if all > 0 and no > 0 and is_appear < 1:
                is_appear = is_appear + 1
                self.led.click()
            if all > 0 and all == yes and no < 1 and is_appear > 0:
                self.led.click()
                is_appear = 0
        pass

    def close_video(self):
        global start_video
        start_video = False
        self.center = QLabel()
        self.center.setText("占位")
        self.center.setGeometry(QtCore.QRect(0, 0, 900, 700))
        self.setCentralWidget(self.center)
        path = 'icons/j.jpg'
        pic = QtGui.QPixmap(path).scaled(self.center.width(), self.center.height())
        self.center.setPixmap(pic)

        global video_time_list
        global video_all_list
        global video_no_list
        global video_yes_list
        global video_table_name
        cursor = conn.cursor()
        for theTime, all, yes, no in zip(video_time_list, video_all_list, video_yes_list, video_no_list):
            insert_data_sql = 'insert into `%s`(detect_time,allp,yesp,nop)' % video_table_name + ' values (%s,%s,%s,%s)'
            cursor.execute(insert_data_sql, [theTime, all, yes, no])
            conn.commit()

    def select_all_video(self):
        cursor = conn.cursor()
        select_all_sql = "select video_name from video"
        cursor.execute(select_all_sql)
        result = cursor.fetchall()
        self.resultlist1.clear()
        for r in result:
            self.resultlist1.addItem(r[0])

    def select_video(self):
        self.select_video_name = self.resultlist1.currentItem().text()
        self.label_24.setText(self.select_video_name)

    def select_video_result(self):
        cursor = conn.cursor()
        select_video_sql = "select detect_time,allp,yesp,nop from `%s`"%self.select_video_name
        cursor.execute(select_video_sql)
        result = cursor.fetchall()
        self.resultlist1.clear()
        self.video_result = result
        for r in result:
            item_text = r[0] + "         " + str(r[1]) + "/" + str(r[2]) + "/" + str(r[3])
            self.resultlist1.addItem(item_text)

    def generate_csv_path(self):
        self.cwd = os.getcwd()
        dir_choose = QFileDialog.getExistingDirectory(self,
                                                      "选取生成csv文件夹",
                                                      self.cwd)  # 起始路径
        if dir_choose == "":
            print("\n取消选择")
            return
        print("\n你选择的文件夹为:")
        print(dir_choose)
        self.out_csv_name = "csv_result.csv"
        print(self.out_csv_name)
        f = dir_choose + "/" + self.out_csv_name
        print(f)
        with open(f, "a+") as fp:
            pass
        self.lineEdit_11.setText(f)

    def generate_csv(self):
        name_list = ["detect_time","all","yes","no"]
        result = pd.DataFrame(columns=name_list,data=self.video_result)
        result.to_csv(self.lineEdit_11.text())



    def doCameraOutput(self):
        self.cwd = os.getcwd()
        dir_choose = QFileDialog.getExistingDirectory(self,
                                                      "选取摄像头结果保存文件夹",
                                                      self.cwd)  # 起始路径
        if dir_choose == "":
            print("\n取消选择")
            return
        print("\n你选择的文件夹为:")
        print(dir_choose)
        self.out_camera_path = "camera_result.mp4"
        print(self.out_camera_path)
        f = dir_choose + "/" + self.out_camera_path
        print(f)
        with open(f, "a+") as fp:
            pass
        self.lineEdit.setText(f)

    def close_camera(self):
        global start_camera
        start_camera = False
        self.center = QLabel()
        self.center.setText("占位")
        self.center.setGeometry(QtCore.QRect(0, 0, 900, 700))
        self.setCentralWidget(self.center)
        path = 'icons/j.jpg'
        pic = QtGui.QPixmap(path).scaled(self.center.width(), self.center.height())
        self.center.setPixmap(pic)

        global camera_time_list
        global camera_all_list
        global camera_no_list
        global camera_yes_list
        global camera_table_name
        cursor = conn.cursor()
        for theTime, all, yes, no in zip(camera_time_list, camera_all_list, camera_yes_list, camera_no_list):
            insert_data_sql = 'insert into `%s`(detect_time,allp,yesp,nop)' % camera_table_name + ' values (%s,%s,%s,%s)'
            cursor.execute(insert_data_sql, [theTime, all, yes, no])
            conn.commit()

    def select_all_camera(self):
        cursor = conn.cursor()
        select_all_sql = "select camera_name from camera"
        cursor.execute(select_all_sql)
        result = cursor.fetchall()
        self.resultlist1.clear()
        for r in result:
            self.resultlist1.addItem(r[0])


    def quit(self):
        i = 1/0


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(open('./custom/styleSheet.qss', encoding='utf-8').read())
    window = MyApp()
    window.showMaximized()
    sys.exit(app.exec_())
