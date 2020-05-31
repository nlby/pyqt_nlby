import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import cv2
import os
import generate_the_image as gti 
from tqdm import tqdm
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk

label_dict = {}
label_dict['nodule'] = 1
label_dict['stripe'] = 5
label_dict['artery'] = 31
label_dict['lymph'] = 32
def get_file_name(file_path):
    files = []
    for f_name in [f for f in os.listdir(file_path) if f.endswith('.mhd')]:
        files.append(f_name)
    return sorted(files)

def load_itk(file):
    itkimage = sitk.ReadImage(file)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

def detect_img(yolo,dirname,app):#批量处理图片
      ##该目录为测试照片的存储路径，每次测试照片的数量可以自己设定
    path = os.path.join(dirname)
    pic_list = os.listdir(path)
    count = 0
    yolo = YOLO()
    for filename in pic_list:
        tmp_file = pic_list[count]
        abs_path = path + pic_list[count]
        image = Image.open(abs_path)
        r_image,detect_result,nresult = yolo.detect_image(image)
        #r_image.show()		
        r_image.save("result1/"+tmp_file)
        # app.paramList.append(nresult[0])
        paramsList.append(nresult[0])
        print(nresult[0] + "--------hhh" + dirname + "result1/" + tmp_file)
        count = count + 1
    print(count)    


def detect_image_raw(file_path,save_path,app):
    gti.generate_images(file_path,save_path)
    yolo = YOLO()
    detect_img(yolo,save_path,app)
    save_name = 'result1/1.csv'
    seriesuid, coordX, coordY, coordZ, class_label, probability = [], [], [], [], [], []
    clipmin=-1000; clipmax=600
    files = get_file_name(file_path)
    for f_name in tqdm(files):
        result_id = int(f_name.replace('.mhd', ''))
        current_file = os.path.join(file_path, f_name)
        ct, origin, spacing = load_itk(current_file)
        ct = ct.clip(min=clipmin, max=clipmax)
        for num in range(ct.shape[0]):
            image = Image.fromarray(ct[num])
            detect_result = yolo.ddetect_image(image)
            
            for one_result in detect_result:
                result_probability = one_result[1]
                result_label = int(label_dict[one_result[0]])
                result_x = (one_result[2] + one_result[4]) / 2
                result_x = result_x * spacing[2] + origin[2]
                result_y = (one_result[3] + one_result[5]) / 2
                result_y = result_y * spacing[1] + origin[1]
                result_z = num
                result_z = result_z * spacing[0] + origin[0]
                # print(result_id, result_x, result_y, result_z, result_label, result_probability)
                seriesuid.append(result_id)
                coordX.append(result_x)
                coordY.append(result_y)
                coordZ.append(result_z)
                class_label.append(result_label)
                probability.append(result_probability)
    dataframe = pd.DataFrame({'seriesuid': seriesuid, 'coordX': coordX, 'coordY': coordY, 'coordZ': coordZ, 'class': class_label, 'probability': probability})
    columns = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability']
    dataframe.to_csv(save_name, index=False, sep=',', columns=columns)
    yolo.close_session()
FLAGS = None

if __name__ == '__main__':
    detect_image_raw('testB/','testB_png/')
    # detect_img(YOLO(), "C:/for_gui/")
    # detect_img(YOLO(), "D:\\Source\\qq\\1029253541\\FileRecv\\for_gui\\for_gui\\")
    # # class YOLO defines the default value, so suppress any default here
    # parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    # '''
    # Command line options
    # '''
    # parser.add_argument(
    #     '--model', type=str,
    #     help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    # )
    #
    # parser.add_argument(
    #     '--anchors', type=str,
    #     help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    # )
    #
    # parser.add_argument(
    #     '--classes', type=str,
    #     help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    # )
    #
    # parser.add_argument(
    #     '--gpu_num', type=int,
    #     help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    # )
    # '''
    # Command line positional arguments -- for image detection mode
    # '''
    # parser.add_argument(
    #     '--image', default=True, action="store_true",
    #     help='Image detection mode, will ignore all positional arguments'
    # )
    # parser.add_argument(
    #     '--image_r', default=True, action="store_true",
    #     help='Image detection mode, will ignore all positional arguments'
    # )
    # '''
    # Command line positional arguments -- for video detection mode
    # '''
    # parser.add_argument(
    #     "--video_input", nargs='?', type=str, default=False,
    #     help = "Video input path"
    # )
    #
    # parser.add_argument(
    #     "--video_output", nargs='?', type=str, default="",
    #     help = "[Optional] Video output path"
    # )
    #
    #
    # FLAGS = parser.parse_args()
	#
    # if FLAGS.image_r:
    #    detect_image_raw('../testB','testB_png/')
    # if FLAGS.image:
    #     """
    #     Image detection mode, disregard any remaining command line arguments
    #     """
    #     print("Image detection mode")
    #     if "input" in FLAGS:
    #         print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
    #     detect_img(YOLO(**vars(FLAGS)))
    # if FLAGS.video_input:
    #     detect_video(YOLO(**vars(FLAGS)), FLAGS.video_input, FLAGS.video_output)
    # else:
    #     print("Must specify at least video_input_path.  See usage with --help.")
