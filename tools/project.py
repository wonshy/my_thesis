import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import json
import argparse
import glob


tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


#config file
REF_JSON_PATH_BASE='/root/autodl-tmp/openlane'
DIFF_FILE_PATH='/root/autodl-tmp/diff_file'


#head tag
FILE_HEAD_TAG = 'segment-'
#tail tag
FILE_TAIL_TAG = '_with_camera_labels'
#class
IMAGE_PATH='/root/autodl-tmp/extend/images'
LABEL_PATH='/root/autodl-tmp/extend/labels'



#test data filter
DATA_PATH='/root/autodl-tmp/openlane/lane3d_1000/test'
VALIDATION_PATH='/root/autodl-tmp/openlane/lane3d_1000/validation'
TEST_CASES=["curve_case" ,"extreme_weather_case" ,"intersection_case","merge_split_case","night_case","up_down_case"]





def file_find(start, name):
    for relpath, dirs, files in os.walk(start):
        if name in files:
            full_path = os.path.join(start, relpath, name)
            return os.path.normpath(os.path.abspath(full_path))

def dir_find(start, name):
    for relpath, dirs, files in os.walk(start):
        if name in dirs:
            full_path = os.path.join(start, relpath, name)
            return os.path.normpath(os.path.abspath(full_path))

# open data.json
def json_intrinsic_extrinsic_extract(file):
    intrinsic = np.zeros((3,3))
    extrinsic = np.zeros((4,4))

    with open(file, 'r') as f:
        data = json.load(f)
        intrinsic = np.array(data["intrinsic"])
        extrinsic = np.array(data["extrinsic"])
    return intrinsic,extrinsic

def string_file_add(file, context):
    with open(file,'a') as f:
        f.write(context)


def create_folder(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)

def image_json_extract(tfrecord_file):
    create_folder(IMAGE_PATH)
    create_folder(LABEL_PATH)
    dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')


    #extruct data
    for data in dataset:
        #TODO:print context name and timestamp 
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        #print(frame.context.name, frame.timestamp_micros)

        class_path='unknown'

        # Fetch camera calibration.
        for index, calibration in enumerate(frame.context.camera_calibrations):

            # print(calibration.name)

            #create image path
            frame_json_directory = LABEL_PATH + '/' + class_path +'/'+ FILE_HEAD_TAG + frame.context.name + FILE_TAIL_TAG
            json_name = frame_json_directory + '/' + str(frame.timestamp_micros) + '00.' + str(calibration.name - 1) + '.json'

            extrinsic=np.array(calibration.extrinsic.transform).reshape(4,4)
            intrinsic_orig = calibration.intrinsic

            intrinsic=np.zeros((3,3))
            intrinsic[0,0]= intrinsic_orig[0]
            intrinsic[1,1]= intrinsic_orig[1]
            intrinsic[0,2]= intrinsic_orig[2]
            intrinsic[1,2]= intrinsic_orig[3]
            intrinsic[2,2]=1

            #compare file
            if calibration.name == 1:
                ref_json_path=dir_find(REF_JSON_PATH_BASE, FILE_HEAD_TAG + frame.context.name + FILE_TAIL_TAG)
                if ref_json_path is not None and os.path.exists(ref_json_path):
                    ref_json_file = file_find(ref_json_path, str(frame.timestamp_micros) + '0' + str(calibration.name - 1) + '.json')
                    if  ref_json_file is not None  and os.path.exists(ref_json_file):
                        ref_intrinsic, ref_extrinsic= json_intrinsic_extrinsic_extract(ref_json_file)
                        if ((ref_intrinsic == intrinsic).all and (ref_extrinsic == extrinsic).all):
                            #assign class
                            class_path = ref_json_file.split('/')[-3]
                        else:
                            string_file_add(DIFF_FILE_PATH ,json_name+'\n')
                            class_path='unknown'
                    else:
                        string_file_add(DIFF_FILE_PATH ,json_name+'\n')
                        class_path='unknown'
                else:
                    string_file_add(DIFF_FILE_PATH ,json_name+'\n')
                    class_path='unknown'
            else:
                # save name :2,3,4,5
                create_folder(frame_json_directory)
                with open(json_name, 'w') as f:
                    f.write( json.dumps({"intrinsic":intrinsic.tolist(), "extrinsic": extrinsic.tolist()}) )


        #TODO: frame.name和calibration.name 都是1，2，3，4，5
        #Fetch camera frames.
        for index, img in enumerate(frame.images):

            #create image path
            frame_image_directory = IMAGE_PATH + '/' + class_path +'/'+ FILE_HEAD_TAG + frame.context.name + FILE_TAIL_TAG
            image_name = frame_image_directory + '/' + str(frame.timestamp_micros) + '00.' + str(img.name - 1)  + '.jpg'
            #save name: 2,3,4,5 
            if img.name != 1:
                create_folder(frame_image_directory)
                with open(image_name, 'wb') as f:
                    f.write(img.image)


def test_data_filter():
    #1.iterate case   
    for case in TEST_CASES:
        case_path = os.path.join(DATA_PATH, case) # DATA_PATH+'/'+ case

        #2. iterate segment
        for segment in os.listdir(case_path):
            validation_segment=dir_find(VALIDATION_PATH, segment)
            if validation_segment  is not None:
                #3. iterate file
                for file in os.listdir(os.path.join(case_path, segment)):
                    validation_file = file_find(validation_segment, file)
                    if validation_file  is not None:
                        pass
                    else:
                        print(os.path.join(case_path, segment, file))
                        os.remove(os.path.join(case_path, segment, file))
            else:
                print(segment)
                os.remove(os.path.join(case_path, segment))





def main():
    parser = argparse.ArgumentParser(description='paper work')
    parser.add_argument('--tfrecord_dir', type=str, help='The path of tfrecord ')
    parser.add_argument('--test_filter', action='store_true', help='test data filter')

    args = parser.parse_args() 

    if not args.test_filter:
        files_list=glob.glob(args.tfrecord_dir+"/*.tfrecord")
        for file in files_list:
            # print("==========")
            # print(file)
            image_json_extract(file)
        print("not ok ")
    else:
        print("filter")
        test_data_filter()


if __name__ == '__main__':
    main()

