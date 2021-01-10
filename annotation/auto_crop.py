# USAGE
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

import numpy as np
import argparse
import os
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
                help="path to input image")
ap.add_argument("-p", "--prototxt", default='deploy.prototxt.txt',
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default='res10_300x300_ssd_iter_140000.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# モデルを読み込む
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

data_dir = args["dir"].replace("\\", "/")
annotated_dir = data_dir + 'trimed'
os.mkdir(annotated_dir)
files = os.listdir(data_dir)
img_files = [
    f for f in files if '.jpeg' in f or '.jpg' in f or '.png' in f]
ng_imgs = []
for img_file in img_files:
    image = cv2.imread(img_file)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # 指定した自信度より高い画像を保存する
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print(startX, startY, endX, endY)
            out_img = image[startY:endY, startX:endX]

            # show the output image
            file_dir = annotated_dir + '/' + \
                img_file.replace('.', '_{}.'.format(i))
            try:
                cv2.imwrite(file_dir, out_img)
            except cv2.error:
                ng_imgs.append(file_dir)
