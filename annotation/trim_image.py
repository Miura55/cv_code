import cv2
import os
import sys

if __name__ == '__main__':
    args = sys.argv
    data_dir = args[1].replace('\\', '/')
    annotated_dir = data_dir + 'trimed'
    os.mkdir(annotated_dir)
    files = os.listdir(data_dir)
    img_files = [
        f for f in files if '.jpeg' in f or '.jpg' in f or '.png' in f]
    len_data = len(img_files)
    count = 0
    for img_file in img_files:
        # Read image
        img_dir = data_dir + img_file
        img = cv2.imread(img_dir)
        # Select ROI
        selected = cv2.selectROI(img)
        count += 1
        if sum(selected):
            # Crop image
            imCrop = img[int(selected[1]):int(selected[1]+selected[3]),
                         int(selected[0]):int(selected[0]+selected[2])]
            # write annotated img
            file_dir = './' + annotated_dir + '/' + img_file
            cv2.imwrite(file_dir, imCrop)
            print('save {}'.format(file_dir))
            print('{}/{} saved!'.format(count, len_data))
