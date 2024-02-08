from multiprocessing import Pool
import cv2
import os
import numpy as np
from tqdm import tqdm

def read_image(image_file):
    return cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

def array_in_list(arr, list_of_arrays):
    for other_arr in list_of_arrays:
        if np.array_equal(arr, other_arr):
            return True
    return False

def check_overlap(image):
    image_path = os.path.join(images_path, image)
    other_img_path = os.path.join(other_path, image)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(image_path, img)
    if np.any((img != 0) & (img != 255)):
        print(image_path)
        os.system(f'sudo rm -rf {image_path}')
    if array_in_list(img, test_data):
        os.system(f'sudo rm -rf {image_path}')
        os.system(f'sudo rm -rf {other_img_path}')
        return image
    return None

if __name__ == '__main__':
    folder_path_list = [
        '/mnt/public/usr/wangmingze/Datasets/CD/BANDON/test_bx.txt',
        '/mnt/public/usr/wangmingze/Datasets/CD/mass_build/test.txt',
        '/mnt/public/usr/wangmingze/Datasets/CD/WHU-main/test_tif.txt',
        '/mnt/public/usr/wangmingze/Datasets/CD/AerialImageDataset/val.txt',
    ]

    image_files = []
    for folder_path in folder_path_list:
        with open(folder_path, 'r') as f:
            for line in f.readlines():
                a, b, cd, label_a, label_b = line.strip().split('\t')
                image_files.append(label_a)

    with Pool(40) as p:
        test_data = list(tqdm(p.imap(read_image, image_files), total=len(image_files)))

    images_path = '/mnt/public/usr/wangmingze/Datasets/CD/rs_builds/gt'
    other_path = '/mnt/public/usr/wangmingze/Datasets/CD/rs_builds/img'
    images = os.listdir(images_path)
    length = len(images)
    images = sorted(images)[length//4*3+1 : length//4*4]

    with Pool(40) as p:
        overlap = list(tqdm(p.imap(check_overlap, images), total=len(images)))
    overlap = [x for x in overlap if x is not None]
    length = len(overlap)
    print(f'overlap: {length}')

    # img_path = '/mnt/public/usr/wangmingze/Datasets/CD/rs_builds/try'
    # gt_path = '/mnt/public/usr/wangmingze/Datasets/CD/rs_builds/try2'
    # for name in overlap:
    #     img = os.path.join(img_path, name)
    #     gt = os.path.join(gt_path, name)
    #     os.system(f'sudo rm -rf {img}')
    #     os.system(f'sudo rm -rf {gt}')

    import pdb; pdb.set_trace() 
        # # 使用多进程并行处理图像
        # with Pool(80) as p:
        #     list(tqdm(p.imap(process_image, image_files), total=len(image_files)))