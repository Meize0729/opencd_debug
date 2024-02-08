import cv2
import cv2
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import tifffile


def slide_crop(filename):
    real_path = os.path.join(folder_path, filename)
    
    name = filename.split('.')[0]

    img = cv2.imread(real_path)

    # 添加
    top = left = 0  # 不在顶部和左边填充
    bottom = right = 36  # 在底部和右边填充18个像素
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    crop_width, crop_height = 512, 512
    step_x, step_y = 512, 512
    height, width, _ = img.shape
    crops = []
    for i in range(0, width - crop_width + 1, step_x):
        for j in range(0, height - crop_height + 1, step_y):
            crop = img[j:j + crop_height, i:i + crop_width]
            crops.append(crop)
    for i, crop in enumerate(crops):
        cv2.imwrite(f'{out_path}/{name}_crop_{i}.png', crop)

if __name__ == '__main__':
    # 文件夹路径
    folder_path = '/mnt/public/usr/wangmingze/Datasets/CD/mass_build/png/test_labels'
    real_folder_path = folder_path.split('/')[-1]
    out_path = folder_path.replace(real_folder_path, 'test_pad_512_labels')
    os.system(f'rm -rf {out_path}')
    os.system(f'mkdir {out_path}')

    # 获取文件夹下所有的文件路径
    filenames = [filename for filename in os.listdir(folder_path) if
                 filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.tif')]
    
    # 使用多进程并行处理图像
    with Pool(10) as p:
        list(tqdm(p.imap(slide_crop, filenames), total=len(filenames)))
    
