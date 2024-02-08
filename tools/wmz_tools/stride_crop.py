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

    # img = cv2.resize(img, (1536, 1536), interpolation=cv2.INTER_NEAREST)
    # img[img < 128] = 0
    # img[img >= 128] = 255

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
    folder_path = '/mnt/public/usr/wangmingze/Datasets/CD/xView2_test/AB'
    real_folder_path = folder_path.split('/')[-1]
    out_path = folder_path.replace(real_folder_path, 'AB_512_nooverlap')
    os.system(f'rm -rf {out_path}')
    os.system(f'mkdir {out_path}')

    # 获取文件夹下所有的文件路径
    filenames = [filename for filename in os.listdir(folder_path) if
                 filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.tif')]
    
    # 使用多进程并行处理图像
    with Pool(80) as p:
        list(tqdm(p.imap(slide_crop, filenames), total=len(filenames)))
    
