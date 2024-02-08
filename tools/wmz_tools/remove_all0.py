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
    img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
    # import pdb; pdb.set_trace()
    img[img != 0] = 1
    # os.system(f'rm -rf {filepath}')
    # 保存调整后的图像
    cv2.imwrite(f'{out_path}/{name}.png', img)

if __name__ == '__main__':
    # 文件夹路径
    folder_path = '/mnt/public/usr/wangmingze/Datasets/CD/WHU-EAST_ASIA/train/label_png'
    real_folder_path = folder_path.split('/')[-1]
    out_path = folder_path.replace(real_folder_path, 'label_png_01')
    os.system(f'rm -rf {out_path}')
    os.system(f'mkdir {out_path}')

    # 获取文件夹下所有的文件路径
    filenames = [filename for filename in os.listdir(folder_path) if
                 filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.tif')]
    import pdb; pdb.set_trace()
    # 使用多进程并行处理图像
    # for filename in filenames:
        # slide_crop(filename)
    with Pool(80) as p:
        list(tqdm(p.imap(slide_crop, filenames), total=len(filenames)))
    
