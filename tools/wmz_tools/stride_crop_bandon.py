import cv2
import cv2
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm



def slide_crop(args):
    filename, out_path = args
    real_path = filename

    name = filename.split('/')[-1].split('.')[0]
    try:
        img = cv2.imread(real_path)
        img[img != 0] = 255
    except:
        print(real_path)
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
    folder_path = '/mnt/public/usr/wangmingze/Datasets/CD/BANDON/test'
    out_path_old = folder_path.replace(folder_path.split('/')[-1], folder_path.split('/')[-1]+'_nooverlap')
    os.system(f'mkdir {out_path_old}')
    # names = ['imgs']
    names = ['building_labels', 'labels_unch0ch1ig255']
    for name in names:
        path1 = os.path.join(folder_path, name) # '/mnt/public/usr/wangmingze/Datasets/CD/BANDON/train/imgs'
        out_path1 = os.path.join(out_path_old, name)
        os.system(f'mkdir {out_path1}')

        for place in os.listdir(path1):
            path2 = os.path.join(path1, place) # '/mnt/public/usr/wangmingze/Datasets/CD/BANDON/train/imgs/bj'
            out_path2 = os.path.join(out_path1, place)
            os.system(f'mkdir {out_path2}')

            for time in os.listdir(path2):
                path3 = os.path.join(path2, time)  # '/mnt/public/usr/wangmingze/Datasets/CD/BANDON/train/imgs/bj/t1'
                out_path = os.path.join(out_path2, time)
                os.system(f'mkdir {out_path}')
                print(path3)
                # 获取文件夹下所有的文件路径
                filenames = [os.path.join(path3, filename) for filename in os.listdir(path3) if
                            filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.tif')]
                
                with Pool(64) as p:
                    _args = zip(filenames, [out_path] * len(filenames))
                    list(tqdm(p.imap(slide_crop, _args), total=len(filenames)))


    
