import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import tifffile
from PIL import Image
import numpy as np

# def process_image(filepath):
#     # 读取图像
#     img = tifffile.imread(filepath)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     filepath = filepath.replace('.tif', '.png')
#     # 保存调整后的图像
#     cv2.imwrite(os.path.join('/mnt/public/usr/wangmingze/Datasets/CD/WHU-main/test/image_png', os.path.basename(filepath)), img)
# if __name__ == '__main__':
#     # 文件夹路径
#     folder_path = '/mnt/public/usr/wangmingze/Datasets/CD/WHU-main/test/image'
#     os.system('rm -rf /mnt/public/usr/wangmingze/Datasets/CD/WHU-main/test/image_png')
#     os.makedirs('/mnt/public/usr/wangmingze/Datasets/CD/WHU-main/test/image_png', exist_ok=True)

#     # 获取文件夹下所有的文件路径
#     filepaths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
#                  filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.tif')]

#     # 使用多进程并行处理图像
#     with Pool() as p:
#         list(tqdm(p.imap(process_image, filepaths), total=len(filepaths)))

filepath = '/mnt/public/usr/wangmingze/Datasets/CD/AOI_4_Shanghai_Train/RGB-PanSharpen/RGB-PanSharpen_AOI_4_Shanghai_img37.tif'
img = tifffile.imread(filepath)
# 找出图像数据的最大值和最小值
min_val = np.min(img)
max_val = np.max(img)

# 将图像数据归一化到0-1之间
img_normalized = (img - min_val) / (max_val - min_val)

# 将归一化后的图像数据转换为0-255范围内的uint8类型
img = (img_normalized * 255).astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('/mnt/public/usr/wangmingze/Datasets/CD/AOI_4_Shanghai_Train/RGB-PanSharpen/RGB-PanSharpen_AOI_4_Shanghai_img3_nonono.png', img)
import pdb; pdb.set_trace()