# import os
# import cv2
# import numpy as np
# from multiprocessing import Pool
# from tqdm import tqdm

# # 假设你有一个存有文件路径的列表 files
# files = ['/mnt/public/usr/wangmingze/Datasets/CD/WHU-main/train.txt', 
#          '/mnt/public/usr/wangmingze/Datasets/CD/WHU-main/val.txt'
#          ]

# def count_zeros_and_255s(label_path):
#     label_img = cv2.imread(label_path.strip(), cv2.IMREAD_UNCHANGED)
#     zero_count = np.sum(label_img == 0)
#     two_fifty_five_count = np.sum(label_img == 255)

#     return zero_count, two_fifty_five_count

# all_image_paths = []

# for file_path in files:
#     with open(file_path, 'r') as f:
#         diff_lines = f.readlines()
#         image_paths = [line.split('\t')[3] for line in diff_lines]
#         all_image_paths.extend(image_paths)

# pool = Pool(10)

# results = list(tqdm(pool.imap(count_zeros_and_255s, all_image_paths), total=len(all_image_paths)))

# zero_count = sum(i[0] for i in results)
# two_fifty_five_count = sum(i[1] for i in results)

# if two_fifty_five_count != 0:
#     ratio = zero_count / two_fifty_five_count
#     print('The ratio of 0 to 255 is: {}'.format(ratio))
# else:
#     print('Error: Zero 255s')
# pool.close()
# pool.join()

import os
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

# 假设你有一个存有文件路径的列表 files
file_path = '/mnt/public/usr/wangmingze/opencd/data_list/data_list_all_CD.txt'

def count_zeros_and_255s(label_path):
    try:
        label_img = cv2.imread(label_path.strip(), cv2.IMREAD_UNCHANGED)
    except:
        print(label_path.strip())
    zero_count = np.sum(label_img == 0)
    two_fifty_five_count = np.sum(label_img != 0)

    return zero_count, two_fifty_five_count

all_image_paths = []
with open(file_path, 'r') as f:
    diff_lines = f.readlines()
    for line in diff_lines:
        line = line.strip()
        if line and not line.startswith("#"):
            with open(line, 'r') as infile:
                for subline in infile.readlines():
                    if subline and not subline.startswith("#"):
                        a, b, cd, la,lb = subline.split('\t')
                        if b == '**':
                            image_path = la
                            all_image_paths.append(image_path)
                        elif la == '**':
                            image_path = cd
                            all_image_paths.append(image_path)
                        else:
                            for pic_path in [cd]:
                                image_path = pic_path
                                all_image_paths.append(image_path)

pool = Pool(64)

results = list(tqdm(pool.imap(count_zeros_and_255s, all_image_paths), total=len(all_image_paths)))

zero_count = sum(i[0] for i in results)
two_fifty_five_count = sum(i[1] for i in results)

if two_fifty_five_count != 0:
    ratio = zero_count / two_fifty_five_count
    print(zero_count)
    print(two_fifty_five_count)
    print('The ratio of 0 to 255 is: {}'.format(ratio))
else:
    print('Error: Zero 255s')
pool.close()
pool.join()

import numpy as np

class_num = [zero_count, two_fifty_five_count]
class_arr = np.array(class_num, dtype=np.float32)
weight = 0.5 + 0.5 * (1/class_arr/np.sum(1/class_arr))
print(', '.join(['{:.07f}'.format(k) for k in weight.tolist()]))
