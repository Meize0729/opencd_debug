from multiprocessing import Pool
import cv2
import numpy as np
import random
from tqdm import tqdm

def process_line(line):
    items = line.split('\t')
    label_path = items[2]
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    num_zeros = np.sum(label_img == 0)
    num_ones = np.sum(label_img != 0)
    is_zero = num_ones == 0
    return (line, is_zero, num_zeros, num_ones)

if __name__ == "__main__":
    # 读取文件
    with open('/mnt/public/usr/wangmingze/Datasets/CD/S2Looking/train.txt', 'r') as f:
        lines = f.readlines()

    zero_labels = []
    non_zero_labels = []

    total_zeros_before = 0
    total_ones_before = 0
    total_zeros_after = 0
    total_ones_after = 0

    # Create a multiprocessing Pool
    pool = Pool()
    for result in tqdm(pool.imap_unordered(process_line, lines), total=len(lines)):
        line, is_zero, num_zeros, num_ones = result
        if is_zero:
            zero_labels.append(line)
        else:
            non_zero_labels.append(line)
        total_zeros_before += num_zeros
        total_ones_before += num_ones
    print(len(zero_labels))
    # 移除70%的全0标签图片，并加入保留数据的统计
    keep_rate = 0
    random.shuffle(zero_labels)

    keep_zero_labels = zero_labels[:int(len(zero_labels)*keep_rate)]
    remove_zero_labels = zero_labels[int(len(zero_labels)*keep_rate):]

    for result in tqdm(pool.imap_unordered(process_line, remove_zero_labels), total=len(remove_zero_labels)):
        _, _, num_zeros, _ = result
        total_zeros_after += num_zeros
        total_ones_after += 0  # all ones in kept zeroes are still zero

    total_zeros_after = total_zeros_before - total_zeros_after
    total_ones_after += total_ones_before

    # 计算0和1的比例
    print('Before removal, Zero:255 = {}:{}, ratio={}'.format(total_zeros_before, total_ones_before, total_zeros_before/total_ones_before))
    print('After removal,  Zero:255 = {}:{}, ratio={}'.format(total_zeros_after, total_ones_after, total_zeros_after/total_ones_after))

    # 保存新的txt文件
    with open(f'/mnt/public/usr/wangmingze/Datasets/CD/S2Looking/train_keep{keep_rate}.txt', 'w') as f:
        f.writelines(non_zero_labels)
        f.writelines(keep_zero_labels)

'''
levir-cd:
    前：
        The total number of zero images is: 1309 -> 392
        The total number of non-zero images is: 2696
        Before removal, 0:255 = 1001141738:48743093, ratio=20.53915080850532
        After removal,  0:255 = 102760448:48743093,  ratio=2.1082053204953572

'''