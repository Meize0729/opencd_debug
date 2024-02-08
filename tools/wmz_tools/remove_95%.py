import os
import cv2
import numpy as np
from multiprocessing import Pool
import tqdm

def process_line(line):
    parts = line.split('\t')
    if len(parts) < 4:
        print("!!!!!")
        return None  # 格式不正确的行

    mask_path = parts[3].strip()
    if not os.path.exists(mask_path):
        return None  # mask 文件不存在

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None  # 无法读取mask

    zeros = np.sum(mask == 0)
    ones = np.sum(mask == 1)
    if zeros / mask.size > 0.95 :
        return None  # 0或1占据超过95%

    return line  # 保留这一行

def filter_lines(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with Pool(processes=96) as pool:  # 调整进程数以适应你的系统
        results = list(tqdm.tqdm(pool.imap(process_line, lines), total=len(lines)))
    print(len(results))

    with open(output_file, 'w') as f:
        for line in results:
            if line is not None:
                f.write(line)

if __name__ == "__main__":
    input_file = '/mnt/public/usr/wangmingze/Datasets/CD/openai_challenge/tmp1_wo_remove.txt'
    output_file = '/mnt/public/usr/wangmingze/Datasets/CD/openai_challenge/tmp1_removed.txt'
    filter_lines(input_file, output_file)