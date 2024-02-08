import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def process_image(file, src_folder1, src_folder2, dst_folder):
    img1 = cv2.imread(os.path.join(src_folder1, file), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(src_folder2, file), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print(f"Error reading file {file}")
        return
        
    result = ((img1 == 38) | (img2 == 38)).astype('uint8') * 255
    cv2.imwrite(os.path.join(dst_folder, file), result)

def main():
    src_folder1 = './label1'   # Update source folder path
    src_folder2 = './label2'   # Update source folder path
    dst_folder = './label'  # Update destination folder path

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files_list1 = os.listdir(src_folder1)
    files_list2 = os.listdir(src_folder2)

    files_list1 = [file for file in files_list1 if file.endswith('.png')]
    files_list2 = [file for file in files_list2 if file.endswith('.png')]

    common_files = list(set(files_list1) & set(files_list2))
  
    pool = Pool(processes = cpu_count())
    func = partial(process_image, src_folder1=src_folder1, src_folder2=src_folder2, dst_folder=dst_folder)
    for _ in tqdm(pool.imap_unordered(func, common_files), total=len(common_files)):
        pass
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()