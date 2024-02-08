import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import tifffile

def process_image(filepath):
    # 读取图像
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    # img = tifffile.imread(filepath)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 调整大小
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)

    img[img < 128] = 0
    img[img >= 128] = 255
    filepath = filepath.replace('.tif', '.png')
    # 保存调整后的图像
    cv2.imwrite(os.path.join('/mnt/public/usr/wangmingze/Datasets/CD/EGY_BCD/label_512/', os.path.basename(filepath)), img)

if __name__ == '__main__':
    # 文件夹路径
    folder_path = '/mnt/public/usr/wangmingze/Datasets/CD/EGY_BCD/label'
    os.system('rm -rf /mnt/public/usr/wangmingze/Datasets/CD/EGY_BCD/label_512')
    os.makedirs('/mnt/public/usr/wangmingze/Datasets/CD/EGY_BCD/label_512', exist_ok=True)

    # 获取文件夹下所有的文件路径
    filepaths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
                 filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.tif')]

    # 使用多进程并行处理图像
    with Pool(80) as p:
        list(tqdm(p.imap(process_image, filepaths), total=len(filepaths)))



# import cv2

# # 读取图像 A、B 和标签
# img_a = cv2.imread('/Users/meize/Desktop/work_project/数据集/EGY_BCD/A/Cairo_0_0.png', cv2.IMREAD_UNCHANGED)
# img_b = cv2.imread('/Users/meize/Desktop/work_project/数据集/EGY_BCD/B/Cairo_0_0.png', cv2.IMREAD_UNCHANGED)
# label = cv2.imread('/Users/meize/Desktop/work_project/数据集/EGY_BCD/label/Cairo_0_0.png', cv2.IMREAD_GRAYSCALE)

# # 调整 A 和 B 的大小
# img_a = cv2.resize(img_a, (512, 512), interpolation=cv2.INTER_LINEAR)
# img_b = cv2.resize(img_b, (512, 512), interpolation=cv2.INTER_LINEAR)

# # 调整标签的大小
# label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)

# # 将标签的值限制为 0 或 255
# label[label < 128] = 0
# label[label >= 128] = 255

# # 保存调整后的图像 A、B 和标签
# cv2.imwrite('/Users/meize/Desktop/work_project/数据集/000result/A/1.png', img_a)
# cv2.imwrite('/Users/meize/Desktop/work_project/数据集/000result/B/1.png', img_b)
# cv2.imwrite('/Users/meize/Desktop/work_project/数据集/000result/label/1.png', label)