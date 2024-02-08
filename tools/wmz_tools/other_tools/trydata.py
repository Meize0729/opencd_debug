import os
from PIL import Image

# 读取第一个文件夹下所有的图片，存成set
folder1 = '/mnt/public/usr/wangmingze/Datasets/CD/LEVIR-CD+/train/A'
images1 = set()
for filename in os.listdir(folder1):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # 添加你需要的图片格式
        with Image.open(os.path.join(folder1, filename)) as img:
            images1.add(img.tobytes())  # 将图片转换为字节并添加到集合中

# 读取并遍历第二个文件夹下所有的图片，如果这个图片出现在set里，则+1
folder2 = '/mnt/public/usr/wangmingze/Datasets/CD/levir-CD/test/A'
count = 0
for filename in os.listdir(folder2):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # 添加你需要的图片格式
        with Image.open(os.path.join(folder2, filename)) as img:
            if img.tobytes() in images1:  # 如果图片在第一个文件夹的图片集合中
                count += 1

print('Number of matching images:', count)