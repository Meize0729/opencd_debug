import os
from multiprocessing import Pool

def process_file(filename):
    # 构建对应的 B 和标签文件名
    b_filename = filename
    label_filename = filename.split('.')[0] + '.png'
    if not (b_filename in b_filenames and label_filename in label_filenames):
        return None
    # 构建每一组对应数据的绝对路径，并将路径添加到 result 列表中
    a_path = os.path.join(a_folder, filename)
    b_path = os.path.join(b_folder, b_filename)
    label_path = os.path.join(label_folder, label_filename)
    return f'{a_path}\t{b_path}\t{label_path}\t**\t**\n'

# 定义 A、B 和标签文件夹的路径
folder = '/mnt/public/usr/wangmingze/Datasets/CD/SECOND/test'
mode = folder.split('/')[-1]
a_folder = folder + '/A'
b_folder = folder + '/B'
label_folder = folder + '/label'

# 列出 A，B和label 文件夹下的所有文件名
a_filenames = os.listdir(a_folder)
b_filenames = os.listdir(b_folder)
label_filenames = os.listdir(label_folder)
# 使用 Pool 对象创建多个进程
with Pool(10) as p:
    results = p.map(process_file, a_filenames)
# Filter out None and write data_strs to txt file
with open(f'{folder}/../test_new.txt', 'w') as f:
    for result in results:
        if result is not None:
            f.write(result)