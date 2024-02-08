import os

def delete_non_number_files(path):
    for file_name in os.listdir(path):   # 遍历path下的所有文件和文件夹
        file_path = os.path.join(path, file_name)  # 获取文件的绝对路径

        if os.path.isfile(file_path):   # 如果是文件
            # 判断文件名是否符合要求，如果不符合则删除
            if not file_name.split('.')[0].isdigit():
                os.remove(file_path)
                print(f"Delete file {file_path}")
        
        elif os.path.isdir(file_path):  # 如果是文件夹
            # 递归遍历子文件夹
            delete_non_number_files(file_path)

if __name__ == '__main__':
    path = '/mnt/public/usr/wangmingze/Datasets/CD/WHU-512/train'   # 指定需要遍历的文件夹路径
    delete_non_number_files(path)