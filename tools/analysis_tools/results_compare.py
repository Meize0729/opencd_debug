import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

# 这是你的字符串列表
# strings = ['fc_siam_conc', 'fc_siam_diff', 'bit_r18', 'snunet_c32', 'tinycd', 'tinycd_v2_l', 
#            'changeformer_mitb0', 'changeformer_mitb1', 'changer_s50', 'ViT_L_finetune']

strings = [
    '/mnt/public/usr/wangmingze/Datasets/CD/AerialImageDataset/val/images_512_nooverlap',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_bx_ablation/inria/unet/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_bx_ablation/inria/deeplabv3/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_bx_ablation/inria/segformer-mitb0/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_bx_ablation/inria/buildformer/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_bx_ablation/inria/swin_b/compare_pixel',
]


# 这是你要保存图片的路径
save_dir = '/mnt/public/usr/wangmingze/opencd/pictures_tmp/tmp_inria'

# 创建保存路径
os.makedirs(save_dir, exist_ok=True)

# 获取第一个路径下的所有文件名
# s2looking 看到1640了
file_names = os.listdir(os.path.join(strings[0]))

def process_file(file_name):
    # 创建一个新的图像
    fig, axs = plt.subplots(1, 6, figsize=(40, 8))

    flag = False
    # 遍历所有字符串
    for i, string in enumerate(strings):
        # 计算当前图片的行和列
        # row = i // 5
        # col = i % 5

        # 读取图片
        img_path = os.path.join(string, file_name)
        img = mpimg.imread(img_path)
        nonzero_pixels = np.count_nonzero(img)
        total_pixels = img.shape[0] * img.shape[1]
        nonzero_ratio = nonzero_pixels / total_pixels
        if nonzero_ratio < 0.15:
            flag = True
            break

        # 在对应的位置显示图片和字符串
        # axs[row, col].imshow(img)
        # axs[row, col].set_title(string)
        # axs[row, col].axis('off')
    
        axs[i].imshow(img)  # 直接使用索引，因为只有一行
        axs[i].set_title(string.split('/')[-3])
        axs[i].axis('off')

    if flag:
        plt.close(fig)
    else:
        # 调整子图之间的间距
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # 保存图片
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

# 创建一个进程池，指定进程数为4
pool = Pool(64)

# 使用多进程处理所有文件，并使用tqdm显示进度
for _ in tqdm(pool.imap(process_file, file_names), total=len(file_names)):
    pass

# 关闭进程池，等待所有进程结束
pool.close()
pool.join()

index = 0
# # 遍历所有文件名
for file_name in file_names:
    index += 1
    # 使用imgcat显示图片
    save_path = os.path.join(save_dir, file_name)

    if not os.path.exists(save_path):
        continue
    print(index)
    print(file_name)
    os.system('imgcat {}'.format(save_path))

    # 等待用户按回车键
    input("Press Enter to continue...")

# file_names = ['test_82_crop_3.png', 'test_103_crop_0.png', 'test_21_crop_3.png', 'test_71_crop_3.png', 'test_21_crop_1.png', 'test_113_crop_2.png', 'test_113_crop_3.png', 'test_77_crop_3.png']
# 'test_10_crop_3.png', '
# ['121.png', '123.png', '125.png','137.png', '143.png', 

# good
# file_names = ['128.png', '143.png', '209.png', '2_1380.png', '2_1403.png', '2_1449.png', '2_1520.png', '2_1568.png', '2_1569.png', '2_176.png', '2_863.png', '2_932.png', '491.png', '83.png']
# file_names = ['austin1_0_39.png', 'austin1_0_40.png', 'austin1_0_58.png', 'austin1_0_9.png','austin2_0_42.png', 'austin3_0_0.png', 'austin5_0_59.png']
# file_names = ['austin2_0_99.png', 'austin4_0_40.png', 'austin4_0_41.png', 'chicago1_0_41.png', 'chicago3_0_15.png', 'chicago4_0_31.png', 'chicago5_0_41.png', 'tyrol-w4_0_50.png', 'vienna2_0_43.png', 'vienna2_0_53.png','vienna2_0_89.png','vienna3_0_24.png']