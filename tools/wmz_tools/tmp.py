import cv2
import numpy as np

from osgeo import gdal
ds = gdal.Open('/mnt/public/usr/wangmingze/Datasets/CD/openai_challenge/image/grid_001.tif')

src_ds = gdal.Open('/mnt/public/usr/wangmingze/Datasets/CD/AOI_4_Shanghai_Train/RGB-PanSharpen/RGB-PanSharpen_AOI_4_Shanghai_img3899.tif')
band = src_ds.GetRasterBand(1)
data_type = band.DataType
print("Data Type:", gdal.GetDataTypeName(data_type))

# # 获取统计信息
stats = band.GetStatistics(True, True)
print("Statistics: Min={0}, Max={1}, Mean={2}, StdDev={3}".format(*stats))
import pdb; pdb.set_trace()

# # 执行转换
# dst_ds = gdal.Translate('/mnt/public/usr/wangmingze/Datasets/CD/AOI_4_Shanghai_Train/RGB-PanSharpen_AOI_4_Shanghai_img4100.png', src_ds, format='PNG', outputType=gdal.GDT_Byte, scaleParams=[[stats[0], stats[1], 0, 255]])

from osgeo import gdal
import os

# 设置原始图像路径和输出目录
input_tif_path = '/mnt/public/usr/wangmingze/Datasets/CD/openai_challenge/image/grid_001.tif'
output_dir = '/mnt/public/usr/wangmingze/Datasets/CD/openai_challenge/tmp'


ds = gdal.Open(input_tif_path)
import pdb; pdb.set_trace()
if ds is None:
    raise Exception("无法打开图像")

# 计算完整块的数量
x_blocks = ds.RasterXSize // 512
y_blocks = ds.RasterYSize // 512
num_bands = ds.RasterCount

# 遍历每个块
for i in range(x_blocks):
    for j in range(y_blocks):
        x_offset = i * 512
        y_offset = j * 512

        # 对于多波段图像，需要分别处理每个波段
        block_data = []
        for b in range(1, num_bands + 1):
            band = ds.GetRasterBand(b)
            data = band.ReadAsArray(x_offset, y_offset, 512, 512)
            if data is None:
                raise Exception(f"读取波段 {b} 数据失败")
            block_data.append(data)

        output_path = os.path.join(output_dir, f'block_{i}_{j}.tif')
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path, 512, 512, num_bands, band.DataType)
        if out_ds is None:
            raise Exception(f"创建输出文件失败: {output_path}")

        # 设置地理变换和投影
        geo_transform = list(ds.GetGeoTransform())
        geo_transform[0] += x_offset * geo_transform[1]
        geo_transform[3] += y_offset * geo_transform[5]
        out_ds.SetGeoTransform(geo_transform)
        out_ds.SetProjection(ds.GetProjection())

        for k in range(num_bands):
            out_band = out_ds.GetRasterBand(k+1)
            out_band.WriteArray(block_data[k])

        out_ds = None

ds = None


