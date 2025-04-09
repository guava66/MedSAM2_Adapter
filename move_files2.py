import os
import shutil
import SimpleITK as sitk
import nibabel as nib
import numpy as np

# 定义路径
src_dir = '/data/pyhData/MedSAM-MedSAM2/data/Brain/Flair/segs'
dest_dir = '/data/pyhData/MedSAM-MedSAM2/data/Brain/segs'

# 如果目标文件夹不存在，则创建它
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 遍历源目录下所有子文件和文件夹
for root, dirs, files in os.walk(src_dir):
    # 跳过目标文件夹，防止在目标文件夹下再处理文件
    if root == dest_dir:
        continue
        
    for file in files:
        file_path=os.path.join(root,file)
                # 读取图像
        img_sitk = sitk.ReadImage(file_path)

        # 获取图像数据
        img_data = sitk.GetArrayFromImage(img_sitk)
        print(f"图像空间分辨率: {img_sitk.GetSpacing()}")

        # 读取 NIfTI 文件
        img = nib.load(file_path)

        # 获取图像数据
        img_data = img.get_fdata()

        # 获取数据中的唯一值
        unique_values = np.unique(img_data)

        # 打印唯一值
        print(f"图像数据的唯一值种类: {unique_values}")


        img_shape = img_data.shape

        # 打印图像的维度顺序
        print(f"图像形状: {img_shape}")
