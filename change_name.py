# import os

# # 定义FLAIR文件夹路径
# flair_dir = '/data/pyhData/MedSAM-MedSAM2/data/Brain/segs'

# # 遍历FLAIR目录中的所有文件
# for filename in os.listdir(flair_dir):
#     file_path = os.path.join(flair_dir, filename)
    
#     # 确保只处理文件
#     if os.path.isfile(file_path):
#         # 获取文件名的前9个字符并构建新的文件名
#         new_filename = filename[0:9] + '_FLAIR_brain.nii'
#         new_file_path = os.path.join(flair_dir, new_filename)
        
#         # 重命名文件
#         os.rename(file_path, new_file_path)
#         print(f"文件重命名: {filename} -> {new_filename}")


import nibabel as nib
import numpy as np
import SimpleITK as sitk

# 文件路径
file_path = '/data/pyhData/MedSAM-MedSAM2/data/Brain/Flair/segs/sub-00003_FLAIR_brain.nii'

# 读取 NIfTI 文件
img = nib.load(file_path)

# 获取图像数据
img_data = img.get_fdata()

# 获取数据中的唯一值
unique_values = np.unique(img_data)

# 打印唯一值
print(f"图像数据的唯一值种类: {unique_values}")

# 读取图像
img_sitk = sitk.ReadImage(file_path)

# 获取图像数据
img_data = sitk.GetArrayFromImage(img_sitk)

# 打印基本信息
print(f"图像数据类型: {img_data.dtype}")
print(f"图像维度: {img_data.shape}")
print(f"图像空间分辨率: {img_sitk.GetSpacing()}")
print(f"图像的头信息: {img_sitk}")
