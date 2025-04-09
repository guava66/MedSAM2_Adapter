import os
import shutil
from tqdm import tqdm  # 导入 tqdm

# 定义源文件夹和目标文件夹
src_imgs_dir = "/data/pyhData/MedSAM-MedSAM2/data/npy/imgs"
src_gts_dir = "/data/pyhData/MedSAM-MedSAM2/data/npy/gts"
dst_imgs_dir = "/data/pyhData/MedSAM-MedSAM2/data/nan_npy/imgs"
dst_gts_dir = "/data/pyhData/MedSAM-MedSAM2/data/nan_npy/gts"

# 确保目标文件夹存在
os.makedirs(dst_imgs_dir, exist_ok=True)
os.makedirs(dst_gts_dir, exist_ok=True)

# 需要移动的文件列表
files_to_move = [
    "TCGA_CS_4943_20000902_20.npy",
    "TCGA_CS_5393_19990606_1.npy",
    "TCGA_CS_5396_20010302_1.npy",
    "TCGA_CS_5396_20010302_2.npy",
    "TCGA_CS_5397_20010315_1.npy",
    "TCGA_CS_6188_20010812_22.npy",
    "TCGA_CS_6188_20010812_23.npy",
    "TCGA_CS_6188_20010812_24.npy",
    "TCGA_CS_6290_20000917_1.npy",
    "TCGA_DU_5849_19950405_1.npy",
    "TCGA_DU_5849_19950405_38.npy",
    "TCGA_DU_6408_19860521_56.npy",
    "TCGA_DU_8162_19961029_37.npy",
    "TCGA_DU_8164_19970111_37.npy",
    "TCGA_DU_A5TP_19970614_1.npy",
    "TCGA_DU_A5TP_19970614_38.npy",
    "TCGA_DU_A5TR_19970726_1.npy",
    "TCGA_DU_A5TS_19970726_35.npy",
    "TCGA_DU_A5TU_19980312_1.npy",
    "TCGA_DU_A5TU_19980312_23.npy",
    "TCGA_FG_7637_20000922_1.npy",
    "TCGA_HT_7602_19951103_1.npy",
    "TCGA_HT_7684_19950816_1.npy",
    "TCGA_HT_7684_19950816_2.npy",
    "TCGA_HT_7684_19950816_3.npy",
    "TCGA_HT_7690_19960312_1.npy",
    "TCGA_HT_7690_19960312_24.npy",
    "TCGA_HT_7860_19960513_1.npy",
    "TCGA_HT_8111_19980330_1.npy",
    "TCGA_HT_8111_19980330_22.npy"
]

# 使用 tqdm 显示进度条
for file_name in tqdm(files_to_move, desc="Moving files"):
    # 移动 imgs 中的文件
    src_img_path = os.path.join(src_imgs_dir, file_name)
    dst_img_path = os.path.join(dst_imgs_dir, file_name)
    if os.path.exists(src_img_path):
        shutil.move(src_img_path, dst_img_path)
    else:
        print(f"File {src_img_path} does not exist in imgs folder.")

    # 移动 gts 中的文件
    src_gt_path = os.path.join(src_gts_dir, file_name)
    dst_gt_path = os.path.join(dst_gts_dir, file_name)
    if os.path.exists(src_gt_path):
        shutil.move(src_gt_path, dst_gt_path)
    else:
        print(f"File {src_gt_path} does not exist in gts folder.")