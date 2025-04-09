# -*- coding: utf-8 -*-
# %% import packages
# pip install connected-components-3d
import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import os

join = os.path.join
from tqdm import tqdm
import cc3d

import multiprocessing as mp
from functools import partial

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-modality", type=str, default="MR", help="CT or MR, [default: CT]")
parser.add_argument("-anatomy", type=str, default="Abd",
                    help="Anaotmy name, [default: Abd]")
parser.add_argument("-img_name_suffix", type=str, default="_T1_brain.nii",
                    help="Suffix of the image name, [default: .nii.gz]")
parser.add_argument("-gt_name_suffix", type=str, default="_T1_Labels_Manual.nii",
                    help="Suffix of the ground truth name, [default: .nii.gz]")
parser.add_argument("-img_path", type=str, default=r"/data/pyhData/brain_dataset/18TEMPORAL LOBE EPILPSY MULTIPLE MODALITY DATA/HFH_T1_HP_SEG_TLE_1/T1",
                    help="Path to the nii images, [default: data/FLARE22Train/images]")
parser.add_argument("-gt_path", type=str, default=r"/data/pyhData/brain_dataset/18TEMPORAL LOBE EPILPSY MULTIPLE MODALITY DATA/HFH_T1_HP_SEG_TLE_1/label",
                    help="Path to the ground truth, [default: data/FLARE22Train/labels]")
parser.add_argument("-output_path", type=str, default=r"/data/pyhData/brain_dataset/18TEMPORAL LOBE EPILPSY MULTIPLE MODALITY DATA/HFH_T1_HP_SEG_TLE_1",
                    help="Path to save the npy files, [default: ./data/npz]")
parser.add_argument("-num_workers", type=int, default=1,
                    help="Number of workers, [default: 4]")
parser.add_argument("-window_level", type=int, default=40,
                    help="CT window level, [default: 40]")
parser.add_argument("-window_width", type=int, default=400,
                    help="CT window width, [default: 400]")
parser.add_argument("-save_nii",default=False,
                    help="Save the image and ground truth as nii files for sanity check; they can be removed")

args = parser.parse_args()

# convert nii image to npz files, including original image and corresponding masks
modality = args.modality  # CT or MR
anatomy = args.anatomy  # anantomy + dataset name
img_name_suffix = args.img_name_suffix  # "_0000.nii.gz"
start = img_name_suffix.index('_') + 1
end = img_name_suffix.index('.')
substring = img_name_suffix[start:end]

gt_name_suffix = args.gt_name_suffix  # ".nii.gz"
prefix ="HFH_T1_HP_SEG_TLE_1"+"_"

nii_path = args.img_path  # path to the nii images
gt_path = args.gt_path  # path to the ground truth
output_path = args.output_path  # path to save the preprocessed files
npz_tr_path = join(output_path, "npz_train", prefix[:-1])
os.makedirs(npz_tr_path, exist_ok=True)
npz_ts_path = join(output_path, "npz_test", prefix[:-1])
os.makedirs(npz_ts_path, exist_ok=True)

num_workers = args.num_workers

voxel_num_thre2d = 100
voxel_num_thre3d = 1000

names = sorted(os.listdir(gt_path))
print(f"ori # files {len(names)=}")
names = [
    name
    for name in names
    if os.path.exists(join(nii_path,name.split(gt_name_suffix)[0] + img_name_suffix))
]
print(f"after sanity check # files {len(names)=}")

# set label ids that are excluded
remove_label_ids = [
    12
]  # remove deodenum since it is scattered in the image, which is hard to specify with the bounding box
tumor_id = [1,2,3,4,5]  # only set this when there are multiple tumors; convert semantic masks to instance masks
# set window level and width
# https://radiopaedia.org/articles/windowing-ct
WINDOW_LEVEL = args.window_level # only for CT images
WINDOW_WIDTH = args.window_width # only for CT images

save_nii = args.save_nii
# %% save preprocessed images and masks as npz files
def preprocess(name, npz_path):
    """
    Preprocess the image and ground truth, and save them as npz files

    Parameters
    ----------
    name : str
        name of the ground truth file
    npz_path : str
        path to save the npz files
    """
    image_name =name.split(gt_name_suffix)[0] + img_name_suffix
    gt_name = name
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    # remove label ids
    # 检查数据范围
    gt_data_ori = sitk.GetArrayFromImage(gt_sitk)
    data_min, data_max = gt_data_ori.min(), gt_data_ori.max()

    if data_max <= 1.0:  # 如果值在 [0, 1] 范围内，认为已归一化
        gt_data_ori = (gt_data_ori * 255).astype(np.uint8)  # 还原为整数标签
        print(npz_path)
    else:
        gt_data_ori = gt_data_ori.astype(np.uint8)  # 直接转换为 uint8 类型

    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori == remove_label_id] = 0
 
    if tumor_id is not None:
        # Tumor labels to combine (1: NEC, 2: CE, 3: TC, 4: ED)
        tumor_labels = [1, 2, 3, 4,5]

        # Check which tumor labels are actually present in the data
        present_tumor_labels = [label for label in tumor_labels if np.any(gt_data_ori == label)]

        if present_tumor_labels:  # If there are any tumor labels in the data
            # Combine all present tumor labels into one class (1) and leave background as 0
            gt_data_ori[np.isin(gt_data_ori, present_tumor_labels)] = 1  # Set all present tumor labels to 1
            gt_data_ori[gt_data_ori == 0] = 0  # Ensure background remains 0


    # find non-zero slices
    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index)

    if len(z_index) > 0:
        # crop the ground truth with non-zero slices
        gt_roi = gt_data_ori[z_index, :, :]
        # load image and preprocess
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        # nii preprocess start
        if modality == "CT":
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
        else:
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            image_data_pre[image_data == 0] = 0

        image_data_pre = np.uint8(image_data_pre)
        img_roi = image_data_pre[z_index, :, :]
        np.savez_compressed(join(npz_path, prefix+gt_name.split(gt_name_suffix)[0]+'_'+substring+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())
        print(gt_name.split(gt_name_suffix)[0]+'_'+substring+'.npz'+"+Preprocessing ")
        # save the image and ground truth as nii files for sanity check;
        # they can be removed
        if save_nii:
            img_roi_sitk = sitk.GetImageFromArray(img_roi)
            img_roi_sitk.SetSpacing(img_sitk.GetSpacing())
            sitk.WriteImage(
                img_roi_sitk,
                join(npz_path, prefix + gt_name.split(gt_name_suffix)[0] +'_'+substring+ "_img.nii.gz"),
            )
            gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
            gt_roi_sitk.SetSpacing(img_sitk.GetSpacing())
            sitk.WriteImage(
                gt_roi_sitk,
                join(npz_path, prefix + gt_name.split(gt_name_suffix)[0] +'_'+ substring+"_gt.nii.gz"),
            )

if __name__ == "__main__":
    tr_names = names[:83]
    ts_names = names[83:]

    preprocess_tr = partial(preprocess, npz_path=npz_tr_path)
    preprocess_ts = partial(preprocess, npz_path=npz_ts_path)

    with mp.Pool(num_workers) as p:
        with tqdm(total=len(tr_names)) as pbar:
            pbar.set_description("Preprocessing training data")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_tr, tr_names))):
                pbar.update()
        with tqdm(total=len(ts_names)) as pbar:
            pbar.set_description("Preprocessing testing data")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_ts, ts_names))):
                pbar.update()
