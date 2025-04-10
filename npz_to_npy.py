"""
Convert the preprocessed .npz files to .npy files for training
"""
# %% import packages
import numpy as np
import os
join = os.path.join
listdir = os.listdir
makedirs = os.makedirs
from tqdm import tqdm
import multiprocessing as mp
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-npz_dir", type=str, default=r'/data/pyhData/brain_dataset/18TEMPORAL LOBE EPILPSY MULTIPLE MODALITY DATA/HFH_T1_HP_SEG_TLE_1/npz_train/HFH_T1_HP_SEG_TLE_1',
                    help="Path to the directory containing preprocessed .npz data, [default: data/npz/MedSAM_train/CT_Abd]")
parser.add_argument("-npy_dir", type=str, default=r"/data/pyhData/brain_dataset/18TEMPORAL LOBE EPILPSY MULTIPLE MODALITY DATA/HFH_T1_HP_SEG_TLE_1/npy",
                    help="Path to the directory where the .npy files for training will be saved, [default: ./data/npy]")
parser.add_argument("-num_workers", type=int, default=1,
                    help="Number of workers to convert npz to npy in parallel, [default: 4]")
args = parser.parse_args()
# %%
npz_dir = args.npz_dir
npy_dir = args.npy_dir
makedirs(join(npy_dir, "imgs"), exist_ok=True)
makedirs(join(npy_dir, "gts"), exist_ok=True)
npz_names = [f for f in listdir(npz_dir) if f.endswith(".npz")]
num_workers = args.num_workers
# convert npz files to npy files
def convert_npz_to_npy(npz_name):
    """
    Convert npz files to npy files for training

    Parameters
    ----------
    npz_name : str
        Name of the npz file to be converted
    """
    name = npz_name.split(".npz")[0]
    npz_path = join(npz_dir, npz_name)
    npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    imgs = npz["imgs"]
    gts = npz["gts"]
    if len(gts.shape) > 2: ## 3D image
        for i in range(imgs.shape[0]):
            img_i = imgs[i, :, :]
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
            gt_i = gts[i, :, :]
            gt_i = np.uint8(gt_i)
            gt_i = cv2.resize(gt_i, (256, 256), interpolation=cv2.INTER_NEAREST)
            assert gt_i.shape == (256, 256)
            np.save(join(npy_dir, "imgs", name + "-" + str(i).zfill(3) + ".npy"), img_3c)
            np.save(join(npy_dir, "gts", name + "-" + str(i).zfill(3) + ".npy"), gt_i)
    else: ## 2D image
        if len(imgs.shape) < 3:
            img_3c = np.repeat(imgs[:, :, None], 3, axis=-1)
        else:
            img_3c = imgs

        gt_i = gts
        gt_i = np.uint8(gt_i)
        gt_i = cv2.resize(gt_i, (256, 256), interpolation=cv2.INTER_NEAREST)
        assert gt_i.shape == (256, 256)
        np.save(join(npy_dir, "imgs", name + ".npy"), img_3c)
        np.save(join(npy_dir, "gts", name + ".npy"), gt_i)
# %%
if __name__ == "__main__":
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(npz_names)) as pbar:
            pbar.set_description("Converting npz to npy")
            for i, _ in enumerate(pool.imap_unordered(convert_npz_to_npy, npz_names)):
                pbar.update()
