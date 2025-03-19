import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*iCCP: known incorrect sRGB profile.*")
import os
import numpy as np
import logging as logger
from tqdm import tqdm


def generate_flist(path_input, path_gt, nickname):
    # NOTE: The image and ground-truth should have the same name.
    # Example:
    # path_input = 'tampCOCO/sp_images/'
    # path_gt = 'tampCOCO/sp_masks/'
    # nickname = 'tampCOCO_sp'
    res = []
    flag = False
    image_files = sorted(os.listdir(path_input))
    mask_files = sorted(os.listdir(path_gt))
    assert len(image_files) == len(mask_files), "Images and masks count do not match!"
    for image_file, mask_file in tqdm(zip(image_files, mask_files)):
        res.append((path_input + image_file, path_gt + mask_file))
    save_name = '%s_%s.npy' % (nickname, len(res))
    np.save('flist/' + save_name, np.array(res))
    if flag:
        logger.info('Note: The following score is meaningless since no ground-truth is provided.')
    return save_name


def generate_npy(dataset_name):
    dataset_name = dataset_name
    path_input = f"/mnt/h/Academic/image_forgery/datasets/{dataset_name}/tamper/"
    path_gt = f"/mnt/h/Academic/image_forgery/datasets/{dataset_name}/gt/"
    nickname = f"{dataset_name}"
    generate_flist(path_input, path_gt, nickname)


generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/CASIAv1_Facebook/', '/mnt/h/Academic/image_forgery/datasets/CASIAv1/gt/', 'CASIAv1_Facebook')
generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/CASIAv1_Weibo/', '/mnt/h/Academic/image_forgery/datasets/CASIAv1/gt/', 'CASIAv1_Weibo')
generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/CASIAv1_Wechat/', '/mnt/h/Academic/image_forgery/datasets/CASIAv1/gt/', 'CASIAv1_Wechat')
generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/CASIAv1_Whatsapp/', '/mnt/h/Academic/image_forgery/datasets/CASIAv1/gt/', 'CASIAv1_Whatsapp')

generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/DSO_Facebook/', '/mnt/h/Academic/image_forgery/datasets/DSO/gt/', 'DSO_Facebook')
generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/DSO_Weibo/', '/mnt/h/Academic/image_forgery/datasets/DSO/gt/', 'DSO_Weibo')
generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/DSO_Wechat/', '/mnt/h/Academic/image_forgery/datasets/DSO/gt/', 'DSO_Wechat')
generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/DSO_Whatsapp/', '/mnt/h/Academic/image_forgery/datasets/DSO/gt/', 'DSO_Whatsapp')

generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/Columbia_Facebook/', '/mnt/h/Academic/image_forgery/datasets/Columbia/gt/', 'Columbia_Facebook')
generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/Columbia_Weibo/', '/mnt/h/Academic/image_forgery/datasets/Columbia/gt/', 'Columbia_Weibo')
generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/Columbia_Wechat/', '/mnt/h/Academic/image_forgery/datasets/Columbia/gt/', 'Columbia_Wechat')
generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/Columbia_Whatsapp/', '/mnt/h/Academic/image_forgery/datasets/Columbia/gt/', 'Columbia_Whatsapp')

generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/NIST_Facebook/', '/mnt/h/Academic/image_forgery/datasets/NIST/gt/', 'NIST_Facebook')
generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/NIST_Weibo/', '/mnt/h/Academic/image_forgery/datasets/NIST/gt/', 'NIST_Weibo')
generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/NIST_Wechat/', '/mnt/h/Academic/image_forgery/datasets/NIST/gt/', 'NIST_Wechat')
generate_flist('/mnt/h/Academic/image_forgery/datasets/OSN_dataset/NIST_Whatsapp/', '/mnt/h/Academic/image_forgery/datasets/NIST/gt/', 'NIST_Whatsapp')
