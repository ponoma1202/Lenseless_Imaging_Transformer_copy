import os
import numpy as np
import random

pattern_path= '/home/ponoma/workspace/Lensless_Imaging_Transformer/datasets/' #lenseless camera pictures
ori_path= '/home/ponoma/workspace/Lensless_Imaging_Transformer/datasets/' #lensed camera pictures
save_path=  '/home/ponoma/workspace/Lensless_Imaging_Transformer/datasets/' 

#mirflickr25k
train_ori_mk_files=[]
val_ori_mk_files=[]
train_pattern_mk_files=[] # pattern = diffusercam images
val_pattern_mk_files=[]
test_pattern_mk_files=[]
test_ori_mk_files =[]

# files=os.listdir(pattern_path+'mirflickr_dataset/diffuser_images_npy') # original path: mirflickr25k_1600/train/
# #files.sort()
# random.shuffle(files)
# i = 0
# for file in files:
#     if i < 0.7 * len(files):
#         train_pattern_mk_files.append(pattern_path+'mirflickr_dataset/diffuser_images_npy/'+file)
#         train_ori_mk_files.append(ori_path + 'mirflickr_dataset/ground_truth_lensed/' + file[:-3] + 'jpg')   
#     elif i < 0.85 * len(files):
#         val_pattern_mk_files.append(pattern_path+'mirflickr_dataset/diffuser_images_npy/'+file)
#         val_ori_mk_files.append(ori_path + 'mirflickr_dataset/ground_truth_lensed' + file[:-3] + 'jpg')
#     else: 
#         test_pattern_mk_files.append(pattern_path+'mirflickr_dataset/diffuser_images_npy/'+file)
#         test_ori_mk_files.append(ori_path + 'mirflickr_dataset/ground_truth_lensed/' + file[:-3] + 'jpg')
#     i += 1

# files=os.listdir(pattern_path+'mirflickr_dataset/ground_truth_lensed_npy')
# files.sort()
# for file in files:
#     val_pattern_mk_files.append(pattern_path+'mirflickr_dataset/ground_truth_lensed_npy'+file)
#     val_ori_mk_files.append(ori_path + 'mirflickr25k/val/' + file[:-3]+'jpg')

#fruits,PetImages
ori_fP_files = []
pattern_fP_files = []
ori_val_files = []
pattern_val_files = []
ori_test_files = []
pattern_test_files = []

ori_folder_name='PetImages/'
pattern_folder_name = 'PetImages_1600/'

folders=os.listdir(pattern_path+pattern_folder_name)
folders.sort()
for folder in folders:
    # if folder.startswith("."):      # Added this line because my PetImages has a hidden directory for some reason.
    #     continue
    files=os.listdir(pattern_path+pattern_folder_name+folder+'/')
    #files.sort()
    random.shuffle(files)
    i = 0
    for file in files:
        if os.path.exists(ori_path+ori_folder_name+folder+'/'+file[:-3]+'jpg'):     
            if i < 0.7 * len(files):
                ori_fP_files.append(ori_path+ori_folder_name+folder+'/'+file[:-3]+'jpg')
                pattern_fP_files.append(pattern_path + pattern_folder_name + folder + '/' + file)
            elif i < 0.85 * len(files):
                ori_val_files.append(ori_path+ori_folder_name+folder+'/'+file[:-3]+'jpg')
                pattern_val_files.append(pattern_path + pattern_folder_name + folder + '/' + file)
            else:
                ori_test_files.append(ori_path+ori_folder_name+folder+'/'+file[:-3]+'jpg')
                pattern_test_files.append(pattern_path + pattern_folder_name + folder + '/' + file)
        if os.path.exists(ori_path + ori_folder_name + folder + '/' + file[:-3] + 'JPEG'):
            ori_fP_files.append(ori_path + ori_folder_name + folder + '/' + file[:-3] + 'JPEG')
            pattern_fP_files.append(pattern_path + pattern_folder_name + folder + '/' + file)
        i += 1

train_patterns=train_pattern_mk_files+pattern_fP_files
train_targets=train_ori_mk_files+ori_fP_files

val_patterns=val_pattern_mk_files+pattern_val_files
val_targets=val_ori_mk_files+ori_val_files

test_patterns=test_pattern_mk_files+pattern_test_files
test_targets=test_ori_mk_files+ori_test_files

np.save(save_path+'train_patterns.npy',train_patterns)     
np.save(save_path+'train_targets.npy',train_targets)
np.save(save_path+'val_patterns.npy',val_patterns)
np.save(save_path+'val_targets.npy',val_targets)
np.save(save_path+'test_patterns.npy',test_patterns)
np.save(save_path+'test_targets.npy',test_targets)
