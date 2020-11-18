import numpy as np
import os, pdb
import shutil
np.random.seed(2020)

def check_exist_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


# create save dirs
save_tr_dir = "train"
save_tr_img_dir = os.path.join(save_tr_dir, "image_npy")
check_exist_path(save_tr_img_dir)
save_tr_label_dir = os.path.join(save_tr_dir, "label_npy")
check_exist_path(save_tr_label_dir)

save_va_dir = "val"
save_va_img_dir = os.path.join(save_va_dir, "image_npy")
check_exist_path(save_va_img_dir)
save_va_label_dir = os.path.join(save_va_dir, "label_npy")
check_exist_path(save_va_label_dir)

# split
read_img_dir = "image_npy"
read_label_dir = "label_npy"
img_files = os.listdir(read_img_dir)

all_index = np.arange(len(img_files))
np.random.shuffle(all_index)

# set test sample number
num_te = 200
tr_index = all_index[num_te:]
va_index = all_index[:num_te]

#tr_ratio = 0.7
#tr_index = all_index[:int(len(all_index)*tr_ratio)]
#va_index = all_index[int(len(all_index)*tr_ratio):]

f_tr = open("train.txt","w")
f_va = open("val.txt","w")

for tri in tr_index:
    # cp images
    img_filename = img_files[tri]
    ori_path = os.path.join(read_img_dir, img_filename)
    target_path = os.path.join(save_tr_img_dir, img_filename)
    shutil.copyfile(ori_path, target_path)
    # cp labels
    ori_path_l = os.path.join(read_label_dir, img_filename)
    target_path_l = os.path.join(save_tr_label_dir, img_filename)
    shutil.copyfile(ori_path_l, target_path_l)
    f_tr.write(img_filename + "\n")
    

for vai in va_index:
    # cp images
    img_filename = img_files[vai]
    ori_path = os.path.join(read_img_dir, img_filename)
    target_path = os.path.join(save_va_img_dir, img_filename)
    shutil.copyfile(ori_path, target_path)
    # cp labels
    ori_path_l = os.path.join(read_label_dir, img_filename)
    target_path_l = os.path.join(save_va_label_dir, img_filename)
    shutil.copyfile(ori_path_l, target_path_l)
    f_va.write(img_filename + "\n")

f_tr.close()
f_va.close()
print("done")


        
