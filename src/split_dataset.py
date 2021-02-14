import os
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile

################################################################
# Split dataset and save the images to a new directory splitted
################################################################
path_orig="..\\data\\Original"
labels_type=os.listdir(path_orig)

test_size=0.9
path_split="..\\data\\split"

def copy_image_set(path_from, path_split, set_type, label, im_name):
    src=os.path.join(path_from, im_name)
    dst=os.path.join(path_split,set_type,label, im_name)
    copyfile(src, dst)

for label in labels_type:
    path_from=os.path.join(path_orig,label)
    im_name_list=pd.Series(os.listdir(path_from))

    for i, name in enumerate(im_name_list):
        if not name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            im_name_list.pop(i)
    
    X_train, X_test, _, _ = train_test_split(im_name_list, [label]*len(im_name_list), test_size=test_size, random_state=42)
    
    for im_name in X_train:
        copy_image_set(path_from, path_split, "train", label, im_name)
    for im_name in X_test:
        copy_image_set(path_from, path_split, "test", label, im_name)
