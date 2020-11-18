import os, pdb
import numpy as np
import pickle

from collections import defaultdict

print("Computing class weights.")

#########################
# Please set data dir
data_dir = "../data/construction"
#########################


label_img_path = os.path.join(data_dir, "label_npy")
label_img_names = [os.path.join(label_img_path,j) for j in os.listdir(label_img_path)]

# read label list
label_list_name = os.path.join(data_dir, "label_list.txt")

with open(label_list_name, "r") as f:
    label2id = {}
    lines = f.readlines()
    label_dict = {x.strip():i for x,i in zip(lines,range(len(lines)))}

num_classes = len(label_dict)

trainId_to_count = defaultdict(int)

for step, label_path in enumerate(label_img_names):
    if step % 100 == 0:
        print(step)

    label_ar = np.load(label_path, allow_pickle=True)

    for trainId in range(num_classes):
        # count how many pixels in label_img which are of object class trainId:
        trainId_mask = np.equal(label_ar, trainId)
        trainId_count = np.sum(trainId_mask)

        # add to the total count:
        trainId_to_count[trainId] += trainId_count

class_weights = []
total_count = sum(trainId_to_count.values())

for trainId in range(num_classes):
    count = trainId_to_count[trainId]
    trainId_prob = float(count)/float(total_count)
    trainId_weight = 1/np.log(1.02 + trainId_prob)
    class_weights.append(trainId_weight)

# for trainId, count in trainId_to_count.items():
#     trainId_prob = float(count)/float(total_count)
#     trainId_weight = 1/np.log(1.02 + trainId_prob)
#     class_weights.append(trainId_weight)

print(class_weights)

with open(os.path.join(data_dir,"class_weights.pkl"), "wb") as file:
    pickle.dump(class_weights, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)


print("Done")
