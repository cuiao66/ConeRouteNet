

# %%
#####deeplabv3 性能测试
import os
os.sys.path.append("/root/End2End/utils/deeplabv3/model")
from utils.deeplabv3.model.deeplabv3 import DeepLabV3
import os, torch


network = DeepLabV3("eval_val", project_dir="/root/End2End/utils")
network.load_state_dict(torch.load("/root/End2End/utils/deeplabv3/pretrained_models/model_13_2_2_2_epoch_580.pth"))
network = network.cuda(0)
network.eval()


import cv2
import numpy as np


# %%
img = cv2.imread('/root/End2End/TestImg.jpg')
# img = cv2.imread(img_path, -1) # (shape: (1024, 2048, 3))
# resize img without interpolation (want the image to still match
# label_img, which we resize below):
img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_NEAREST) # (shape: (512, 1024, 3))


# normalize the img (with the mean and std for the pretrained ResNet):
img = img/255.0
img = img - np.array([0.485, 0.456, 0.406])
img = img/np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
img = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
img = img.astype(np.float32)

# convert numpy -> torch:
img = torch.from_numpy(img) # (shape: (3, 512, 1024))


# %%
img = img.unsqueeze(0).cuda()




# %%
import time


# %%
start = time.time()
out = network(img)
end = time.time()
print(end - start)


# %%



# %%
def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        19: [81,  0, 81]
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color


# %%
outputs = out.data.cpu().numpy()
pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
pred_label_imgs = pred_label_imgs.astype(np.uint8)
for i in range(pred_label_imgs.shape[0]):
    if i == 0:
        pred_label_img = pred_label_imgs[i] # (shape: (img_h, img_w))

        img = img[i] # (shape: (3, img_h, img_w))

        img = img.data.cpu().numpy()
        img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
        img = img*np.array([0.229, 0.224, 0.225])
        img = img + np.array([0.485, 0.456, 0.406])
        img = img*255.0
        img = img.astype(np.uint8)

        pred_label_img_color = label_img_to_color(pred_label_img)
        overlayed_img = 0.35*img + 0.65*pred_label_img_color
        overlayed_img = overlayed_img.astype(np.uint8)

        cv2.imwrite("./deeplab.png", overlayed_img)



# %%



