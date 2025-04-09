import os 
import glob
import numpy as np
# import pdb
# pdb.set_trace()

import cv2
# from sklearn.model_selection import train_test_split
from PIL import Image

import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
class ImageDataset(Dataset):

    def __init__(self, data_root, transform=None, target_transform=None, load_flag=[], seq_len=6, device='cuda'):

        self.device = device
        self.seq_len = seq_len
        self.img_dir = data_root
        self.target_transform = target_transform
        self.transform = transform

        #self.data_dir_train  = os.path.join(data_root)
        #self.data_dir_test  = os.path.join(data_root)

        if 'left' in load_flag and 'right' in load_flag:
            self.___load_left_right_image()

    def __len__(self):
        return len(self.labels) - self.seq_len

    def __getitem__(self, idx):
        img_path = self.Xs[idx]
        image_left = []
        image_right = []
        # print(img_path[0])
        for path_ in img_path:

            assert len(path_) == 2
            image_left.append(torch.tensor(read_image(path_[0]), dtype=torch.float))
            image_right.append(torch.tensor(read_image(path_[1]), dtype=torch.float))
            # for _img_path in path_:
            #     image.append(torch.tensor(read_image(_img_path), dtype=torch.float))
                # show = ToPILImage()
                # show(read_image(_img_path)).show()

        label = self.labels[idx:idx+len(img_path)]

        if self.transform:
            for img_ in image_left:
                img_ = self.transform(img_)
            for img_ in image_right:
                img_ = self.transform(img_)
        if self.target_transform:
            label = self.target_transform(label)
        
        
        return torch.stack(image_left).to(self.device), torch.stack(image_right).to(self.device), torch.tensor(label, dtype=torch.float).to(self.device)

    # def __getitem__(self, idx):
    #     img_path = self.Xs[idx]
    #     image = []
    #     # print(img_path[0])
    #     for path_ in img_path:
    #         for _img_path in path_:
    #             image.append(torch.tensor(read_image(_img_path), dtype=torch.float))
    #             # show = ToPILImage()
    #             # show(read_image(_img_path)).show()

    #     label = self.labels[idx]
    #     if self.transform:
    #         for img_ in image:
    #             img_ = self.transform(img_)
    #     if self.target_transform:
    #         label = self.target_transform(label)
    #     return image, label

    def __load_left_right_image_single(self):

        xs = []
        for data_dir_train in self.img_dir:

            # Loading from all csv files;  * is HMB
            left_path = os.path.join(data_dir_train, '/left.csv')
            right_path = os.path.join(data_dir_train, '/right.csv')
            # Xs = np.zeros((0, self.seq_len, 3)) # (N,T,3)
            y = np.zeros((0, 2))  # (N,2)
            
            path_prefix = os.path.dirname(left_path)
            # load label
            left_data  = pd.read_csv(left_path)
            right_data  = pd.read_csv(right_path)
            
            left_img_names = left_data['filename'].values
            right_img_names = right_data['filename'].values

            assert len(left_img_names) == len(right_img_names)
        
            # combine sequential image path and point path
             # (Ni,T,2)
            for i in range(len(left_img_names)):
                
                xt = [] # (T,2)
                
                xt.append([os.path.join(path_prefix, left_img_names[i]), os.path.join(path_prefix, right_img_names[i])])
                
                xs.append(xt)
        
                # scale label
            angle = left_data['angle'].values[:]  # n-(self.seq_len-1)
            speed = left_data['speed'].values[:]
            angle_s = self.scale_label(angle, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
            speed_s = self.scale_label(speed, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
            ys = np.stack([angle_s, speed_s], axis=1)
            
            # print('{},{},{}'.format(len(xs), len(ys), (len(left_img_names)-self.seq_len+1)))
            # concatenate all data
            assert len(xs) == len(ys)
            
            # y  = np.concatenate((y, ys), axis=0)
            print("Loading data from {}, {}: {}".format(left_path, right_path, len(xs)))

        self.labels = ys;
        self.Xs = xs;

        pass

    def ___load_left_right_image(self):
        
        x = [] # (N, 2)
        y = np.zeros((0, 2))  # (N,2)
        for data_dir_train in self.img_dir:

            # Loading from all csv files;  * is HMB
            left_path = data_dir_train + '/left.csv'
            right_path = data_dir_train + '/right.csv'
            
            
            
            path_prefix = os.path.dirname(left_path)
            # load label
            left_data  = pd.read_csv(left_path)
            right_data  = pd.read_csv(right_path)
            
            left_img_names = left_data['filename'].values
            right_img_names = right_data['filename'].values

            assert len(left_img_names) == len(right_img_names)
        
            # combine sequential image path and point path
            xs = [] # (Ni,T,2)
            for i in range(len(left_img_names)):
                if i < (self.seq_len-1):
                    continue
                xt = [] # (T,2)
                for t in reversed(range(self.seq_len)):
                    xt.append([os.path.join(path_prefix, left_img_names[i-t]), os.path.join(path_prefix, right_img_names[i-t])])
                
                xs.append(xt)
        
            # scale label
            angle = left_data['angle'].values[self.seq_len-1:]  # n-(self.seq_len-1)
            speed = left_data['speed'].values[self.seq_len-1:]
            angle_s = self.scale_label(angle, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
            speed_s = self.scale_label(speed, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
            ys = np.stack([angle_s, speed_s], axis=1)
            
            y = np.vstack((y, ys))
            x.extend(xs)
            # print('{},{},{}'.format(len(xs), len(ys), (len(left_img_names)-self.seq_len+1)))
            # concatenate all data
            assert len(xs) == len(ys) == (len(left_img_names)-self.seq_len+1)
            
            # 
            print("Loading data from {}, {}: {}".format(left_path, right_path, len(xs)))

        assert len(y) == len(x)
        self.labels = y;
        self.Xs = x;

        pass
    
    def scale_label(self, y, y_min, y_max, a, b):
        """
        Sacle labels to a fixed interval [a,b]
        """
        return a + (b-a)*(y-y_min)/(y_max-y_min)
    
    def __load_train(self):
        """ 
        Load image paths & point paths and labels from multiple csv files      
        Return:
          - self.Xs: saves paths of sequential data: 
                    [[[i1,p1, flag], ...[it,pt,flag]],
                      [[],[]],  
                    ]; size is (N, T, 3)
            FLAG: left: -1; center: 0; right: 1
          - self.y: saves scaled labels:
                   [[angle, speed],
                    [,],
                   ]; size si (N,2)
        """
        # Loading from all csv files;  * is HMB
        file_paths = glob.glob(os.path.join(self.data_dir_train, '*/center.csv'))
        Xs = np.zeros((0, self.seq_len, 3)) # (N,T,3)
        y = np.zeros((0, 2))  # (N,2)

        for file_path in sorted(file_paths):
            path_prefix = os.path.dirname(file_path)
            # load label
            data  = pd.read_csv(file_path)
            img_names = data['filename'].values # relative path
            point_names = data['point_filename'].values
            assert len(img_names) == len(point_names)
            
            # combine sequential image path and point path
            xs = [] # (Ni,T,2)
            for i in range(len(img_names)):
                if i < (self.seq_len-1):
                    continue
                xt = [] # (T,2)
                for t in reversed(range(self.seq_len)):
                    xt.append([os.path.join(path_prefix, img_names[i-t]), 
                               os.path.join(path_prefix, 'points_bin', point_names[i-t][:-3]+'bin'),
                               0.0]) # CAM_FLAG=0 
                xs.append(xt)
            
            # scale label
            angle = data['angle'].values[self.seq_len-1:]  # n-(self.seq_len-1)
            speed = data['speed'].values[self.seq_len-1:]
            angle_s = self.scale_label(angle, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
            speed_s = self.scale_label(speed, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
            ys = np.stack([angle_s, speed_s], axis=1)
            
            # concatenate all data
            assert len(xs) == len(ys) == (len(img_names)-self.seq_len+1)
            Xs = np.concatenate((Xs,xs), axis=0)
            y  = np.concatenate((y, ys), axis=0)
            print("Loading data from {}: {}".format(file_path, len(xs)))

        if self.use_side_cam:
            file_paths_left = glob.glob(os.path.join(self.data_dir_train,'*/left.csv'))
            
            for file_path_left in sorted(file_paths_left):
                path_prefix_left = os.path.dirname(file_path_left)
                # load label
                data_left  = pd.read_csv(file_path_left)
                img_names_left = data_left['filename'].values # relative path
                point_names_left = data_left['point_filename'].values
                assert len(img_names_left) == len(point_names_left)
                
                # combine sequential image path and point path
                xs_left = [] # (Ni,T,3)
                for i in range(len(img_names_left)):
                    if i < (self.seq_len-1):
                        continue
                    xt_left = [] # (T,3)
                    for t in reversed(range(self.seq_len)):
                        xt_left.append([os.path.join(path_prefix_left, img_names_left[i-t]), 
                                os.path.join(path_prefix_left, 'points_bin', point_names_left[i-t][:-3]+'bin'),
                                -1.0])  
                    xs_left.append(xt_left)
                
                # scale label
                angle_left = data_left['angle'].values[self.seq_len-1:]  # n-(self.seq_len-1)
                speed_left = data_left['speed'].values[self.seq_len-1:]
                angle_left_adj = self.__camera_adjust(angle_left, speed_left, camera='left')

                angle_left_s = self.scale_label(angle_left_adj, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
                speed_left_s = self.scale_label(speed_left, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
                ys_left = np.stack([angle_left_s, speed_left_s], axis=1)
                
                # concatenate all data
                assert len(xs_left) == len(ys_left) == (len(img_names_left)-self.seq_len+1)
                Xs = np.concatenate((Xs,xs_left), axis=0)
                y  = np.concatenate((y, ys_left), axis=0)
                print("Loading data from {}: {}".format(file_path_left, len(xs_left)))
            
            ## Load right camera data
            file_paths_right = glob.glob(os.path.join(self.data_dir_train,'*/right.csv'))
            
            for file_path_right in sorted(file_paths_right):
                path_prefix_right = os.path.dirname(file_path_right)
                # load label
                data_right  = pd.read_csv(file_path_right)
                img_names_right = data_right['filename'].values # relative path
                point_names_right = data_right['point_filename'].values
                assert len(img_names_right) == len(point_names_right)
                
                # combine sequential image path and point path
                xs_right = [] # (Ni,T,2)
                for i in range(len(img_names_right)):
                    if i < (self.seq_len-1):
                        continue
                    xt_right = [] # (T,2)
                    for t in reversed(range(self.seq_len)):
                        xt_right.append([os.path.join(path_prefix_right, img_names_right[i-t]), 
                                os.path.join(path_prefix_right, 'points_bin', point_names_right[i-t][:-3]+'bin'),
                                1.0])  
                    xs_right.append(xt_right)
                
                # scale label
                angle_right = data_right['angle'].values[self.seq_len-1:]
                speed_right = data_right['speed'].values[self.seq_len-1:]
                angle_right_adj = self.__camera_adjust(angle_right, speed_right, camera='right')

                angle_right_s = self.scale_label(angle_right_adj, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
                speed_right_s = self.scale_label(speed_right, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
                ys_right = np.stack([angle_right_s, speed_right_s], axis=1)
                
                # concatenate all data
                assert len(xs_right) == len(ys_right) == (len(img_names_right)-self.seq_len+1)
                Xs = np.concatenate((Xs,xs_right), axis=0)
                y  = np.concatenate((y, ys_right), axis=0)
                print("Loading data from {}: {}".format(file_path_right, len(xs_right)))

        if self.balance_angle or self.balance_speed:
            Xs, y = self.balance_data(Xs, y, 
                                      self.balance_angle, 
                                      self.balance_speed,
                                      bin_count=20,
                                      fix_times=1)
        # visualize label distribution
        #self.label_distribution(y)

        # split data
        self.Xs_train, self.Xs_val, self.y_train, self.y_val = train_test_split(Xs, y, test_size=self.val_ratio, random_state=10, shuffle=True)

        self.num_train = len(self.Xs_train)
        self.num_val = len(self.y_val)
        print("Train set: {}; Val set: {}".format(self.num_train, self.num_val))

class ImageWithTSDataset(Dataset):

    '''
    image data with timestamp
    '''

    def __init__(self, data_root, transform=None, target_transform=None, seq_len=6, device='cuda'):

        self.device = device
        self.seq_len = seq_len
        self.img_dir = data_root
        self.target_transform = target_transform
        self.transform = transform


        self.___load_left_right_image()

    def __len__(self):
        return len(self.labels) - self.seq_len

    def __getitem__(self, idx):
        img_path = self.Xs[idx]
        image_left = []
        image_right = []
        # print(img_path[0])
        for path_ in img_path:

            assert len(path_) == 2
            image_left.append(Image.open(path_[0]))
            image_right.append(Image.open(path_[1]))
            # for _img_path in path_:
            #     image.append(torch.tensor(read_image(_img_path), dtype=torch.float))
                # show = ToPILImage()
                # show(read_image(_img_path)).show()

        label = self.labels[idx:idx+len(img_path)]
        if self.transform:
            for i in range(len(image_left)):
                image_left[i] = self.transform(image_left[i])
            for i in range(len(image_right)):
                image_right[i] = self.transform(image_right[i])
        if self.target_transform:
            label = self.target_transform(label)
        
        
        return torch.stack(image_left).to(self.device), torch.stack(image_right).to(self.device), torch.tensor(label, dtype=torch.float).to(self.device)


    def ___load_left_right_image(self):
        
        x = []                  # (N, 2)
        y = np.zeros((0, 2))    # (N,2)
        for data_dir_train in self.img_dir:

            # Loading from all csv files;  * is HMB
            left_path = data_dir_train + '/left.csv'
            right_path = data_dir_train + '/right.csv'
            
            path_prefix = os.path.dirname(left_path)
            
            # load label
            left_data  = pd.read_csv(left_path)
            right_data  = pd.read_csv(right_path)
            
            left_img_names = left_data['filename'].values
            right_img_names = right_data['filename'].values

            assert len(left_img_names) == len(right_img_names)
        
            # combine sequential image path and point path
            xs = [] # (Ni,T,2)
            for i in range(len(left_img_names)):
                if i < (self.seq_len-1):
                    continue
                xt = [] # (T,2)
                for t in reversed(range(self.seq_len)):
                    xt.append([os.path.join(path_prefix, left_img_names[i-t]), os.path.join(path_prefix, right_img_names[i-t])])
                
                xs.append(xt)
        
            # scale label
            angle = left_data['angle'].values[self.seq_len-1:]  # n-(self.seq_len-1)
            speed = left_data['speed'].values[self.seq_len-1:]
            angle_s = self.scale_label(angle, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
            speed_s = self.scale_label(speed, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
            ys = np.stack([angle_s, speed_s], axis=1)
            
            y = np.vstack((y, ys))
            x.extend(xs)
            # print('{},{},{}'.format(len(xs), len(ys), (len(left_img_names)-self.seq_len+1)))
            # concatenate all data
            assert len(xs) == len(ys) == (len(left_img_names)-self.seq_len+1)
            
            # 
            print("Loading data from {}, {}: {}".format(left_path, right_path, len(xs)))

        assert len(y) == len(x)
        self.labels = y;
        self.Xs = x;

        pass
    
    def scale_label(self, y, y_min, y_max, a, b):
        """
        Sacle labels to a fixed interval [a,b]
        """
        return a + (b - a) * (y - y_min) / (y_max - y_min)
    
    def __load_train(self):
        """ 
        Load image paths & point paths and labels from multiple csv files      
        Return:
          - self.Xs: saves paths of sequential data: 
                    [[[i1,p1, flag], ...[it,pt,flag]],
                      [[],[]],  
                    ]; size is (N, T, 3)
            FLAG: left: -1; center: 0; right: 1
          - self.y: saves scaled labels:
                   [[angle, speed],
                    [,],
                   ]; size si (N,2)
        """
        # Loading from all csv files;  * is HMB
        file_paths = glob.glob(os.path.join(self.data_dir_train, '*/center.csv'))
        Xs = np.zeros((0, self.seq_len, 3)) # (N,T,3)
        y = np.zeros((0, 2))  # (N,2)

        for file_path in sorted(file_paths):
            path_prefix = os.path.dirname(file_path)
            # load label
            data  = pd.read_csv(file_path)
            img_names = data['filename'].values # relative path
            point_names = data['point_filename'].values
            assert len(img_names) == len(point_names)
            
            # combine sequential image path and point path
            xs = [] # (Ni,T,2)
            for i in range(len(img_names)):
                if i < (self.seq_len-1):
                    continue
                xt = [] # (T,2)
                for t in reversed(range(self.seq_len)):
                    xt.append([os.path.join(path_prefix, img_names[i-t]), 
                               os.path.join(path_prefix, 'points_bin', point_names[i-t][:-3]+'bin'),
                               0.0]) # CAM_FLAG=0 
                xs.append(xt)
            
            # scale label
            angle = data['angle'].values[self.seq_len-1:]  # n-(self.seq_len-1)
            speed = data['speed'].values[self.seq_len-1:]
            angle_s = self.scale_label(angle, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
            speed_s = self.scale_label(speed, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
            ys = np.stack([angle_s, speed_s], axis=1)
            
            # concatenate all data
            assert len(xs) == len(ys) == (len(img_names)-self.seq_len+1)
            Xs = np.concatenate((Xs,xs), axis=0)
            y  = np.concatenate((y, ys), axis=0)
            print("Loading data from {}: {}".format(file_path, len(xs)))

        if self.use_side_cam:
            file_paths_left = glob.glob(os.path.join(self.data_dir_train,'*/left.csv'))
            
            for file_path_left in sorted(file_paths_left):
                path_prefix_left = os.path.dirname(file_path_left)
                # load label
                data_left  = pd.read_csv(file_path_left)
                img_names_left = data_left['filename'].values # relative path
                point_names_left = data_left['point_filename'].values
                assert len(img_names_left) == len(point_names_left)
                
                # combine sequential image path and point path
                xs_left = [] # (Ni,T,3)
                for i in range(len(img_names_left)):
                    if i < (self.seq_len-1):
                        continue
                    xt_left = [] # (T,3)
                    for t in reversed(range(self.seq_len)):
                        xt_left.append([os.path.join(path_prefix_left, img_names_left[i-t]), 
                                os.path.join(path_prefix_left, 'points_bin', point_names_left[i-t][:-3]+'bin'),
                                -1.0])  
                    xs_left.append(xt_left)
                
                # scale label
                angle_left = data_left['angle'].values[self.seq_len-1:]  # n-(self.seq_len-1)
                speed_left = data_left['speed'].values[self.seq_len-1:]
                angle_left_adj = self.__camera_adjust(angle_left, speed_left, camera='left')

                angle_left_s = self.scale_label(angle_left_adj, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
                speed_left_s = self.scale_label(speed_left, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
                ys_left = np.stack([angle_left_s, speed_left_s], axis=1)
                
                # concatenate all data
                assert len(xs_left) == len(ys_left) == (len(img_names_left)-self.seq_len+1)
                Xs = np.concatenate((Xs,xs_left), axis=0)
                y  = np.concatenate((y, ys_left), axis=0)
                print("Loading data from {}: {}".format(file_path_left, len(xs_left)))
            
            ## Load right camera data
            file_paths_right = glob.glob(os.path.join(self.data_dir_train,'*/right.csv'))
            
            for file_path_right in sorted(file_paths_right):
                path_prefix_right = os.path.dirname(file_path_right)
                # load label
                data_right  = pd.read_csv(file_path_right)
                img_names_right = data_right['filename'].values # relative path
                point_names_right = data_right['point_filename'].values
                assert len(img_names_right) == len(point_names_right)
                
                # combine sequential image path and point path
                xs_right = [] # (Ni,T,2)
                for i in range(len(img_names_right)):
                    if i < (self.seq_len-1):
                        continue
                    xt_right = [] # (T,2)
                    for t in reversed(range(self.seq_len)):
                        xt_right.append([os.path.join(path_prefix_right, img_names_right[i-t]), 
                                os.path.join(path_prefix_right, 'points_bin', point_names_right[i-t][:-3]+'bin'),
                                1.0])  
                    xs_right.append(xt_right)
                
                # scale label
                angle_right = data_right['angle'].values[self.seq_len-1:]
                speed_right = data_right['speed'].values[self.seq_len-1:]
                angle_right_adj = self.__camera_adjust(angle_right, speed_right, camera='right')

                angle_right_s = self.scale_label(angle_right_adj, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
                speed_right_s = self.scale_label(speed_right, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
                ys_right = np.stack([angle_right_s, speed_right_s], axis=1)
                
                # concatenate all data
                assert len(xs_right) == len(ys_right) == (len(img_names_right)-self.seq_len+1)
                Xs = np.concatenate((Xs,xs_right), axis=0)
                y  = np.concatenate((y, ys_right), axis=0)
                print("Loading data from {}: {}".format(file_path_right, len(xs_right)))

        if self.balance_angle or self.balance_speed:
            Xs, y = self.balance_data(Xs, y, 
                                      self.balance_angle, 
                                      self.balance_speed,
                                      bin_count=20,
                                      fix_times=1)
        # visualize label distribution
        #self.label_distribution(y)

        # split data
        self.Xs_train, self.Xs_val, self.y_train, self.y_val = train_test_split(Xs, y, test_size=self.val_ratio, random_state=10, shuffle=True)

        self.num_train = len(self.Xs_train)
        self.num_val = len(self.y_val)
        print("Train set: {}; Val set: {}".format(self.num_train, self.num_val))

class OpticalFlowDataset(Dataset):

    def __init__(self, data_root, transform=None, target_transform=None, seq_len=6, device='cuda'):

        self.device = device
        self.seq_len = seq_len
        self.img_dir = data_root
        self.target_transform = target_transform
        self.transform = transform

        self.cache = []
        self.___load_image()

    def __len__(self):
        return len(self.labels) - self.seq_len - 1 # 连续两帧出一帧光流

    def __getitem__(self, idx):

        optical = []
        
        img_path = self.Xs[idx]
        image_center = []

        if idx >= 1:
            optical = self.cache[1:]

            image_center.append(Image.open(img_path[-2]).convert('L'))
            image_center.append(Image.open(img_path[-1]).convert('L'))
            
            if self.transform:
                for i in range(len(image_center)):
                    image_center[i] = self.transform(image_center[i]).permute(1, 2, 0)

            tmp = cv2.calcOpticalFlowFarneback(np.asarray(image_center[0]), np.asarray(image_center[1]), None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
            optical.append(torch.from_numpy(tmp).permute(2, 0, 1))

            self.cache = optical

        else:

            for path_ in img_path:

                image_center.append(Image.open(path_).convert('L'))

            if self.transform:
                for i in range(len(image_center)):
                    image_center[i] = self.transform(image_center[i]).permute(1, 2, 0)

            for i in range(len(image_center) - 1):
                
                tmp = cv2.calcOpticalFlowFarneback(np.asarray(image_center[i]), np.asarray(image_center[i+1]), None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                optical.append(torch.from_numpy(tmp).permute(2, 0, 1))
            
            

            self.cache = optical

        label = self.labels[idx+1:idx+len(img_path)]
        
        if self.target_transform:
            label = self.target_transform(label)
            
        
        return torch.stack(optical).to(self.device), torch.tensor(label, dtype=torch.float).to(self.device)


    def ___load_image(self):
        
        x = []                  # (N, 2)
        y = np.zeros((0, 2))    # (N, 2)
        for data_dir_train in self.img_dir:

            # Loading from all csv files;  * is HMB
            center_path = data_dir_train + '/center.csv'
            
            path_prefix = os.path.dirname(center_path)
            
            # load label
            center_data  = pd.read_csv(center_path)
            
            center_img_names = center_data['filename'].values
            xs = []
            for i in range(len(center_img_names) - 1):
                if i < (self.seq_len-1):
                    continue
                xt = [] # (T,2)
                for t in reversed(range(self.seq_len + 1)):
                    xt.append(os.path.join(path_prefix, center_img_names[i-t]))
                
                xs.append(xt)
        
            # scale label
            angle = center_data['angle'].values[self.seq_len:]
            speed = center_data['speed'].values[self.seq_len:]
            angle_s = self.scale_label(angle, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
            speed_s = self.scale_label(speed, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
            ys = np.stack([angle_s, speed_s], axis=1)
            
            y = np.vstack((y, ys))
            x.extend(xs)
            # print('{},{},{}'.format(len(xs), len(ys), (len(left_img_names)-self.seq_len+1)))
            # concatenate all data
            assert len(xs) == len(ys) == (len(center_img_names)-self.seq_len)
            
            # 
            print("Loading data from {}: {}".format(center_path, len(xs)))

        assert len(y) == len(x)
        self.labels = y;
        self.Xs = x;

        pass

    def scale_label(self, y, y_min, y_max, a, b):
        """
        Sacle labels to a fixed interval [a,b]
        """
        return a + (b - a) * (y - y_min) / (y_max - y_min)
    
    pass

if __name__ == '__main__':
    import argparse
    import time 
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToPILImage

    show = ToPILImage()
    ## import open3d as o3d
    ## from tools.cloud_visualizer import Visualizer

    ## ===  Evaluation Procedure === ## 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data/Udacity/Ch2_002/HMB_1/', help='dataset path')

    Flags = parser.parse_args()
    dataset = ImageDataset(Flags.data_root, transform=None, target_transform=None, load_flag=['left', 'right'], seq_len=5)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # dataloader = DataLoader(Flags.data_root, height=200, width=200, seq_len = 5)
    # print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
    # print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
    # print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)

    # Display image and label.
    train_features, train_labels = next(iter(dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    
    img = train_features[1].squeeze()
    print(img.shape)
    label = train_labels[0]
    show(img).show()
    # plt.imshow(img)
    # plt.show()
    print(f"Label: {label}")

     
    X_image_batch, _, y_batch = dataloader.load_train_batch(batch_size)
    print(X_image_batch.shape, y_batch.shape)

    angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
    speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)

    for i in range(batch_size):
        print('Angle: ', angle[i])
        print('Speed: ', speed[i])
        im = X_image_batch[i,:,:,0]
        #im = np.asarray(im, np.uint8)
        flow = X_image_batch[i,:,:,1:3]*5
    
        fig = plt.figure()
        plt.imshow(dataloader.draw_flow(im, flow))
        plt.show()
'''
    elif Flags.test_mode == '3':
        dataloader = DataLoader(Flags.data_root, input_cfg='GRAYF-T', 
                                height=200, width=200, 
                                seq_len = 5,
                                num_point= None,
                                use_side_cam=True)
        print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
        print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
        print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)
        
        batch_size = 16
        seq_len = 5
        X_image_batch, _, y_batch = dataloader.load_train_batch(batch_size)
        print(X_image_batch.shape, y_batch.shape)

        angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
        speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)

        for i in range(batch_size):
            print('Angle: ', angle[i])
            print('Speed: ', speed[i])
            for t in range(seq_len):    
                im = X_image_batch[i,t,:,:,0]
                flow = X_image_batch[i,t,:,:,1:3]*5
            
                fig = plt.figure()
                plt.imshow(dataloader.draw_flow(im, flow))
                plt.show()

    elif Flags.test_mode == '4':
        dataloader = DataLoader(Flags.data_root, input_cfg='XYZ', 
                                height=200, width=200,
                                seq_len = None,
                                num_point= Flags.num_point, 
                                use_side_cam=True)
        print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
        print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
        print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)
        
        batch_size = 16
        _, X_cloud_batch, y_batch = dataloader.load_val_batch(batch_size)
        print(X_cloud_batch.shape, y_batch.shape)

        angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
        speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)
        
        vis = Visualizer(play_mode='manual')
        for i in range(batch_size):
            print('Angle: ', angle[i])
            print('Speed: ', speed[i])
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(X_cloud_batch[i])
            vis.show([cloud])

    elif Flags.test_mode == '5':
        dataloader = DataLoader(Flags.data_root, input_cfg='XYZF', 
                                height=200, width=200, 
                                seq_len = None,
                                num_point= Flags.num_point,
                                use_side_cam=True)
        print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
        print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
        print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)
        
        batch_size = 16
        for k in range(3): # test batch
            _, X_cloud_batch, y_batch = dataloader.load_val_batch(batch_size)
            print(X_cloud_batch.shape, y_batch.shape)

            angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
            speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)
            
            vis = Visualizer(play_mode='manual')
            for i in range(batch_size):
                print('Angle: ', angle[i])
                print('Speed: ', speed[i])
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(X_cloud_batch[i][:,:3])
                vis.show([cloud])

    elif Flags.test_mode == '6':
        dataloader = DataLoader(Flags.data_root, input_cfg='XYZF-T', 
                                height=200, width=200, 
                                seq_len = 5,
                                num_point= Flags.num_point,
                                use_side_cam=True)
        print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
        print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
        print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)
        batch_size = 16
        seq_len = 5
        _, X_cloud_batch, y_batch = dataloader.load_train_batch(batch_size)
        print(X_cloud_batch.shape, y_batch.shape)

        angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
        speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)
        
        vis = Visualizer(play_mode='manual')
        for i in range(batch_size):
            print('Angle: ', angle[i])
            print('Speed: ', speed[i])
            cloud = o3d.geometry.PointCloud()
            for t in range(seq_len):
                cloud.points = o3d.utility.Vector3dVector(X_cloud_batch[i,t,:,:3])
                vis.show([cloud])

    elif Flags.test_mode == '7':
        dataloader = DataLoader(Flags.data_root, input_cfg='GRAYF-XYZF-T', 
                                height=200, width=200,
                                seq_len = 5,
                                num_point= 10000,
                                aug_cfg=Flags.aug_cfg)
        print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
        print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
        print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)
        
        batch_size = 5
        seq_len = 5
        X_image_batch, X_cloud_batch, y_batch = dataloader.load_train_batch(batch_size)
        print(X_image_batch.shape, X_cloud_batch.shape, y_batch.shape)

        angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
        speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)

        vis = Visualizer(play_mode='manual')
        for i in range(batch_size):
            print('Angle: ', angle[i])
            print('Speed: ', speed[i])
            cloud = o3d.geometry.PointCloud()
            for t in range(seq_len):
                fig = plt.figure()
                plt.imshow(X_image_batch[i,t,:,:,0], cmap='gray')
                plt.show()
                cloud.points = o3d.utility.Vector3dVector(X_cloud_batch[i,t,:,:3])
                vis.show([cloud])

'''
