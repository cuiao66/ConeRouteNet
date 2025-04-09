import os
import json
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import torchvision.transforms as transforms

img_pre_process = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

class CARLA_Data(Dataset):

    def __init__(self, root, config):

        self.seq_len = config.seq_len
        self.ignore_sides = config.ignore_sides
        self.ignore_rear = config.ignore_rear

        self.input_resolution = config.input_resolution
        self.scale = config.scale

        self.data_step = config.data_step

        self.num_points = config.num_points

        self.lidar = []
        self.front = []
        self.tel = []
        self.seg_front = []
        self.seg_right = []
        self.seg_tel = []
        self.left = []
        self.seg_left = []
        self.right = []
        self.rear = []
        self.x = []
        self.y = []
        self.x_command = []
        self.y_command = []
        self.theta = []
        self.steer = []
        self.throttle = []
        self.brake = []
        self.command = []
        self.velocity = []

        self.min_x = config.min_x
        self.max_x = config.max_x
        self.max_y = config.max_y
        self.min_y = config.min_y
        
        for sub_root in tqdm(root, file=sys.stdout):
            preload_file = os.path.join(sub_root, 'rg_lidar_diag_pl_'+str(self.seq_len)+'.npy')

            # dump to npy if no preload
            if not os.path.exists(preload_file):
                preload_front = []
                preload_tel = []
                preload_tel_seg = []
                preload_front_seg = []
                preload_left = []
                preload_left_seg = []
                preload_right = []
                preload_right_seg = []
                preload_rear = []
                preload_lidar = []
                preload_x = []
                preload_y = []
                preload_x_command = []
                preload_y_command = []
                preload_theta = []
                preload_steer = []
                preload_throttle = []
                preload_brake = []
                preload_command = []
                preload_velocity = []

                # list sub-directories in root 
                root_files = os.listdir(sub_root)
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    ## print(route_dir)
                    # subtract final frames (pred_len) since there are no future waypoints
                    # first frame of sequence not used
                    
                    num_seq = len(os.listdir(route_dir+"/rgb_front/")) - self.seq_len - 1
                    
                    for seq in range(0, num_seq, config.data_step):
                        fronts = []
                        seg_fronts = []
                        lefts = []
                        seg_lefts = []
                        rights = []
                        seg_rights = []
                        rears = []
                        seg_tels = []
                        tels = []
                        lidars = []
                        xs = []
                        ys = []
                        thetas = []

                        # read files sequentially (past and current frames)
                        for i in range(self.seq_len):
                            # images
                            filename = f"{str(seq + 1 + i).zfill(4)}.png"
                            fronts.append(route_dir+"/rgb_front/"+filename)
                            lefts.append(route_dir+"/rgb_left/"+filename)
                            rights.append(route_dir+"/rgb_right/"+filename)
                            rears.append(route_dir+"/rgb_rear/"+filename)
                            tels.append(route_dir+"/rgb_tel/"+filename)
                            
                            seg_fronts.append(route_dir+"/seg_front/"+filename)
                            seg_rights.append(route_dir+"/seg_right/"+filename)
                            seg_lefts.append(route_dir+"/seg_left/"+filename)
                            seg_tels.append(route_dir+"/seg_tel/"+filename)

                            # point cloud
                            lidars.append(route_dir + f"/lidar/{str(seq + 1 + i).zfill(4)}.npy")
                            
                            # position
                            with open(route_dir + f"/measurements/{str(seq + 1 + i).zfill(4)}.json", "r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])
                            thetas.append(data['theta'])

                        with open(route_dir + f"/measurements/{str(seq + 1 + i).zfill(4)}.json", "r") as read_file:
                                data = json.load(read_file)
                        # get control value of final frame in sequence
                        preload_x_command.append(data['x_command'])
                        preload_y_command.append(data['y_command'])
                        preload_steer.append(data['steer'])
                        preload_throttle.append(data['throttle'])
                        preload_brake.append(data['brake'])
                        preload_command.append(data['command'])
                        preload_velocity.append(data['speed'])

                        # # read files sequentially (future frames)
                        # for i in range(self.seq_len, self.seq_len + self.pred_len):
                        #     # point cloud
                        #     lidars.append(route_dir + f"/lidar/{str(seq*self.seq_len+1+i).zfill(4)}.npy")
                            
                        #     # position
                        #     with open(route_dir + f"/measurements/{str(seq*self.seq_len+1+i).zfill(4)}.json", "r") as read_file:
                        #         data = json.load(read_file)
                        #     xs.append(data['x'])
                        #     ys.append(data['y'])

                        #     # fix for theta=nan in some measurements
                        #     if np.isnan(data['theta']):
                        #         thetas.append(0)
                        #     else:
                        #         thetas.append(data['theta'])

                        preload_front.append(fronts)
                        preload_tel.append(tels)
                        preload_front_seg.append(seg_fronts)
                        preload_tel_seg.append(seg_tels)
                        preload_left_seg.append(seg_lefts)
                        preload_right_seg.append(seg_rights)
                        preload_left.append(lefts)
                        preload_right.append(rights)
                        preload_rear.append(rears)
                        preload_lidar.append(lidars)
                        preload_x.append(xs)
                        preload_y.append(ys)
                        preload_theta.append(thetas)

                # dump to npy
                preload_dict = {}
                preload_dict['front'] = preload_front
                preload_dict['tel'] = preload_tel
                preload_dict['seg_front'] = preload_front_seg
                preload_dict['seg_tel'] = preload_tel_seg
                preload_dict['left'] = preload_left
                preload_dict['seg_left'] = preload_left_seg
                preload_dict['right'] = preload_right
                preload_dict['seg_right'] = preload_right_seg
                preload_dict['rear'] = preload_rear
                preload_dict['lidar'] = preload_lidar
                preload_dict['x'] = preload_x
                preload_dict['y'] = preload_y
                preload_dict['x_command'] = preload_x_command
                preload_dict['y_command'] = preload_y_command
                preload_dict['theta'] = preload_theta
                preload_dict['steer'] = preload_steer
                preload_dict['throttle'] = preload_throttle
                preload_dict['brake'] = preload_brake
                preload_dict['command'] = preload_command
                preload_dict['velocity'] = preload_velocity
                np.save(preload_file, preload_dict)

            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)

            # assert len(preload_dict.item()['front']) == len(preload_dict.item()['seg_right'])
            
            self.front += preload_dict.item()['front']
            self.seg_front += preload_dict.item()['seg_front']
            self.tel += preload_dict.item()['tel']
            self.seg_tel += preload_dict.item()['seg_tel']
            self.left += preload_dict.item()['left']
            self.seg_left += preload_dict.item()['seg_left']
            self.right += preload_dict.item()['right']
            self.seg_right += preload_dict.item()['seg_right']
            self.rear += preload_dict.item()['rear']
            self.lidar += preload_dict.item()['lidar']
            self.x += preload_dict.item()['x']
            self.y += preload_dict.item()['y']
            self.x_command += preload_dict.item()['x_command']
            self.y_command += preload_dict.item()['y_command']
            self.theta += preload_dict.item()['theta']
            self.steer += preload_dict.item()['steer']
            self.throttle += preload_dict.item()['throttle']
            self.brake += preload_dict.item()['brake']
            self.command += preload_dict.item()['command']
            self.velocity += preload_dict.item()['velocity']
            print("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

    def __len__(self):
        """Returns the length of the dataset. """
                    
        return len(self.front)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['tels'] = []
        data['seg_fronts'] = []
        data['seg_tels'] = []
        data['lefts'] = []
        data['seg_lefts'] = []
        data['rights'] = []
        data['seg_rights'] = []
        data['rears'] = []
        data['lidars'] = []

        seq_fronts = self.front[index]
        seq_seg_fronts = self.seg_front[index]
        seq_tels = self.tel[index]
        seq_seg_tels = self.seg_tel[index]
        seq_lefts = self.left[index]
        seq_seg_lefts = self.seg_left[index]
        seq_rights = self.right[index]
        seq_seg_rights = self.seg_right[index]
        seq_rears = self.rear[index]
        seq_lidars = self.lidar[index]
        seq_x = self.x[index]
        seq_y = self.y[index]
        seq_theta = self.theta[index]

        full_lidar = []
        pos = []
        neg = []
        # print(seq_fronts[0])
        for i in range(self.seq_len):
            data['fronts'].append(img_pre_process(torch.from_numpy(np.array(scale_and_crop_image(
                                Image.open(seq_fronts[i]), scale=self.scale, crop=self.input_resolution), 
                                dtype=np.float32))/255.))
            data['seg_fronts'].append(filter_sem(np.array(
                scale_and_crop_image(Image.open(seq_seg_fronts[i]), scale=self.scale, crop=self.input_resolution))))
            
            data['tels'].append(img_pre_process(torch.from_numpy(np.array(scale_and_crop_image(
                                Image.open(seq_tels[i]), scale=self.scale, crop=self.input_resolution), 
                                dtype=np.float32))/255.))
            data['seg_tels'].append(filter_sem(np.array(
                scale_and_crop_image(Image.open(seq_seg_tels[i]), scale=self.scale, crop=self.input_resolution))))
            
            if not self.ignore_sides:
                data['lefts'].append(img_pre_process(torch.from_numpy(np.array(scale_and_crop_image(
                                Image.open(seq_lefts[i]), scale=self.scale, crop=self.input_resolution), 
                                dtype=np.float32))/255.))
                data['seg_lefts'].append(filter_sem(np.array(
                    scale_and_crop_image(Image.open(seq_seg_lefts[i]), scale=self.scale, crop=self.input_resolution))))
                
                data['rights'].append(img_pre_process(torch.from_numpy(np.array(scale_and_crop_image(
                                Image.open(seq_rights[i]), scale=self.scale, crop=self.input_resolution), 
                                dtype=np.float32))/255.))
                data['seg_rights'].append(filter_sem(np.array(
                    scale_and_crop_image(Image.open(seq_seg_rights[i]), scale=self.scale, crop=self.input_resolution))))
            if not self.ignore_rear:
                data['rears'].append(img_pre_process(torch.from_numpy(np.array(scale_and_crop_image(
                                Image.open(seq_rears[i]), scale=self.scale, crop=self.input_resolution), 
                                dtype=np.float32))/255.))
            
        
            # fix for theta=nan in some measurements
            if np.isnan(seq_theta[i]):
                seq_theta[i] = 0.

        ego_x = seq_x[i]
        ego_y = seq_y[i]
        ego_theta = seq_theta[i]

        # future frames
        for i in range(self.seq_len):
            lidar_unprocessed = np.load(seq_lidars[i])
            full_lidar.append(lidar_unprocessed)

        # lidar and waypoint processing to local coordinates
        waypoints = []
        for i in range(self.seq_len):
            # waypoint is the transformed version of the origin in local coordinates
            # we use 90-theta instead of theta
            # LBC code uses 90+theta, but x is to the right and y is downwards here
            local_waypoint = transform_2d_points(np.zeros((1,3)), 
                np.pi/2-seq_theta[i], -seq_x[i], -seq_y[i], np.pi/2-ego_theta, -ego_x, -ego_y)
            waypoints.append(tuple(local_waypoint[0,:2]))

            # # process only past lidar point clouds
            # if i < self.seq_len:
            #     # convert coordinate frame of point cloud
            #     full_lidar[i][:,1] *= -1 # inverts x, y
            #     full_lidar[i] = transform_2d_points(full_lidar[i], 
            #         np.pi/2-seq_theta[i], -seq_x[i], -seq_y[i], np.pi/2-ego_theta, -ego_x, -ego_y)
            #     lidar_processed = lidar_to_histogram_features(full_lidar[i], crop=self.input_resolution)
            #     data['lidars'].append(lidar_processed)

            keep = (full_lidar[i][:, 0] >= self.min_x) & (full_lidar[i][:, 0] < self.max_x) & \
            (full_lidar[i][:, 1] >= self.min_y) & (full_lidar[i][:, 1] < self.max_y)
            points = full_lidar[i][keep, :]

            point_idxs = np.arange(len(points))
            if len(points) >= self.num_points:
                selected_point_idxs = np.random.choice(point_idxs, self.num_points, replace=False)
            else:
                selected_point_idxs = np.random.choice(point_idxs, self.num_points, replace=True)
            
            points = points[selected_point_idxs, :]
            data['lidars'].append(points)

        # data['waypoints'] = waypoints

        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta)
        R = np.array([
            [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
            [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
            ])
        local_command_point = np.array([self.x_command[index]-ego_x, self.y_command[index]-ego_y])
        local_command_point = R.T.dot(local_command_point)
        data['target_point'] = tuple(local_command_point)

        data['steer'] = self.steer[index]
        data['throttle'] = self.throttle[index]
        data['brake'] = self.brake[index]
        data['command'] = self.command[index]
        data['velocity'] = self.velocity[index]
        
        return data

def filter_sem(sem, labels=[4,6,10,18]):
    # back ground is 1
    resem = np.ones_like(sem)
    for i, label in enumerate(labels):
        resem[sem==label] = i+2
    return resem

def lidar_to_histogram_features(lidar, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 16
        y_meters_max = 32
        xbins = np.linspace(-2*x_meters_max, 2*x_meters_max+1, 2*x_meters_max*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, 0, y_meters_max*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[...,:2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[...,2]<=-2.0]
    above = lidar[lidar[...,2]>-2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([below_features, above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features


def scale_and_crop_image(image, scale=1, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    mode = image.mode 
    image = np.asarray(im_resized)
    start_x = height//2 - crop//2
    start_y = width//2 - crop//2
    cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
    if mode != 'L':
        cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    
    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out