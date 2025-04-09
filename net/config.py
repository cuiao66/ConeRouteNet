import os

class GlobalConfig:
    """ base architecture configurations """
	# Data


    seq_len = 4     # input frame
    imagine_len = 4
    pred_len = imagine_len    # waypoint to pred

    data_step = 5

    train_seg = False


    width = 256
    height = 256
    device = 'cuda:2'
    project_dir = '/root/End2End'
    car_status_len_in = 2

    root_dir = '/root/mnt/DATASET/CARLA/db_extra'
    # train_towns = ['Town01', 'Town02']##, 'Town03', 'Town04' , 'Town06'], 'Town07', 'Town10']
    # val_towns = ['Town05']
    train_data, val_data = [], []
    # for town in train_towns:
    #     train_data.append(os.path.join(root_dir, town+'_tiny'))
    #     train_data.append(os.path.join(root_dir, town+'_short'))
    # for town in val_towns:
    #     val_data.append(os.path.join(root_dir, town+'_short'))
    train_data.append(os.path.join(root_dir, 'Town02'+'_tiny'))
    train_data.append(os.path.join(root_dir, 'Town03'+'_tiny'))
    val_data.append(os.path.join(root_dir, 'Town02'+'_short'))
    val_data.append(os.path.join(root_dir, 'Town03'+'_short'))

    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    feature_map_c = 16
    feature_map_w = 32
    feature_map_h = 32

    lr = 1e-4   # learning rate

    batch_size = 16
    multi_gpus = [2, 3]

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40 # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40 # buffer size

    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.1 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
