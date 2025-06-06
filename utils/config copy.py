import os

class GlobalConfig:
    """ base architecture configurations """
	# Data
    
    post_len = 6
    pred_len = 6        # future waypoints predicted
    pred_feature_len = 6
    waypoint_len = pred_feature_len
    seq_len = post_len + pred_feature_len         # input timesteps

    data_step = 3
    
    width = 256
    height = 256
    device = 'cuda'
    project_dir = '/home/mct/xm/End2End'
    car_status_len_in = 2

    root_dir = '/root/transfuser-main/data/expert'
    train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town06']## , 'Town07', 'Town10']
    val_towns = ['Town05']
    train_data, val_data = [], []
    for town in train_towns:
        train_data.append(os.path.join(root_dir, town+'_tiny'))
        train_data.append(os.path.join(root_dir, town+'_short'))
    for town in val_towns:
        val_data.append(os.path.join(root_dir, town+'_short'))

    # visualizing transformer attention maps
    viz_root = '/mnt/qb/geiger/kchitta31/data_06_21'
    viz_towns = ['Town05_tiny']
    viz_data = []
    for town in viz_towns:
        viz_data.append(os.path.join(viz_root, town))

    ignore_sides = False # don't consider side cameras
    ignore_rear = True # don't consider rear cameras
    n_views = 1 # no. of camera views

    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    feature_map_c = 16
    feature_map_w = 32
    feature_map_h = 32

    lr = 1e-4 # learning rate

    batch_size = 16

    # Conv Encoder
    vert_anchors = 8
    horz_anchors = 8
    anchors = vert_anchors * horz_anchors

	# GPT Encoder
    n_embd = 512
    block_exp = 4
    n_layer = 8
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

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
