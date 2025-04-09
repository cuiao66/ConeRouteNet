import os


class GlobalConfig:
    """base architecture configurations"""

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40  # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40  # buffer size

    max_throttle = 0.75  # upper limit on throttle signal value in dataset
    brake_speed = 0.1  # desired speed below which brake is triggered
    brake_ratio = 1.1  # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.35  # maximum change in speed input to logitudinal controller

    max_speed = 5
    collision_buffer = [2.5, 1.2]
    project_dir = os.environ['project_dir']
    # model_path = "/root/DemoEnd2EndNet/v1/output/gcvit_loss_modify/model_best.pth.tar"
    model_path = "{}/DemoEnd2EndNet/net/output/20230824-094234-DemoEnd2EndNet_baseline-224-with_command/model_best.pth.tar".format(project_dir)
    momentum = 0
    skip_frames = 1
    detect_threshold = 0.04

    model = "DemoEnd2EndNet_baseline"

    use_admin=False,
    only_encoder=True,

    use_hic=True,

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
