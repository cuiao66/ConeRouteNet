import json
import numpy as np
import re
import os

root_path = '/root/mnt/DATASET/CARLA/Interfuser/'

def _load_text(path):
    text = open(path, 'r').read()
    return text

def _load_json(path):
    try:
        json_value = json.load(open(path))
    except Exception as e:
        # _logger.info(path)
        n = path[-9:-5]
        new_path = path[:-9] + "%04d.json" % (int(n) - 1)
        json_value = json.load(open(new_path))
    return json_value

if __name__ == '__main__':

    root = '/root/mnt/DATASET/CARLA/Interfuser'
    route_frames = []

    dataset_indexs = _load_text(os.path.join(root, 'dataset_index.txt')).split('\n')
    pattern = re.compile('.*town(\d\d).*_w(\d+)_.*')
    pattern2 = re.compile('weather-(\d+).*town(\d\d).*')
    for line in dataset_indexs:
        if len(line.split()) != 2:
            continue
        path, frames = line.split()
        frames = int(frames)
        res = pattern.findall(path)
        if len(res) != 1:
            res2 = pattern2.findall(path)
            if len(res2) != 1:
                continue
            else:
                weather = int(res2[0][0])
                town = int(res2[0][1])
        else:
            weather = int(res[0][1])
            town = int(res[0][0])

        for i in range(frames):
            route_frames.append((os.path.join(root, path), i))
    count = 0
    for route_dir, frame_id in route_frames:

        measurements = _load_json(
            os.path.join(route_dir, "measurements", "%04d.json" % frame_id)
        )


        if measurements["is_junction"]:
            count = count + 1

    print("is_junction / total frames = {}".format(count / len(route_frames)))