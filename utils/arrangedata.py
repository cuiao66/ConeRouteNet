
import json
import numpy as np
import os
import lmdb
import cv2
import tqdm

names = ['Town01_tiny', 'Town02_tiny', 'Town03_tiny', 'Town04_tiny', 'Town06_tiny', 'Town07_tiny',
            'Town01_short', 'Town02_short', 'Town03_short', 'Town04_short', 'Town05_short', 'Town06_short']

dataset_path = '/root/End2End/data/extra'
dataset_dst_path = '/root/mnt/DATASET/CARLA/db_extra'

rgb_item = ['rgb_tel', 'rgb_front', 'rgb_left', 'rgb_right']
seg_item = ['seg_tel', 'seg_front', 'seg_left', 'seg_right']
# ['measurements', 'lidar']
for name in names:
    routes = os.listdir(os.path.join(dataset_path, name))
    os.mkdir(os.path.join(dataset_dst_path, name))
    for route in tqdm.tqdm(routes):
        if route[-5:] != '.json':
            route_path = os.path.join(dataset_path, name, route)
            dst_path = os.path.join(dataset_dst_path, name, route)
            # os.mkdir(dst_path)
            db = lmdb.open(dst_path, map_size=1099511627776)
            txn = db.begin(write=True)

            p = os.path.join(route_path, 'rgb_tel')
            files = os.listdir(p)
            txn.put('len'.encode(), np.array(len(files), dtype=np.int8))
            txn.put('Town'.encode(), name[:6].encode())

            for item in rgb_item:
                p = os.path.join(route_path, item)
                files = os.listdir(p)
                for file in files:
                    _p = os.path.join(p, file)
                    img = cv2.imread(_p, cv2.IMREAD_COLOR)
                    _, img_byte = cv2.imencode('.png', img)
                    txn.put('{}_{}'.format(item, file[0:4]).encode(), img_byte)

                    pass
            
            for item in seg_item:
                p = os.path.join(route_path, item)
                files = os.listdir(p)
                for file in files:
                    _p = os.path.join(p, file)
                    img = cv2.imread(_p, cv2.IMREAD_GRAYSCALE)
                    _, img_byte = cv2.imencode('.png', img)
                    txn.put('{}_{}'.format(item, file[0:4]).encode(), img_byte)

                    pass
            
            p = os.path.join(route_path, 'measurements')
            files = os.listdir(p)
            for file in files:
                _p = os.path.join(p, file)
                
                with open(_p) as f:
                    txn.put('measurements_{}'.format(file[0:4]).encode(),  f.read().encode())

            
            p = os.path.join(route_path, 'lidar')
            files = os.listdir(p)
            for file in files:
                _p = os.path.join(p, file)
                
                lidar = np.load(_p)
                txn.put('lidar_{}'.format(file[0:4]).encode(), lidar)
                

            txn.commit()
            db.close()