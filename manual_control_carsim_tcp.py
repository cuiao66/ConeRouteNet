#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import time
import os
import datetime
import pathlib
import json
import copy

import glob
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
from queue import Queue
from queue import Empty
import shutil

try:
    sys.path.append(glob.glob(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import threading

lock = threading.Lock()
b_stop = False
b_collision = False

class GenericMeasurement(object):
    def __init__(self, data, frame):
        self.data = data
        self.frame = frame

class CallBack(object):
    def __init__(self, tag, sensor_type, sensor, data_provider):
        self._tag = tag
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor_type, sensor)

    def __call__(self, data):
        if isinstance(data, carla.libcarla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        elif isinstance(data, GenericMeasurement):
            self._parse_pseudosensor(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        self._data_provider.update_sensor(tag, array, image.frame)

    def _parse_lidar_cb(self, lidar_data, tag):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_radar_cb(self, radar_data, tag):
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self._data_provider.update_sensor(tag, points, radar_data.frame)

    def _parse_gnss_cb(self, gnss_data, tag):
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data.frame)

    def _parse_imu_cb(self, imu_data, tag):
        array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                         ], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, imu_data.frame)

    def _parse_pseudosensor(self, package, tag):
        self._data_provider.update_sensor(tag, package.data, package.frame)


class SensorInterface(object):
    def __init__(self):
        self._sensors_objects = {}
        self._data_buffers = {}
        self._new_data_buffers = Queue()
        self._queue_timeout = 10

        # Only sensor that doesn't get the data on tick, needs special treatment
        self._opendrive_tag = None


    def register_sensor(self, tag, sensor_type, sensor):
        if tag in self._sensors_objects:
            raise Exception("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor

        if sensor_type == 'sensor.opendrive_map': 
            self._opendrive_tag = tag

    def update_sensor(self, tag, data, timestamp):
        if tag not in self._sensors_objects:
            raise Exception("The sensor with tag [{}] has not been created!".format(tag))

        self._new_data_buffers.put((tag, timestamp, data))

    def get_data(self):
        try: 
            data_dict = {}
            while len(data_dict.keys()) < len(self._sensors_objects.keys()):

                # Don't wait for the opendrive sensor
                if self._opendrive_tag and self._opendrive_tag not in data_dict.keys() \
                        and len(self._sensors_objects.keys()) == len(data_dict.keys()) + 1:
                    break

                sensor_data = self._new_data_buffers.get(True, self._queue_timeout)
                data_dict[sensor_data[0]] = ((sensor_data[1], sensor_data[2]))

        except Empty:
            raise Exception("A sensor took too long to send their data")

        return data_dict


def save_handle(dataSetGen, _world):

    global b_save_flag
    global b_stop
    global lock
    while not b_stop:
    
        lock.acquire()
        try:
            data_buf = dataSetGen.sensor_interface.get_data()
            if len(data_buf) == 0 or data_buf['rgb_front'][0] % 5 != 0:
                lock.release()
                continue
        except:
            lock.release()
            time.sleep(0.1)
            continue
        
        frame = dataSetGen.frame_
        dataSetGen.frame_ = dataSetGen.frame_ + 1
        lock.release()

        pos = (_world.player.get_transform().location.x, 
               _world.player.get_transform().location.y) # self._get_position(tick_data)
        theta = _world.imu_sensor.compass # 0 # tick_data['compass']
        v = _world.player.get_velocity()
        c = _world.player.get_control()
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        # weather = tick_data['weather']

        data = {
                'x': pos[0],
                'y': pos[1],
                'theta': theta,
                'speed': speed,
                'target_speed': speed,
                'x_command': 0,
                'y_command': 0,
                'command': 0,
                'steer': c.steer,
                'throttle': c.throttle,
                'brake': c.brake,
                # 'weather': weather,
                # 'weather_id': self.weather_id,
                'near_node_x': 0,
                'near_node_y': 0,
                'far_node_x':0,
                'far_node_y': 0,
                'is_vehicle_present': False,
                'is_pedestrian_present': False,
                'is_red_light_present': False,
                'is_stop_sign_present': False,
                'should_slow': False,
                'should_brake':False,
                }

        measurements_file = dataSetGen.save_path / 'measurements' / ('%04d.json' % frame)
        f = open(measurements_file, 'w')
        json.dump(data, f, indent=4)
        f.close()

        actors_data = dataSetGen.collect_actor_data(_world.world, _world.player)
        actors_data_file = dataSetGen.save_path / "actors_data" / ("%04d.json" % frame)
        f = open(actors_data_file, "w")
        json.dump(actors_data, f, indent=4)
        f.close()


        
        for pos in ['front', 'left', 'right', 'tel', 'rear']:
            name = 'rgb_' + pos
            Image.fromarray(data_buf[name][1]).save(dataSetGen.save_path / name / ('%04d.png' % frame))
            # for sensor_type in ['seg', 'depth']:
            #     name = sensor_type + '_' + pos
            #     Image.fromarray(data_buf[name][1]).save(self.save_path / name / ('%04d.png' % frame))
            # for sensor_type in ['2d_bbs']:
            #     name = sensor_type + '_' + pos
            #     np.save(self.save_path / name / ('%04d.npy' % frame), self.data_buf[name], allow_pickle=True)

        # Image.fromarray(self.data_buf['topdown']).save(self.save_path / 'topdown' / ('%04d.png' % frame))
        # np.save(self.save_path / 'lidar' / ('%04d.npy' % frame), data_buf['lidar'][1], allow_pickle=True)
        # np.save(self.save_path / 'semantic_lidar' / ('%04d.npy' % frame), tick_data['semantic_lidar'], allow_pickle=True)
        # np.save(self.save_path / '3d_bbs' / ('%04d.npy' % frame), self.data_buf['3d_bbs'], allow_pickle=True)
        # np.save(self.save_path / 'affordances' / ('%04d.npy' % frame), self.data_buf['affordances'], allow_pickle=True)

    pass

class DataSetGen:

    weather_index = 0;
    last_time = None
    frame_ = 0
    route = []
    _sensor_data = {
            'width': 768,
            'height': 512,
            'fov': 100
        }
    
    hasSetSensor = False
    _sensors_list = []

    def collect_actor_data(self, _world, _vehicle):
        data = {}
        vehicles = _world.get_actors().filter("*vehicle*")
        for actor in vehicles:
            loc = actor.get_location()
            if loc.distance(_vehicle.get_location()) > 50:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            box = actor.bounding_box.extent
            data[_id]["box"] = [box.x, box.y]
            vel = actor.get_velocity()
            data[_id]["vel"] = [vel.x, vel.y, vel.z]
            data[_id]["tpe"] = 0

        walkers = _world.get_actors().filter("*walker*")
        for actor in walkers:
            loc = actor.get_location()
            if loc.distance(_vehicle.get_location()) > 50:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            box = actor.bounding_box.extent
            data[_id]["box"] = [box.x, box.y]
            vel = actor.get_velocity()
            data[_id]["vel"] = [vel.x, vel.y, vel.z]
            data[_id]["tpe"] = 1

        lights = _world.get_actors().filter("*traffic_light*")
        for actor in lights:
            loc = actor.get_location()
            if loc.distance(_vehicle.get_location()) > 70:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            vel = actor.get_velocity()
            data[_id]["sta"] = int(actor.state)
            data[_id]["tpe"] = 2

            trigger = actor.trigger_volume
            box = trigger.extent
            loc = trigger.location
            ori = trigger.rotation.get_forward_vector()
            data[_id]["taigger_loc"] = [loc.x, loc.y, loc.z]
            data[_id]["trigger_ori"] = [ori.x, ori.y, ori.z]
            data[_id]["trigger_box"] = [box.x, box.y]


        cones = _world.get_environment_objects(carla.CityObjectLabel.Other)
        for actor in cones:
            loc = actor.bounding_box.location
            if loc.distance(_vehicle.get_location()) > 70:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.bounding_box.rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            box = actor.bounding_box.extent
            data[_id]["box"] = [box.x, box.y]
            data[_id]["tpe"] = 3

        return data
    
    def __init__(self, _world):

        self.sensor_interface = SensorInterface()

        save_path_ = r'E:\CarlaDataSet\data'
        now = datetime.datetime.now()
        map_name = _world.map.name.replace('/', '_')
        string = pathlib.Path(map_name).stem + '_'
        string += 'w{}'.format(self.weather_index) + '_'
        string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

        print (string)

        self.save_path = pathlib.Path(save_path_) / string
        self.save_path.mkdir(parents=True, exist_ok=False)
        
        for sensor in self.sensors():
            if hasattr(sensor, 'save') and sensor['save']:
                (self.save_path / sensor['id']).mkdir()

        (self.save_path / '3d_bbs').mkdir(parents=True, exist_ok=True)
        (self.save_path / 'affordances').mkdir(parents=True, exist_ok=True)
        (self.save_path / 'measurements').mkdir(parents=True, exist_ok=True)
        (self.save_path / "actors_data").mkdir(parents=True, exist_ok=True)
        (self.save_path / 'lidar').mkdir(parents=True, exist_ok=True)
        # (self.save_path / 'semantic_lidar').mkdir(parents=True, exist_ok=True)
        (self.save_path / 'topdown').mkdir(parents=True, exist_ok=True)

        for pos in ['front', 'left', 'right', 'tel', 'rear']:
            for sensor_type in ['rgb', 'seg', 'depth', '2d_bbs']:
                name = sensor_type + '_' + pos
                (self.save_path / name).mkdir()

        self.threads_ = []
        for i in range(10):
            save_task = threading.Thread(target=save_handle, args=(self, _world))
            save_task.start()
            self.threads_.append(save_task)

    def sensors(self):
        return [
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                'id': 'rgb_front'
                },
            # {
            #     'type': 'sensor.camera.semantic_segmentation',
            #     'x': 1.3, 'y': 0.0, 'z': 2.3,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
            #     'id': 'seg_front'
            #     },
            # {
            #     'type': 'sensor.camera.depth',
            #     'x': 1.3, 'y': 0.0, 'z': 2.3,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
            #     'id': 'depth_front'
            #     },
                ######################################################
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': 60,
                'id': 'rgb_tel'
                },
            # {
            #     'type': 'sensor.camera.semantic_segmentation',
            #     'x': 1.3, 'y': 0.0, 'z': 2.3,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': 60,
            #     'id': 'seg_tel'
            #     },
            # {
            #     'type': 'sensor.camera.depth',
            #     'x': 1.3, 'y': 0.0, 'z': 2.3,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': 60,
            #     'id': 'depth_tel'
            #     },
            {
                'type': 'sensor.camera.rgb',
                'x': -1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                'id': 'rgb_rear'
                },
            # {
            #     'type': 'sensor.camera.semantic_segmentation',
            #     'x': -1.3, 'y': 0.0, 'z': 2.3,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
            #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
            #     'id': 'seg_rear'
            #     },
            # {
            #     'type': 'sensor.camera.depth',
            #     'x': -1.3, 'y': 0.0, 'z': 2.3,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
            #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
            #     'id': 'depth_rear'
            #     },
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                'id': 'rgb_left'
                },
            # {
            #     'type': 'sensor.camera.semantic_segmentation',
            #     'x': 1.3, 'y': 0.0, 'z': 2.3,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
            #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
            #     'id': 'seg_left'
            #     },
            # {
            #     'type': 'sensor.camera.depth',
            #     'x': 1.3, 'y': 0.0, 'z': 2.3,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
            #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
            #     'id': 'depth_left'
            #     },
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                'id': 'rgb_right'
                },
            # {
            #     'type': 'sensor.camera.semantic_segmentation',
            #     'x': 1.3, 'y': 0.0, 'z': 2.3,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
            #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
            #     'id': 'seg_right'
            #     },
            # {
            #     'type': 'sensor.camera.depth',
            #     'x': 1.3, 'y': 0.0, 'z': 2.3,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
            #     'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
            #     'id': 'depth_right'
            #     },
            # {   
            #     'type': 'sensor.lidar.ray_cast',
            #     'x': 1.3, 'y': 0.0, 'z': 2.5,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
            #     'id': 'lidar'
            #     },
            # {   
            #     'type': 'sensor.lidar.ray_cast_semantic',
            #     'x': 1.3, 'y': 0.0, 'z': 2.5,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
            #     'id': 'semantic_lidar'
            #     },
            # {
            #     'type': 'sensor.other.imu',
            #     'x': 0.0, 'y': 0.0, 'z': 0.0,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #     'sensor_tick': 0.05,
            #     'id': 'imu'
            #     },
            # {
            #     'type': 'sensor.other.gnss',
            #     'x': 0.0, 'y': 0.0, 'z': 0.0,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #     'sensor_tick': 0.01,
            #     'id': 'gps'
            #     },
            # {
            #     'type': 'sensor.speedometer',
            #     'reading_frequency': 20,
            #     'id': 'speed'
            #     }
            ]

    def tick(self, _world):

        def dist2(pos1, pos2):
            return (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + \
                    (pos1[1] - pos2[1]) * (pos1[1] - pos2[1])
        
        pos = (_world.player.get_transform().location.x, 
               _world.player.get_transform().location.y)

        if len(self.route) == 0:
            self.route.append(pos)
        else:
            if dist2(self.route[-1], pos) > 5 * 5:
                self.route.append(pos)
            pass
        pass

    def fill_measurement_route(self):

        def dist2(pos1, pos2):
            return (pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + \
                    (pos1[1] - pos2[1]) * (pos1[1] - pos2[1])

        for i in range(self.frame_):

            measurements_file = self.save_path / 'measurements' / ('%04d.json' % i)
            f = open(measurements_file, 'r')
            data = json.load(f)
            f.close()

            pos = (data['x'], data['y'])
            p = 0
            for _ in range(len(self.route)):
                if dist2(self.route[_], pos) < 5 * 5:
                    p = _
                    break
            _route = self.route[p:]
            if len(_route) >= 1:
                _route = _route[1:]
            else: 
                _route = []
            data['future_waypoints'] = _route
            
            f = open(measurements_file, 'w')
            json.dump(data, f, indent=4)
            f.close()
        pass

    def delete_files(self):

        time.sleep(3)
        shutil.rmtree(self.save_path)
        pass

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_k
    from pygame.locals import K_j
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
global_world = None

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma

        self.dataSetGen = DataSetGen(self)

        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

        self.setup_sensors(self.player, self.dataSetGen.sensors())
        self.start_time = time.time()

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
            blueprint.set_attribute('role_name', self.actor_role_name)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if blueprint.has_attribute('is_invincible'):
                blueprint.set_attribute('is_invincible', 'true')
            # set the max speed
            if blueprint.has_attribute('speed'):
                self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
                self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
            else:
                print("No recommended values for 'speed' attribute")
                
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def tick(self, clock):
        self.hud.tick(self, clock)
        self.dataSetGen.tick(self)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None


    def setup_sensors(self, vehicle, sensors):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        if self.dataSetGen.hasSetSensor:
            return
        
        self.dataSetGen.hasSetSensor = True


        bp_library = self.world.get_blueprint_library()
        for sensor_spec in sensors:
            # These are the pseudosensors (not spawned)
            if sensor_spec['type'].startswith('sensor.speedometer'):
                pass
            #     delta_time = self.world.get_settings().fixed_delta_seconds
            #     frame_rate = 1 / delta_time
            #     sensor = SpeedometerReader(vehicle, frame_rate)
            # # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(str(sensor_spec['type']))
                if sensor_spec['type'].startswith('sensor.camera'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    bp.set_attribute('lens_circle_multiplier', str(3.0))
                    bp.set_attribute('lens_circle_falloff', str(3.0))
                    # bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                    # bp.set_attribute('chromatic_aberration_offset', str(0))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                    sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                    sensor = self.world.spawn_actor(bp, sensor_transform, 
                                                    attach_to=vehicle, 
                                                    attachment_type=carla.AttachmentType.Rigid)
                    
                    sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self.dataSetGen.sensor_interface))

                elif sensor_spec['type'].startswith('sensor.lidar'):
                    bp.set_attribute('range', str(85))
                    bp.set_attribute('rotation_frequency', str(10))
                    bp.set_attribute('channels', str(64))
                    bp.set_attribute('upper_fov', str(10))
                    bp.set_attribute('lower_fov', str(-30))
                    bp.set_attribute('points_per_second', str(600000))
                    # bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
                    # bp.set_attribute('dropoff_general_rate', str(0.45))
                    # bp.set_attribute('dropoff_intensity_limit', str(0.8))
                    # bp.set_attribute('dropoff_zero_intensity', str(0.4))
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                    
                    sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                    sensor = self.world.spawn_actor(bp, sensor_transform, vehicle)

                    sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self.dataSetGen.sensor_interface))

                elif sensor_spec['type'].startswith('sensor.other.radar'):
                    bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('points_per_second', '1500')
                    bp.set_attribute('range', '100')  # meters

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                    
                    sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                    sensor = self.world.spawn_actor(bp, sensor_transform, vehicle)

                    sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self.dataSetGen.sensor_interface))


                elif sensor_spec['type'].startswith('sensor.other.gnss'):
                    bp.set_attribute('noise_alt_stddev', str(0.000005))
                    bp.set_attribute('noise_lat_stddev', str(0.000005))
                    bp.set_attribute('noise_lon_stddev', str(0.000005))
                    bp.set_attribute('noise_alt_bias', str(0.0))
                    bp.set_attribute('noise_lat_bias', str(0.0))
                    bp.set_attribute('noise_lon_bias', str(0.0))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation()

                    sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                    sensor = self.world.spawn_actor(bp, sensor_transform, vehicle)

                    sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self.dataSetGen.sensor_interface))


                elif sensor_spec['type'].startswith('sensor.other.imu'):
                    bp.set_attribute('noise_accel_stddev_x', str(0.001))
                    bp.set_attribute('noise_accel_stddev_y', str(0.001))
                    bp.set_attribute('noise_accel_stddev_z', str(0.015))
                    bp.set_attribute('noise_gyro_stddev_x', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_y', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_z', str(0.001))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                    
                    sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                    sensor = self.world.spawn_actor(bp, sensor_transform, vehicle)

                    sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self.dataSetGen.sensor_interface))

                # create sensor

            # setup callback
            # sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self._agent.sensor_interface))
            sensor_spec['bp'] = bp
            self.dataSetGen._sensors_list.append(sensor)

        # # Tick once to spawn the sensors
        # CarlaDataProvider.get_world().tick()
   

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._carsim_enabled = False
        self._carsim_road = False
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            # world.player.set_autopilot(self._autopilot_enabled)
            # world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        global b_stop
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                b_stop = True
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    b_stop = True
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_k and (pygame.key.get_mods() & KMOD_CTRL):
                    print("k pressed")
                    world.player.enable_carsim("d:/CVC/carsim/DataUE4/ue4simfile.sim")
                elif event.key == K_j and (pygame.key.get_mods() & KMOD_CTRL):
                    self._carsim_road = not self._carsim_road
                    world.player.use_carsim_road(self._carsim_road)
                    print("j pressed, using carsim road =", self._carsim_road)
                # elif event.key == K_i and (pygame.key.get_mods() & KMOD_CTRL):
                #     print("i pressed")
                #     imp = carla.Location(z=50000)
                #     world.player.add_impulse(imp)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            # world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        global b_collision
        global b_stop
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        b_collision = True
        b_stop = True
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=0.8, y=0, z=1.8), carla.Rotation(pitch=.0)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)

        global global_world
        global_world = world

        clock = pygame.time.Clock()
        global b_collision
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock) or b_collision:
                if not b_collision:
                    world.dataSetGen.fill_measurement_route()
                else:
                    world.dataSetGen.delete_files()
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()



import socket
import threading

tcp_port = 12123
client_num = 2


def message_handle(tcp_client, tcp_client_ip):

    while True:
        try:
            recv_data = tcp_client.recv(64)
            if recv_data:
                recv_data = recv_data.decode()
                control__ = str(recv_data).split(',')
                for i in range(len(control__)):
                    if control__[i] == '#' and len(control__) - i > 4:
                        control__ = control__[i+1: i+4]
                        

                        if len(control__) == 3 and global_world is not None:
                            vehicle_control = carla.VehicleControl()
                            vehicle_control.throttle = min(float(control__[1]), 1)
                            vehicle_control.steer = min(float(control__[0]), 1)
                            vehicle_control.brake = min(float(control__[2]), 1)
                            global_world.player.apply_control(vehicle_control)
                            pass

                        break
            else:
                break

        except Exception as e:  # 连接出现异常的处理

            print(str(tcp_client_ip) + str(e))
            break


def server_listen():

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 使用UDP协议
    # server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
    server.bind(("10.146.146.112", tcp_port))
    # server.listen(client_num)

    # while True:
        # client, client_ip = server.accept()
        # client.settimeout(10)  # 设置连接的超时期
        # print(str(client_ip) + "接入")
        # thd = threading.Thread(target=message_handle, args=(client, client_ip))
        # thd.setDaemon(True)  # 设置守护主线程
        # thd.start()
    global b_stop
    while not b_stop:
        try:
            recv_data, client = server.recvfrom(1024)
            if recv_data:
                recv_data = recv_data.decode()
                control__ = str(recv_data).split(',')
                for i in range(len(control__)):
                    if control__[i] == '#' and len(control__) - i > 4:
                        control__ = control__[i+1: i+4]
                        

                        if len(control__) == 3 and global_world is not None:
                            vehicle_control = carla.VehicleControl()
                            vehicle_control.throttle = min(float(control__[1]), 1)
                            vehicle_control.steer = min(float(control__[0]), 1)
                            vehicle_control.brake = min(float(control__[2]), 1)
                            global_world.player.apply_control(vehicle_control)
                            pass

                        break
            else:
                break

        except Exception as e:  # 连接出现异常的处理

            print(str(e))
            break


from net.timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    Mixup,
    FastCollateMixup,
    AugMixDataset,
)
from net.timm.data import create_carla_dataset, create_carla_loader
from net.timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    load_checkpoint,
    convert_splitbn_model,
    model_parameters,
)
from net.timm.utils import *
from net.timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
)
from net.timm.optim import create_optimizer_v2, optimizer_kwargs
from net.timm.scheduler import create_scheduler
from net.timm.utils import ApexScaler, NativeScaler
from net.Demo import * 
from net.timm.data import create_carla_rgb_transform

def handle_end2end_model(dataSetGen, _world):

    model = create_model('DemoEnd2EndNet_baseline')
    pretrained_dict = torch.load("")["state_dict"]
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, False)

    model.cuda()

    model.eval()

    rgb_transform = create_carla_rgb_transform(224)
    rgb_center_transform = create_carla_rgb_transform(224, need_scale=False)

    global b_stop
    while not b_stop:

        try:
            data_buf = dataSetGen.sensor_interface.get_data()
            if len(data_buf) == 0 or data_buf['rgb_front'][0] % 5 != 0:
                continue
        except:

            time.sleep(0.1)
            continue
        
        frame = dataSetGen.frame_
        dataSetGen.frame_ = dataSetGen.frame_ + 1


        pos = (_world.player.get_transform().location.x, 
               _world.player.get_transform().location.y) # self._get_position(tick_data)
        theta = _world.imu_sensor.compass # 0 # tick_data['compass']
        v = _world.player.get_velocity()
        c = _world.player.get_control()
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        # weather = tick_data['weather']

        data = {
                'x': pos[0],
                'y': pos[1],
                'theta': theta,
                'speed': speed,
                'target_speed': speed,
                'x_command': 0,
                'y_command': 0,
                'command': 0,
                'steer': c.steer,
                'throttle': c.throttle,
                'brake': c.brake,
                # 'weather': weather,
                # 'weather_id': self.weather_id,
                'near_node_x': 0,
                'near_node_y': 0,
                'far_node_x':0,
                'far_node_y': 0,
                'is_vehicle_present': False,
                'is_pedestrian_present': False,
                'is_red_light_present': False,
                'is_stop_sign_present': False,
                'should_slow': False,
                'should_brake':False,
                }

        actors_data = dataSetGen.collect_actor_data(_world.world, _world.player)
        
        rgb_left_image = data_buf['rgb_left'][1]
        rgb_right_image = data_buf['rgb_right'][1]
        rgb_front_image = data_buf['rgb_front'][1]
        rgb_center_image = data_buf['rgb_tel'][1]
        
        rgb = (
            rgb_transform(Image.fromarray(rgb_front_image))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_left = (
            rgb_transform(Image.fromarray(rgb_left_image))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_right = (
            rgb_transform(Image.fromarray(rgb_right_image))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_center = (
            rgb_center_transform(Image.fromarray(rgb_center_image))
            .unsqueeze(0)
            .cuda()
            .float()
        )

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = data["command"] - 1
        if cmd < 0:
            cmd = 3
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(data["speed"])
        mes = np.array(cmd_one_hot)
        mes = torch.from_numpy(mes).float()

        input_data = {}
        input_data["rgb"] = rgb
        input_data["rgb_left"] = rgb_left
        input_data["rgb_right"] = rgb_right
        input_data["rgb_center"] = rgb_center

        input_data["measurements"] = mes
        input_data['command'] = cmd
        
        output = model(input_data)


        pass
    pass
# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    # thd = threading.Thread(target=server_listen, args=())
    # thd.setDaemon(True)  # 设置守护主线程
    # thd.start()

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()

