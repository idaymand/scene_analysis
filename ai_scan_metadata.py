import csv
import os
import numpy as np

import tensorflow as tf
from PIL.Image import Image

from estimate_angles_FE_model import angles_from_model, get_score_weights
from metadataHandler import View

from detectable_object import DetectionMap, DetectionContainer, DetectionHistMap, DetectedObject


class AIScanMetadata:
    def __init__(self, configurator, model_file_name, detections=None, i_width=None, i_height=None):
        self.index_output = [5, 4, 6, 1, 0, 2, 3]
        self.detections = detections
        self.GENERAL_IMAGE_SIDE = 300
        self.width = i_width
        self.height = i_height
        self.angleCars = []
        self.configurator = configurator
        self.bestCar = None
        self.params = {}
        self.angleDist = []
        for index in range(len(self.index_output)):
            self.angleDist.append([])
        self.distributions = []
        self.model_file_name = model_file_name

    def process_part_detector_model(self):
        pass

    def process_part_quality_model(self):
        pass

    def process_parts_from_view_metadata(self, input_file_name):
        PARTS = self.configurator.get_parts()
        n = 5
        v = View(input_file_name)
        car_box = v.data['car'][0]['rect']
        score = v.data['car'][0]['Score']
        view_type = 'exterior'
        container = DetectionContainer(PARTS['CAR'], car_box, score, view_type, self.configurator.file_name)
        v, distributions = angles_from_model(v, self.model_file_name)
        self.distributions = []
        self.params['exterior'] = {}
        for index in range(len(self.index_output)):
            self.distributions.append(distributions[self.index_output[index]])
        weights_score = get_score_weights(v.data)
        weights = [np.sum(weights_score)]
        for index in range(len(self.distributions)):
            self.setAngle(self.distributions[index], index, np.array(weights), view_type)
        container.register_angle(self.angleDist)
        best_scene, scenes = container.get_top_n_scene_candidates_only_angle(n)
        return best_scene, scenes

    def process_parts_from_file(self, input_file_name, path_to_image, path_to_csv):
        if os.path.isfile(os.path.join(path_to_image, input_file_name.replace('.csv', ''))):
            self.width, self.height = Image.open(os.path.join(path_to_image, input_file_name.replace('.csv', ''))).size
        self.detections = []
        with open(os.path.join(path_to_csv, input_file_name), 'r') as fr:  # Just use 'w' mode in 3.x
            data = csv.DictReader(fr)
            for row in data:
                box = [row['left'] * self.width, row['top'] * self.height, (row['right'] - row['left']) * self.width,
                       (row['bottom'] - row['top']) * self.height]
                score = row['score']
                class_id = row['class_id']
                part = DetectedObject(class_id, box, score)
                self.detections.append(part)
        return self.detections

    def find_cars(self):
        self.angleCars = DetectionMap(self.configurator)
        for view_type in self.configurator.get_angle_models():
            angle_model_config = self.configurator.get_angle_models()[view_type]
            am_bbox = [max(angle_model_config['bbox'][0], 0), max(angle_model_config['bbox'][1], 0),
                       min(angle_model_config['bbox'][2], self.width), min(angle_model_config['bbox'][3], self.height),
                       angle_model_config['bbox'][4]];
            def_car = DetectionContainer(am_bbox, angle_model_config['classId'], angle_model_config['viewType']);
            if self.bestCar is None:
                self.bestCar = def_car
            self.angleCars.addDetection(def_car);

            for element in self.detections:
                if element.class_id == angle_model_config['classId']:
                    self.angleCars.addDetection(DetectionContainer(element.bbox, element.classId,
                                                                   angle_model_config['viewType']))

            for car in self.angleCars[angle_model_config['classId']]:
                for element in self.detections:
                    if element.ios(car) > \
                            self.configurator.get_classes()['{}'.format(element.class_id)]['iou_merge_threshold']:
                        if angle_model_config.basicObjects.includes(element.class_id):
                            obj1 = DetectionHistMap(element.bbox, element.class_id, element.score, car,
                                                    angle_model_config)
                            car.addDetection(obj1)
                if car.totalScore > self.bestCar.totalScore:
                    self.bestCar = car

                if car.totalScore > def_car.totalScore:
                    def_car = car

            self.params[def_car.viewType] = {}
            self.params[def_car.viewType].car = [def_car.box[0] / self.width, def_car.box[1] / self.height,
                                                 def_car.box[2] / self.width, def_car.box[3] / self.height]
            if angle_model_config['basicObjects'].length:
                self.calcAngle(def_car, angle_model_config)
                def_car.registerAngle(self.angleDist)

    def calcAngle(self, car, angle_model_config):
        pass

    def setAngle(self, angles, ix, weights, viewType):
        angles = np.array(angles)
        INDEX_LUT = self.configurator.get_index_lut()
        ANGLE_LUT = self.configurator.get_angle_lut()
        nx = angles.size / weights.size
        nx = nx.__int__()
        self.angleDist[INDEX_LUT[ix]] = []
        for index in range(0, nx):
            self.angleDist[INDEX_LUT[ix]].append(0)
        for index in range(weights.size):
            dst = angles[index * nx:(index + 1) * nx]
            dst = np.exp(dst)
            tot = np.sum(dst)
            for index1 in range(len(self.angleDist[INDEX_LUT[ix]])):
                self.angleDist[INDEX_LUT[ix]][index1] = self.angleDist[INDEX_LUT[ix]][index1] + dst[0][index1] / tot * \
                                                        weights[index]
        tot2 = np.sum(self.angleDist[INDEX_LUT[ix]])
        for index1 in range(len(self.angleDist[INDEX_LUT[ix]])):
            self.angleDist[INDEX_LUT[ix]][index1] = self.angleDist[INDEX_LUT[ix]][index1] / tot2

        if ix == 3:
            for index1 in range(len(self.angleDist[INDEX_LUT[ix]])):
                a = [self.angleDist[INDEX_LUT[ix]][(index1 - 1 + nx) % nx], self.angleDist[INDEX_LUT[ix]][index1],
                     self.angleDist[INDEX_LUT[ix]][(index1 + 1 + nx) % nx]]
                a.sort()
                self.angleDist[INDEX_LUT[ix]][index1] = a[1]
            for index1 in range(len(self.angleDist[INDEX_LUT[ix]])):
                self.angleDist[INDEX_LUT[ix]][index1] = 0.25 * self.angleDist[INDEX_LUT[ix]][(index1 - 1 + nx) % nx]
                + 0.5 * self.angleDist[INDEX_LUT[ix]][index1] \
                    + 0.25 * self.angleDist[INDEX_LUT[ix]][(index1 + 1 + nx) % nx]

        angles1 = self.angleDist[INDEX_LUT[ix]]
        if self.params[viewType].get('angle', None) is None:
            self.params[viewType]['angle'] = [0, 0, 0, 0, 0, 0]
        if ix == 4:
            self.params[viewType]['standardModel'] = ANGLE_LUT[ix][angles1.index(np.max(angles1))]
        else:
            self.params[viewType]['angle'][INDEX_LUT[ix]] = ANGLE_LUT[ix][angles1.index(np.max(angles1))]
        if self.params[viewType].get('angleConf', None) is None:
            self.params[viewType]['angleConf'] = [0, 0, 0, 0, 0, 0, 0]
        if len(weights) > 1:
            self.params[viewType]['angleConf'][INDEX_LUT[ix]] = 1.0
        else:
            angles.sort()
            mid = np.floor(angles.size / 2)
            mid = mid.__int__()
            med = angles[0][mid]
            if angles.size % 2 > 0:
                med = (angles[0][mid - 1] + angles[0][mid]) / 2
            self.params[viewType]['angleConf'][ix] = (angles[0][0] - angles[0][1]) / (angles[0][0] - med)
