import csv
import json
import os
import shutil

import numpy as np

from scene_analysis_lib.detectable_object import DetectedObject


class ScanSaveData:
    def __init__(self, output_data_path, configurator, output_sub_folder=''):
        self.global_path = output_data_path
        self.data_for_save = []
        self.output_scan_via_path = 'annotations'
        self.input_scan_via_path = 'csv'
        self.output_sub_folder = output_sub_folder
        self.configurator = configurator
        self.data_main_for_save = []
        self.output_main_scan_via_path = 'main_annotation'

    def make_folder_output_via_data(self):
        if os.path.exists(os.path.join(self.global_path, self.output_scan_via_path)):
            shutil.rmtree(os.path.join(self.global_path, self.output_scan_via_path))
        os.makedirs(os.path.join(self.global_path, self.output_scan_via_path), exist_ok=True)

    def save_scan_detections(self, ai_scan):
        if ai_scan is None:
            return
        classes = self.configurator.get_classes()
        parts = self.configurator.get_parts()
        box_count = 0
        element_count = 1
        for obj_list in ai_scan.bestCar.partMap.mapData:
            element_count = element_count + len(obj_list)
        self.data_for_save = []
        class_name = classes[str(ai_scan.bestCar.class_id)]['name']
        box_count = self.add_one_box_to_save_data(ai_scan.bestCar, box_count, class_name, ai_scan,
                                                  element_count, True)
        for class_id, obj_list in ai_scan.bestCar.partMap.mapData.items():
            if int(class_id) == parts.get('MOCK_PART', 0):
                continue
            class_name = classes[class_id]['name']
            box_count = self.add_data_for_save(obj_list, box_count, class_name, ai_scan, element_count, True)
        full_file_name = os.path.join(self.global_path, self.output_sub_folder,
                                      self.input_scan_via_path, ai_scan.image_name + '.csv')
        with open(full_file_name, 'w', encoding='UTF8') as f:
            w = csv.DictWriter(f, self.get_via_header(), delimiter=',', lineterminator='\n')
            w.writeheader()
            w.writerows(self.data_for_save)

    def save_scan_data(self, ai_scan, main_image=False, only_original=False, mock_parts=False):
        if ai_scan is None:
            return
        classes = self.configurator.get_classes()
        parts = self.configurator.get_parts()
        self.data_for_save = []
        self.data_main_for_save = []
        box_count = 0
        element_count = 1
        mock_class_id = self.configurator.get_parts()['MOCK_PART']
        for class_id, obj_list in ai_scan.bestCar.partMap.mapData.items():
            if str(mock_class_id) == class_id and mock_parts is False:
                continue
            element_count = element_count + len(obj_list)
        if ai_scan.damage_object is not None and only_original is False:
            element_count += 1
        if only_original is False:
            for obj_list in ai_scan.bestCar.virtualPartMap.mapData:
                element_count = element_count + len(obj_list)
        if only_original is False:
            element_count = element_count+9
        class_name = classes[str(ai_scan.bestCar.class_id)]['elementName']
        box_count = self.add_one_box_to_save_data(ai_scan.bestCar, box_count, class_name, ai_scan,
                                                  element_count)
        for class_id, obj_list in ai_scan.bestCar.partMap.mapData.items():
            if str(mock_class_id) == class_id and mock_parts is False:
                continue
            class_name = classes[class_id]['elementName']
            box_count = self.add_data_for_save(obj_list, box_count, class_name, ai_scan, element_count)
        if only_original is False:
            if mock_parts is True:
                for point_quadrant in ai_scan.bestCar.quadrants_selected_points:
                    obj = DetectedObject(96, [point_quadrant[0], point_quadrant[1], 1, 1], ai_scan.bestCar.score)
                    class_name = 'q'
                    box_count = self.add_one_box_to_save_data(obj, box_count, class_name, ai_scan, element_count)
            for class_id, obj_list in ai_scan.bestCar.virtualPartMap.mapData.items():
                class_name = '{}'.format(parts[class_id])
                box_count = self.add_data_for_save(obj_list, box_count, class_name, ai_scan, element_count,
                                                   save_mock_part=False)
        if ai_scan.damage_object is not None and only_original is False:
            if ai_scan.damage_object.part_location != '':
                class_name = '{}_{}'.format(ai_scan.damage_object.damage_type, ai_scan.damage_object.part_location)
            else:
                if ai_scan.damage_object.part_name != '':
                    class_name = '{}_{}'.format(ai_scan.damage_object.damage_type, ai_scan.damage_object.part_name)
                else:
                    class_name = '{}'.format(ai_scan.damage_object.damage_type)
            box_count = self.add_one_box_to_save_data(ai_scan.damage_object, box_count, class_name, ai_scan,
                                                      element_count)
        if ai_scan.best_scene is not None and only_original is False:
            if 'angle' in ai_scan.params[ai_scan.bestCar.viewType].keys():
                class_name = '{}.{}.yaw-{}.ar-{}'.format(ai_scan.best_scene.base_location,
                                                         str(ai_scan.best_scene.scene_id),
                                                         ai_scan.params[ai_scan.bestCar.viewType]['angle'][1],
                                                         np.round(ai_scan.bestCar.aspect_ratio(), 2))
            else:
                class_name = f'{ai_scan.best_scene.base_location}.{str(ai_scan.best_scene.scene_id)}'

            obj = DetectedObject(96, [40, 80, 700, 10], ai_scan.best_scene.score)
            self.add_one_box_to_save_data(obj, box_count, class_name, ai_scan, element_count)

        if ai_scan.params[ai_scan.bestCar.viewType].get('sceneConf', 0) > 0 and only_original is False:
            scene_id = ai_scan.params[ai_scan.bestCar.viewType].get('scene')
            scene_score = ai_scan.params[ai_scan.bestCar.viewType].get('sceneConf', 0)
            id_base_location = ai_scan.configurator.get_object_constructed_from()[str(scene_id)]['base_location']
            base_location = \
            ai_scan.configurator.get_location()[ai_scan.configurator.get_location_index()[str(id_base_location)]][
                'name']
            class_name = 'rami_{}_{}'.format(base_location, str(scene_id))
            obj = DetectedObject(96, [40, 140, 700, 10], np.round(scene_score, 4))
            self.add_one_box_to_save_data(obj, box_count, class_name, ai_scan, element_count)
        if ai_scan.params[ai_scan.bestCar.viewType].get('BodyTypeConf', 0) > 0 and only_original is False:
            body_type = ai_scan.params[ai_scan.bestCar.viewType].get('BodyType')
            body_type_score = ai_scan.params[ai_scan.bestCar.viewType].get('BodyTypeConf', 0)
            class_name = 'BT_{}_{}'.format(body_type, str(np.round(body_type_score, 3)))
            obj = DetectedObject(96, [40, 210, 700, 10], body_type_score)
            self.add_one_box_to_save_data(obj, box_count, class_name, ai_scan, element_count)
        full_file_name = os.path.join(self.global_path,
                                      self.output_scan_via_path, ai_scan.image_name + '.csv')
        with open(full_file_name, 'w', encoding='UTF8') as f:
            w = csv.DictWriter(f, self.get_via_header(), delimiter=',', lineterminator='\n')
            w.writeheader()
            w.writerows(self.data_for_save)
        if main_image is True:
            full_file_name = os.path.join(self.global_path, self.output_sub_folder,
                                          self.output_main_scan_via_path, ai_scan.image_name + '.csv')
            with open(full_file_name, 'w', encoding='UTF8') as f:
                w = csv.DictWriter(f, self.get_via_header(), delimiter=',', lineterminator='\n')
                w.writeheader()
                w.writerows(self.data_main_for_save)
            full_file_name = os.path.join(self.global_path, self.output_sub_folder,
                                          self.output_main_scan_via_path, ai_scan.image_name)
            file_name = os.path.join(self.global_path, self.output_sub_folder, ai_scan.image_name)
            shutil.copy(file_name, full_file_name)

    def save_body_type_data(self, body_type_data, event_id=10000000, file_name='body_type_csv'):
        output_folder = os.path.join('./stat', str(event_id))
        os.makedirs(output_folder, exist_ok=True)
        body_type_header = ["eventid", "imagename", "car_x", "car_width", "direction",
                            "score_direction", "body_type", "score_body_type"]
        file_name = '{}.csv'.format(file_name)
        if len(body_type_data) > 0:
            full_file_name = os.path.join(output_folder, file_name)
            with open(full_file_name, 'w', encoding='UTF8') as f:
                w = csv.DictWriter(f, body_type_header, delimiter=',', lineterminator='\n')
                w.writeheader()
                w.writerows(body_type_data)

    def save_all_scan_data(self, data, additional_information=None, only_original=False):
        index = 0
        for ai_scan in data:
            self.output_scan_via_path = ai_scan.output_path
            self.output_sub_folder = ai_scan.output_sub_folder
            self.save_scan_data(ai_scan, only_original=only_original)
            index += 1
        for key, value in additional_information.items():
            if ('scan' in value.keys()) is True:
                self.output_scan_via_path = value['scan'].output_path
                self.output_sub_folder = value['scan'].output_sub_folder
                self.save_scan_data(value['scan'], True)

    def add_data_for_save(self, obj_list, box_count, class_name, ai_scan, element_count, region_attributes=False,
                          save_mock_part=False):
        for obj in obj_list:
            if save_mock_part is True:
                for obj_mock in obj.triangle_basis:
                    if obj_mock.class_id == self.configurator.get_parts()['MOCK_PART']:
                        location_mock = 'mock'
                        box_count = self.add_one_box_to_save_data(obj_mock, box_count, location_mock, ai_scan,
                                                                  element_count, region_attributes)
                        break
            if region_attributes is False:
                location = '{}_{}'.format(class_name, obj.base_location)
            else:
                location = class_name
            box_count = self.add_one_box_to_save_data(obj, box_count, location, ai_scan,
                                                      element_count, region_attributes)
        return box_count

    def add_one_box_to_save_data(self, obj, box_count, class_name, ai_scan, element_count,
                                 region_attributes=False, center_annotation=False):
        shape_dict = json.dumps({'name': 'rect', 'x': obj.box[0], 'y': obj.box[1],
                                 'width': obj.box[2],
                                 'height': obj.box[3]})
        if region_attributes is False:
            text = {'text': '{}_{}'.format(class_name, str(np.round(obj.score, 4)))}
        else:
            text = {'Name': class_name, 'Score': str(obj.score),
                    'Location': obj.base_location, 'Source': 'Original'}
        attr_dict = json.dumps(text)
        my_dict = {"#filename": ai_scan.image_name, "file_size": ai_scan.image_size,
                   "file_attributes": "{}",
                   "region_count": element_count, "region_id": box_count,
                   "region_shape_attributes": shape_dict, "region_attributes": attr_dict}
        box_count += 1
        self.data_for_save.append(my_dict)
        self.data_main_for_save.append(my_dict)
        if center_annotation is True:
            shape_dict = json.dumps({'name': 'rect', 'x': obj.center[0], 'y': obj.center[1],
                                     'width': 1,
                                     'height': 1})
            if region_attributes is False:
                text = {'text': 'center_{}'.format(class_name)}
            else:
                text = {'Name': 'center_{}'.format(class_name),
                        'Location': obj.base_location}
            attr_dict = json.dumps(text)
            my_dict = {"#filename": ai_scan.image_name, "file_size": ai_scan.image_size,
                       "file_attributes": "{}",
                       "region_count": element_count, "region_id": box_count,
                       "region_shape_attributes": shape_dict, "region_attributes": attr_dict}
            box_count += 1
            self.data_for_save.append(my_dict)
            self.data_main_for_save.append(my_dict)
        return box_count

    @staticmethod
    def get_via_header():
        return ['#filename', 'file_size', 'file_attributes', 'region_count', 'region_id',
                'region_shape_attributes', 'region_attributes']
