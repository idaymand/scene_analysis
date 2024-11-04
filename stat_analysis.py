import csv
import json
import os
import shutil
from distutils.util import strtobool

import cv2
import numpy as np
from ravin_utils.storage_utils.s3_tools import S3Handler
from scan_save_data import ScanSaveData

from scene_analysis_lib.ai_scans_processing import AIScansProcessing
from scene_analysis_lib.detectable_object import DetectionContainer, DetectedObject
from scene_analysis_lib.estimate_angles_FE_model import init_model, TypeModel, init_model_image
from metadataHandler import View

from PIL import Image

from scene_analysis_lib.ai_scan_metadata import AIScanMetadata, Scene
from scene_analysis_lib.json_configuration import Configurator


def sort_images(image_score):
    for key in image_score.keys():
        return image_score[key]


class StatAnalysis:
    def __init__(self, data_path, model_full_file_name=None,
                 input_sub_folder='csv'):
        self.data_path = data_path
        self.data_container = {}
        self.configurator = Configurator()
        self.model_full_file_name = model_full_file_name
        self.input_sub_folder = input_sub_folder
        self.global_path = './output'
        self.scan_data = []
        self.output_scan_via_path = 'annotations'
        self.model = None
        self.image_model_config = '{}.json'.format(os.getenv('IMAGE_MODEL_NAME', ''))
        if os.path.exists(self.image_model_config):
            with open(self.image_model_config, 'r') as f_config:
                self.configurator.model_config = json.load(f_config)
        self.use_image_model = bool(strtobool(os.getenv('USE_IMAGE_MODEL', '0')))
        if self.use_image_model is True:
            self.model = init_model_image(os.getenv('IMAGE_MODEL_NAME', ''))
        else:
            if self.model_full_file_name is not None:
                self.model = init_model(model_full_file_name)
        self.body_type_model = init_model_image(os.getenv('BODY_TYPE_MODEL_NAME', ''))
        self.scans_processing = AIScansProcessing(self.configurator)
        self.scan_save_data = ScanSaveData(self.global_path, self.configurator)

    def make_folder_output_data(self, global_path):
        if len(self.data_container) == 0:
            return
        for scene in self.data_container:
            if os.path.exists(os.path.join(global_path, scene)):
                shutil.rmtree(os.path.join(global_path, scene))
            os.makedirs(os.path.join(global_path, scene), exist_ok=True)

    def beauty_shots_analysis(self):
        # WALKAROUND_LEFT_SIDE = 268,
        # WALKAROUND_LEFT_REAR_SIDE = 269,
        # WALKAROUND_REAR_SIDE = 270,
        # WALKAROUND_RIGHT_REAR_SIDE = 271,
        # WALKAROUND_RIGHT_SIDE = 272,
        # WALKAROUND_RIGHT_FRONT_SIDE = 273,
        # WALKAROUND_FRONTSIDE = 274,
        # WALKAROUND_LEFT_FRONT_SIDE = 275,
        self.data_container = {'268': [],
                               '269': [],
                               '270': [],
                               '271': [],
                               '272': [],
                               '273': [],
                               '274': [],
                               '275': []}
        self.make_folder_output_data(self.global_path)
        self.register_via_flow_data()
        self.save_data()

    def beauty_shots_analysis(self):
        # WALKAROUND_LEFT_SIDE = 268,
        # WALKAROUND_LEFT_REAR_SIDE = 269,
        # WALKAROUND_REAR_SIDE = 270,
        # WALKAROUND_RIGHT_REAR_SIDE = 271,
        # WALKAROUND_RIGHT_SIDE = 272,
        # WALKAROUND_RIGHT_FRONT_SIDE = 273,
        # WALKAROUND_FRONTSIDE = 274,
        # WALKAROUND_LEFT_FRONT_SIDE = 275,
        self.data_container = {'268': [],
                               '269': [],
                               '270': [],
                               '271': [],
                               '272': [],
                               '273': [],
                               '274': [],
                               '275': []}
        self.make_folder_output_data(self.global_path)
        self.register_via_flow_data()
        self.save_data()

    def interior_analysis(self):
        # VIN = 150,
        # TIRE_STICKER = 195,
        # ODOMETER = 151,
        # DASHBOARD = 201,
        # SCREEN = 202,
        # KEYS_AND_MANUAL = 196,
        # FRONT_SEATS = 203,
        # REAR_SEATS = 204,
        # KEYS = 205,
        # TRUNK = 197,
        # SPARE_WHEEL = 206,
        # UNDERCARRIAGE_REAR = 198,
        # UNDERCARRIAGE_FRONT = 199,
        # ENGINE = 200,
        self.make_folder_output_data(self.global_path)
        self.data_container = {
            '150': [],
            '195': [],
            '151': [],
            '201': [],
            '202': [],
            '196': [],
            '203': [],
            '204': [],
            '205': [],
            '197': [],
            '206': [],
            '198': [],
            '199': [],
            '200': []
        }
        self.register_flow_data()
        self.save_data()

    def damage_localization_analysis(self):
        self.register_flow_history_data()
        self.scan_save_data.save_all_scan_data(self.scans_processing.scan_data)

    def register_flow_data(self):
        dir_name = os.path.dirname(self.data_path)
        dir_name_csv = os.path.join(dir_name, self.input_sub_folder)
        with open(self.data_path, 'r') as fr:
            lines = fr.readlines()
        for line in lines:
            line = line.replace(',\n', '')
            base_name_csv = '{}.{}'.format(os.path.basename(line), 'csv')
            ai_scan = AIScanMetadata(self.configurator, self.model)
            ai_scan.process_parts_from_file(base_name_csv, dir_name, dir_name_csv)
            best_scene, scenes = ai_scan.scan_image()
            if best_scene is not None and scenes is not None:
                for scene in scenes:
                    for key in scene.keys():
                        if self.data_container.get(key, None) is not None:
                            list_images = self.data_container.get(key)
                            parts_score = ai_scan.bestCar.get_scene_part_score(key)
                            angle_score = ai_scan.bestCar.get_scene_angle_score(key)
                            list_images.append({os.path.basename(line): [scene.get(key), parts_score, angle_score]})
            print('file {} was processed'.format(os.path.basename(line)))

    def register_flow_history_data(self):
        dir_name = os.path.dirname(self.data_path)
        dir_name_csv = os.path.join(dir_name, self.input_sub_folder)
        with open(self.data_path, 'r') as fr:
            lines = fr.readlines()
        self.scans_processing.scan_data = []
        self.scans_processing.processing_list_of_csv_data(lines, dir_name, dir_name_csv, True)

    def make_output_folders_for_remarketing(self, event_id, remove_images=True):
        if os.path.exists(os.path.join(self.global_path, '{}'.format(str(event_id)))) and remove_images is True:
            shutil.rmtree(os.path.join(self.global_path, '{}'.format(str(event_id))))
        os.makedirs(os.path.join(self.global_path, '{}'.format(str(event_id))), exist_ok=True)
        if os.path.exists(os.path.join(self.global_path, str(event_id),
                                       self.scan_save_data.output_scan_via_path)):
            shutil.rmtree(os.path.join(self.global_path, '{}'.format(str(event_id)),
                                       self.scan_save_data.output_scan_via_path))
        os.makedirs(os.path.join(self.global_path, str(event_id),
                                 self.scan_save_data.output_scan_via_path), exist_ok=True)
        if os.path.exists(os.path.join(self.global_path, str(event_id),
                                       self.scan_save_data.input_scan_via_path)):
            shutil.rmtree(os.path.join(self.global_path, '{}'.format(str(event_id)),
                                       self.scan_save_data.input_scan_via_path))
        os.makedirs(os.path.join(self.global_path, str(event_id),
                                 self.scan_save_data.input_scan_via_path), exist_ok=True)
        if os.path.exists(os.path.join(self.global_path, str(event_id),
                                       self.scan_save_data.output_main_scan_via_path)):
            shutil.rmtree(os.path.join(self.global_path, '{}'.format(str(event_id)),
                                       self.scan_save_data.output_main_scan_via_path))
        os.makedirs(os.path.join(self.global_path, str(event_id),
                                 self.scan_save_data.output_main_scan_via_path), exist_ok=True)

    def processing_data_analysis(self, rows_data, event_id, remove_images, remote_storage):
        parts = self.configurator.data_configurator.get('ALL_MAPPING', {}).get('backward', {}).get('detectedPart', {})
        ai_scan = AIScanMetadata(self.configurator, model=self.model)
        ai_scan.body_type_model = self.body_type_model
        ai_scan.use_image_model = self.use_image_model
        file_name = rows_data[0]['path']
        file_size = int(rows_data[0]['file_size'])
        image_width = int(rows_data[0]['image_width'])
        image_height = int(rows_data[0]['image_height'])
        if event_id == 0 or event_id != int(rows_data[0]['eventid']):
            event_id = int(rows_data[0]['eventid'])
            self.make_output_folders_for_remarketing(event_id, remove_images)
        ai_scan.image_size = file_size
        ai_scan.width = image_width
        ai_scan.height = image_height
        if image_width < 0 or image_height < 0:
            return None, None
        base_name = os.path.basename(file_name)
        dst_path = os.path.join(self.global_path, '{}'.format(str(event_id)), base_name)
        ai_scan.image_name = base_name
        if not os.path.exists(dst_path):
            remote_storage.download(awsFileName=file_name, fileLocal=dst_path,
                                    aws_blobName=os.getenv('S3_BUCKET', ''))
        ai_scan.image = cv2.imread(dst_path)
        if ai_scan.image_size == 0:
            ai_scan.image_size = os.path.getsize(dst_path)
        for row_data in rows_data:
            class_id = parts.get(row_data['part_name'])
            left = int(row_data['left'])
            top = int(row_data['top'])
            width = int(row_data['width'])
            height = int(row_data['height'])
            score = float(row_data['score'])
            obj = DetectedObject(class_id, [left, top, width, height], score)
            ai_scan.detections.append(obj)
        return ai_scan, event_id

    def analysis_full_event(self, remove_images=True):
        remote_storage = S3Handler()
        processing = AIScansProcessing(self.configurator, self.model)
        ai_scan_prev = None
        with open(os.path.join(self.data_path), 'r') as f:
            csv_dict_reader = csv.DictReader(f)
            file_name = ''
            movie_name = ''
            rows_data = []
            event_id = 0
            for row in csv_dict_reader:
                if movie_name == '':
                    movie_name = os.path.basename(file_name).split('_')[0]
                if file_name == '' or file_name == row['path']:
                    rows_data.append(row)
                    file_name = row['path']
                else:
                    if movie_name != os.path.basename(file_name).split('_')[0]:
                        ai_scan_prev = None
                        movie_name = os.path.basename(file_name).split('_')[0]
                    ai_scan, event_id = self.processing_data_analysis(rows_data, event_id, remove_images,
                                                                      remote_storage)
                    ai_scan = processing.history_scene_analysis(ai_scan, ai_scan_prev, False)
                    if ai_scan is not None:
                        ai_scan_prev = ai_scan
                        if ai_scan.damage_object is not None and ai_scan.best_scene is not None:
                            print(ai_scan.damage_object.damage_type, ai_scan.damage_object.part_name,
                                  ai_scan.damage_object.part_location, ai_scan.damage_object.viewType,
                                  ai_scan.best_scene.scene_id)
                        if ai_scan.best_scene is not None:
                            print('file {} was processed, scene {}'.format(os.path.basename(file_name),
                                                                           ai_scan.best_scene.scene_id))
                        else:
                            print('file {} was processed'.format(os.path.basename(file_name)))
                    file_name = row['path']
                    rows_data = [row]
                    self.scan_save_data.output_sub_folder = str(event_id)
                    self.scan_save_data.save_scan_detections(ai_scan)
                    self.scan_save_data.save_scan_data(ai_scan)

    def analysis_remarketing(self, remove_images=True):
        number_scenes = self.scans_processing.number_scenes
        remote_storage = S3Handler()
        # WALKAROUND_LEFT_SIDE = 268,
        # WALKAROUND_LEFT_REAR_SIDE = 269,
        # WALKAROUND_REAR_SIDE = 270,
        # WALKAROUND_RIGHT_REAR_SIDE = 271,
        # WALKAROUND_RIGHT_SIDE = 272,
        # WALKAROUND_RIGHT_FRONT_SIDE = 273,
        # WALKAROUND_FRONTSIDE = 274,
        # WALKAROUND_LEFT_FRONT_SIDE = 275,
        self.data_container = {'268': {'score': 0, 'image': ''},
                               '269': {'score': 0, 'image': ''},
                               '270': {'score': 0, 'image': ''},
                               '271': {'score': 0, 'image': ''},
                               '272': {'score': 0, 'image': ''},
                               '273': {'score': 0, 'image': ''},
                               '274': {'score': 0, 'image': ''},
                               '275': {'score': 0, 'image': ''}
                               }
        with open(os.path.join(self.data_path), 'r') as f:
            csv_dict_reader = csv.DictReader(f)
            file_name = ''
            rows_data = []
            body_type_data = []
            event_id = 0
            prev_event_id = 0
            for row in csv_dict_reader:
                if file_name == '' or file_name == row['path']:
                    rows_data.append(row)
                    file_name = row['path']
                else:
                    ai_scan, event_id = self.processing_data_analysis(rows_data, event_id, remove_images,
                                                                      remote_storage)
                    if prev_event_id == 0:
                        prev_event_id = event_id
                    if prev_event_id != event_id:
                        self.scan_save_data.save_all_scan_data(self.scan_data, self.data_container)
                        prev_event_id = event_id
                        self.data_container = {'268': {'score': 0, 'image': ''},
                                               '269': {'score': 0, 'image': ''},
                                               '270': {'score': 0, 'image': ''},
                                               '271': {'score': 0, 'image': ''},
                                               '272': {'score': 0, 'image': ''},
                                               '273': {'score': 0, 'image': ''},
                                               '274': {'score': 0, 'image': ''},
                                               '275': {'score': 0, 'image': ''}
                                               }
                        self.scan_data = []
                    old_output_scan_via_path = self.output_scan_via_path
                    self.scan_save_data.output_scan_via_path = os.path.join(str(event_id),
                                                                            self.output_scan_via_path)
                    ai_scan.output_sub_folder = str(event_id)
                    ai_scan.output_path = self.scan_save_data.output_scan_via_path
                    self.scan_data.append(ai_scan)
                    if ai_scan is not None:
                        best_scene, scenes = ai_scan.scan_image(number_scenes)
                        # self.table_distance_scan_detections.append(
                        #     {'image_name': ai_scan.image_name,
                        #      'table_distances': self.set_table_distance_scan_detections(ai_scan)})
                        best_scene, scenes = ai_scan.clarification_scene_candidates_score(best_scene, scenes)
                        if best_scene is not None:
                            scene_id, scene_score = list(best_scene.keys())[0], list(best_scene.values())[0]
                            current_scene = Scene(scene_id, scene_score,
                                                  ai_scan.configurator)
                            current_scene.best_car = ai_scan.bestCar
                            ai_scan.best_scene = current_scene
                            ai_scan.bestCar.find_virtual_parts(ai_scan.best_scene)
                            ai_scan.bestCar.update_part_location(current_scene)
                            for index in range(0, len(scenes)):
                                scene = Scene(list(scenes[index].keys())[0], list(scenes[index].values())[0],
                                              ai_scan.configurator)
                                if str(scene.scene_id) in self.data_container.keys():
                                    if self.data_container[str(scene.scene_id)]['score'] < \
                                            scene.score:
                                        self.data_container[str(scene.scene_id)]['score'] = \
                                            scene.score
                                        self.data_container[str(scene.scene_id)]['image'] = \
                                            ai_scan.image_name
                                        self.data_container[str(scene.scene_id)]['scan'] = ai_scan
                        #                       self.scan_save_data.save_scan_data(ai_scan)
                        base_name = os.path.basename(file_name)
                        if ai_scan.best_scene is not None:
                            print('eventid {}, image {} processed, scene {}'.format(str(event_id), base_name,
                                                                                    str(ai_scan.best_scene.scene_id)))
                        else:
                            print('eventid {}, image {} processed, scene {}'.format(str(event_id), base_name,
                                                                                    'None'))
                        bt_data = {"eventid": event_id,
                                   "imagename": base_name,
                                   "car_x": ai_scan.bestCar.center[0],
                                   "car_width": ai_scan.bestCar.box[2],
                                   "direction": ai_scan.params[ai_scan.bestCar.viewType].get("BodyTypeDirection"),
                                   "score_direction": ai_scan.params[ai_scan.bestCar.viewType].get(
                                       "BodyTypeDirectionConf"),
                                   "body_type": ai_scan.params[ai_scan.bestCar.viewType].get("BodyType"),
                                   "score_body_type": ai_scan.params[ai_scan.bestCar.viewType].get("BodyTypeConf")
                                   }
                        body_type_data.append(bt_data)
                    self.scan_save_data.output_scan_via_path = old_output_scan_via_path
                    file_name = row['path']
                    rows_data = [row]
        self.scan_save_data.save_all_scan_data(self.scan_data, self.data_container, only_original=False)
        self.scan_save_data.save_body_type_data(body_type_data, event_id=event_id)

    def analysis_virtual_parts(self, remove_images=True):
        remote_storage = S3Handler()
        with open(os.path.join(self.data_path), 'r') as f:
            csv_dict_reader = csv.DictReader(f)
            file_name = ''
            rows_data = []
            event_id = 0
            prev_event_id = 0
            for row in csv_dict_reader:
                if file_name == '' or file_name == row['path']:
                    rows_data.append(row)
                    file_name = row['path']
                else:
                    ai_scan, event_id = self.processing_data_analysis(rows_data, event_id, remove_images,
                                                                      remote_storage)
                    if prev_event_id == 0:
                        prev_event_id = event_id
                    if prev_event_id != event_id:
                        self.scan_save_data.save_all_scan_data(self.scan_data, self.data_container)
                        prev_event_id = event_id
                        self.scan_data = []
                    old_output_scan_via_path = self.output_scan_via_path
                    self.scan_save_data.output_scan_via_path = os.path.join(str(event_id),
                                                                            self.output_scan_via_path)
                    ai_scan.output_sub_folder = str(event_id)
                    ai_scan.output_path = self.scan_save_data.output_scan_via_path
                    self.scan_data.append(ai_scan)
                    if ai_scan is not None:
                        base_name = os.path.basename(file_name)
                        best_scene, scenes = ai_scan.scan_image()
                        if best_scene is not None:
                            scene_id, scene_score = list(best_scene.keys())[0], list(best_scene.values())[0]
                            current_scene = Scene(scene_id, scene_score,
                                                  ai_scan.configurator)
                            current_scene.best_car = ai_scan.bestCar
                            ai_scan.best_scene = current_scene
                            ai_scan.bestCar.find_virtual_parts(ai_scan.best_scene)
                            ai_scan.bestCar.update_part_location(current_scene)
                        if ai_scan.best_scene is not None:
                            print('eventid {}, image {} processed, scene {}'.format(str(event_id), base_name,
                                                                                    str(ai_scan.best_scene.scene_id)))
                        else:
                            print('eventid {}, image {} processed, scene {}'.format(str(event_id), base_name,
                                                                                    'None'))
                    self.scan_save_data.output_scan_via_path = old_output_scan_via_path
                    file_name = row['path']
                    rows_data = [row]
        self.scan_save_data.save_all_scan_data(self.scan_data, self.data_container)

    def processing_event_save_data(self, event):
        annotations = os.path.join(os.environ.get('ANNOTATIONS', ''))
        self.scans_processing.scan_data = []
        self.scans_processing.processing_event(event)
        if int(os.environ.get('VIRTUAL_PARTS_DEBUG', 0)):
            for str1, str2, view in event.viewIterator():
                data = view.generateCSVcontent(str2)
                dst_file_name = os.path.join(annotations, str2 + '.csv')
                with open(dst_file_name, 'w') as f:  # Just use 'w' mode in 3.x
                    w = csv.DictWriter(f, View.getHeader(), delimiter=',', lineterminator='\n')
                    w.writeheader()
                    w.writerows(data)
                print('{} is processed'.format(str2))

    def register_via_flow_data(self):
        len_list_image = 10
        data_rows = []
        with open(self.data_path, 'r') as fr:
            data = csv.DictReader(fr)
            for row in data:
                data_rows.append(row)
        index = 0
        while index < len(data_rows):
            row = data_rows[index]
            region_count = int(row.get('region_count', 0))
            if region_count > 0:
                file_name = row.get('#filename', '')
                if os.path.isfile(os.path.join(os.path.dirname(self.data_path), file_name)):
                    i_width, i_height = Image.open(os.path.join(os.path.dirname(self.data_path), file_name)).size
                ai_scan = AIScanMetadata(self.configurator, self.model, i_width=i_width, i_height=i_height)
                ai_scan.process_parts_from_data_dict(data_rows[index:index + region_count])
                index = index + region_count
                best_scene, scenes = ai_scan.scan_image()
                if best_scene is not None and scenes is not None:
                    for scene in scenes:
                        for key in scene.keys():
                            if self.data_container.get(key, None) is not None:
                                list_images = self.data_container.get(key)
                                parts_score = ai_scan.bestCar.get_scene_part_score(key)
                                angle_score = ai_scan.bestCar.get_scene_angle_score(key)
                                list_images.append({file_name: [scene.get(key), parts_score, angle_score]})
                                if len(list_images) > len_list_image:
                                    list_images.sort(key=sort_images, reverse=True)
                                    self.data_container[key] = list_images[0:len_list_image]
                print('file {} was processed'.format(file_name))

    def save_data(self):
        dir_name = os.path.dirname(self.data_path)
        for scene in self.data_container:
            list_images = self.data_container[scene]
            list_images.sort(key=sort_images, reverse=True)
            with open(os.path.join(self.global_path, scene, '{}.csv'.format(scene)), 'w') as file:
                writer = csv.DictWriter(file, fieldnames=['file_name', 'score', 'parts_score', 'angle_score'])
                writer.writeheader()
                for image in list_images[0:8]:
                    for key, value in image.items():
                        shutil.copyfile(os.path.join(dir_name, key),
                                        os.path.join(self.global_path, scene, key))
                        row = {'file_name': key, 'score': value[0], 'parts_score': value[1], 'angle_score': value[2]}
                        writer.writerow(row)
                print('csv for {} saved'.format(scene))

    def register_fe_data(self):
        dir_name = os.path.dirname(self.data_path)
        dir_name_view_csv = os.path.join(dir_name, 'view_csv')
        with open(self.data_path, 'r') as fr:
            lines = fr.readlines()
        for line in lines:
            line = line.replace(',\n', '')
            base_name_csv = '{}_{}.{}'.format(os.path.basename(line).split('.jpg')[0], 'view.jpg', 'csv')
            if os.path.isfile(os.path.join(dir_name_view_csv, base_name_csv)):
                with open(os.path.join(dir_name_view_csv, base_name_csv), encoding="utf-8") as file:
                    data_view = csv.DictReader(file)
                    for row_view in data_view:
                        provider = row_view.get('provider')
                        real_score = row_view.get('real_score', 0)
                        if real_score is None or real_score == '':
                            real_score = 0
                        parts = row_view.get('parts_score')
                        angle_sc = row_view.get('angle_score', 0)
                        if angle_sc is None or angle_sc == '':
                            angle_sc = 0
                        if self.data_container.get(str(provider), None) is not None:
                            list_images = self.data_container.get(str(provider))
                            list_images.append({os.path.basename(line): [real_score, parts, angle_sc]})
            print('file {} was processed'.format(os.path.basename(line)))
