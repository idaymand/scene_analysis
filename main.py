import csv
import json
import os.path
from distutils.util import strtobool

import cv2
import numpy as np
import pkg_resources
from PIL import Image

from scene_analysis_lib.ai_scans_processing import AIScansProcessing
from scene_analysis_lib.detectable_object import DetectedObject

from scene_analysis_lib.ai_scan_metadata import Scene
from scan_save_data import ScanSaveData
from scene_analysis_lib.ai_scan_metadata import AIScanMetadata
from scene_analysis_lib.estimate_angles_FE_model import init_model, init_model_image
from scene_analysis_lib.json_configuration import Configurator
# from libs.base_lib.pose_features import pose_features_hv
from metadataHandler import View
from stat_analysis import StatAnalysis


def get_best_scene1(input_file_name, path_to_image, path_to_csv):
    configurator = Configurator()
    model = init_model(r'/Users/innadaymand/Projects/scene-analysis/model/converged_7params_wide_gaus_10001.h5')
    scan = AIScanMetadata(configurator, model)
    scan.process_parts_from_file(input_file_name, path_to_image, path_to_csv)
    best_scene, scenes = scan.scan_image()
    print(best_scene)
    print(scenes)


def get_best_scene(input_file_name, image_file_name):
    configurator = Configurator()
    model = init_model(r'/Users/innadaymand/Documents/Data/P.2.3.1/P.2.3.1.h5')
    scan = AIScanMetadata(configurator, model)
    with open(input_file_name, 'r') as f:
        dict_data = json.load(f)
    view = View(dict_data)
    #    pose_features_hv(dict_data, [])
    view.image_name = os.path.basename(image_file_name)
    view.n_x, view.n_y = Image.open(image_file_name).size
    view.file_size = os.path.getsize(image_file_name)
    scan.process_parts_from_view_metadata(view)
    best_scene, scenes = scan.scan_best_car()
    print(best_scene)
    print(scenes)


def get_virtual_parts_via_view(input_file_name, image_file_name):
    configurator = Configurator()
    model = init_model(r'/Users/innadaymand/Documents/Data/P.2.3.1/P.2.3.1.h5')
    scan = AIScanMetadata(configurator, model)
    scan_save_data = ScanSaveData('./output', configurator)
    view = View(input_file_name)
    scan.process_parts_from_view_metadata(view)
    best_scene, scenes = scan.scan_best_car()
    if best_scene is not None:
        key, value = list(best_scene.items())[0]
        best_scene = {int(key): round(value, 3)}
        current_scene = Scene(list(best_scene.keys())[0], list(best_scene.values())[0],
                              scan.configurator.get_object_constructed_from())
        current_scene.best_car = scan.bestCar
        scan.best_scene = current_scene
        scan.bestCar.update_virtual_parts(scan.best_scene, None, None)
    scan_save_data.save_scan_data(scan)
    print(best_scene)
    print(scenes)


def get_data_from_file(input_file_name, scan):
    if os.path.isfile(input_file_name.replace('.csv', '')):
        scan.width, scan.height = Image.open(input_file_name.replace('.csv', '')).size
        scan.image_name = os.path.basename(input_file_name.replace('.csv', ''))
        scan.image_size = os.path.getsize(input_file_name.replace('.csv', ''))
    with open(input_file_name, 'r') as fr:  # Just use 'w' mode in 3.x
        data = csv.DictReader(fr)
        for row in data:
            left = float(row['xmin'])
            top = float(row['ymin'])
            right = float(row['xmax'])
            bottom = float(row['ymax'])
            box = [left, top, right - left, bottom - top]
            score = 1.0
            class_id = int(scan.configurator.get_parts().get(row['class_name'].upper(),
                                                             scan.configurator.get_parts().get(
                                                                 scan.configurator.align_config.get(
                                                                     row['class_name'].upper(), 1))))
            part = DetectedObject(class_id, box, score)
            scan.detections.append(part)


def get_virtual_parts(input_file_name):
    configurator = Configurator()
    angle_model_path = \
        pkg_resources.resource_filename('scene_analysis_lib',
                                        os.path.join('model', os.environ.get('ANGLE_MODEL', 'P.2.3.1.h5')))
    model = init_model(angle_model_path)
    scan_save_data = ScanSaveData('./output', configurator)
    scan = AIScanMetadata(configurator, model)
    scan.process_parts_from_file(os.path.basename(input_file_name), os.path.dirname(input_file_name),
                                 os.path.dirname(input_file_name))
    # get_data_from_file(input_file_name, scan)
    best_scene, scenes = scan.scan_image()
    if best_scene is not None:
        key, value = list(best_scene.items())[0]
        best_scene = {int(key): round(value, 3)}
        current_scene = Scene(list(best_scene.keys())[0], list(best_scene.values())[0],
                              scan.configurator.get_object_constructed_from())
        current_scene.best_car = scan.bestCar
        scan.best_scene = current_scene
        scan.bestCar.update_virtual_parts(scan.best_scene, None, None)
        for key, v_parts in scan.bestCar.virtualPartMap.mapData.items():
            for part in v_parts:
                print(part.box, configurator.get_classes().get(key, {}).get('name', ''))
    scan_save_data.save_scan_data(scan)
    print(best_scene)
    print(scenes)


def get_virtual_parth_from_via_file(image_path, csv_file_path):
    data_rows = []
    with open(csv_file_path, 'r') as fr:
        data = csv.DictReader(fr)
        for row in data:
            data_rows.append(row)
    configurator = Configurator()
    model_name = os.getenv('IMAGE_MODEL_NAME', None)
    use_image_model = bool(strtobool(str(os.getenv('USE_IMAGE_MODEL', 0))))
    if use_image_model is True and model_name is not None:
        model_config_name = '{}.json'.format(model_name)
        if os.path.exists(os.path.join(model_name, model_config_name)):
            with open(os.path.join(model_name, model_config_name), 'r') as f_config:
                model_config = json.load(f_config)
        else:
            model_config = None
        model = init_model_image(model_name)
    else:
        model_config = None
        angle_model_path = \
            pkg_resources.resource_filename('scene_analysis_lib',
                                            os.path.join('model', os.environ.get('ANGLE_MODEL', 'E_9_3.4.2.h5')))
        model = init_model(angle_model_path)
    configurator.model_config = model_config
    scan_save_data = ScanSaveData('./output', configurator)
    file_name = data_rows[0].get('#filename', '')
    image = cv2.imread(os.path.join(os.path.dirname(image_path), file_name))
    i_height, i_width, _ = image.shape
    scan = AIScanMetadata(configurator, model, i_width=i_width, i_height=i_height)
    scan.image = image
    scan.image_size = os.path.getsize(os.path.join(os.path.dirname(image_path), file_name))
    scan.process_parts_from_data_dict(data_rows)
    best_scene, scenes = scan.scan_image()
    if best_scene is not None:
        key, value = list(best_scene.items())[0]
        best_scene = {int(key): round(value, 3)}
        current_scene = Scene(list(best_scene.keys())[0], list(best_scene.values())[0],
                              scan.configurator)
        current_scene.best_car = scan.bestCar
        scan.best_scene = current_scene
        scan.bestCar.update_virtual_parts(scan.best_scene, None, None)
        for key, v_parts in scan.bestCar.virtualPartMap.mapData.items():
            for part in v_parts:
                print(part.box, configurator.get_parts().get(key, ''))
    scan_save_data.save_scan_data(scan)
    print(best_scene)
    print(scenes)


def get_virtual_parts_flow(input_file_name):
    configurator = Configurator()
    angle_model_path = \
        pkg_resources.resource_filename('scene_analysis_lib',
                                        os.path.join('model', os.environ.get('ANGLE_MODEL', 'P.2.3.1.h5')))
    model = init_model(angle_model_path)
    scan_save_data = ScanSaveData('./output', configurator)
    scan = AIScanMetadata(configurator, model)
    scan.process_parts_from_file(os.path.basename(input_file_name), os.path.dirname(input_file_name),
                                 os.path.dirname(input_file_name))
    # get_data_from_file(input_file_name, scan)
    best_scene, scenes = scan.scan_image()
    if best_scene is not None:
        key, value = list(best_scene.items())[0]
        best_scene = {int(key): round(value, 3)}
        current_scene = Scene(list(best_scene.keys())[0], list(best_scene.values())[0],
                              scan.configurator)
        current_scene.best_car = scan.bestCar
        scan.best_scene = current_scene
        scan.bestCar.update_virtual_parts(scan.best_scene, None, None)
        for key, v_parts in scan.bestCar.virtualPartMap.mapData.items():
            for part in v_parts:
                print(part.box, configurator.get_classes().get(key, {}).get('name', ''))
    scan_save_data.save_scan_data(scan)
    print(best_scene)
    print(scenes)


def get_virtual_parts_with_damage(input_file_name):
    configurator = Configurator()
    angle_model_path = \
        pkg_resources.resource_filename('scene_analysis_lib',
                                        os.path.join('model', os.environ.get('ANGLE_MODEL', 'P.2.3.1.h5')))
    model = init_model(angle_model_path)
    scan_save_data = ScanSaveData('./output', configurator)
    scan = AIScanMetadata(configurator, model)
    get_data_from_file(input_file_name, scan)
    best_scene, scenes = scan.scan_image()
    scan.find_damages()
    if best_scene is not None:
        key, value = list(best_scene.items())[0]
        best_scene = {int(key): round(value, 3)}
        current_scene = Scene(list(best_scene.keys())[0], list(best_scene.values())[0],
                              scan.configurator.get_object_constructed_from())
        current_scene.best_car = scan.bestCar
        scan.best_scene = current_scene
        scan.bestCar.update_virtual_parts(scan.best_scene, None, None)
        for key, v_parts in scan.bestCar.virtualPartMap.mapData.items():
            for part in v_parts:
                print(part.box, configurator.get_classes().get(key, {}).get('name', ''))
    scan_save_data.save_scan_data(scan)
    print(best_scene)
    print(scenes)


def get_beauty_shots_data(file_data, model_file_name):
    beauty_shots_analysis = StatAnalysis(file_data, model_file_name)
    beauty_shots_analysis.beauty_shots_analysis()


def get_interior_data(file_data, model_file_name):
    interior_analysis = StatAnalysis(file_data, model_file_name)
    interior_analysis.interior_analysis()


def get_damage_data(file_data, model_file_name, input_sub_folder='csv',
                    output_sub_folder='annotations'):
    damage_analysis = StatAnalysis(file_data, model_file_name, input_sub_folder,
                                   output_sub_folder)
    damage_analysis.damage_localization_analysis()


def remarketing_analysis(file_data, model_file_name=None):
    remarketing = StatAnalysis(file_data,
                               model_full_file_name=model_file_name)
    remarketing.analysis_remarketing(False)


def find_roi_for_picture(input_file_name):
    configurator = Configurator()
    angle_model_path = \
        pkg_resources.resource_filename('scene_analysis_lib',
                                        os.path.join('model', os.environ.get('ANGLE_MODEL', 'E_9_3.4.2.h5')))
    model = init_model(angle_model_path)
    scan_save_data = ScanSaveData('./output', configurator)
    scan = AIScanMetadata(configurator, model)
    scan.process_parts_from_file(os.path.basename(input_file_name), os.path.dirname(input_file_name),
                                 os.path.dirname(input_file_name))
    # get_data_from_file(input_file_name, scan)
    best_scene, scenes = scan.scan_image()
    if best_scene is not None:
        key, value = list(best_scene.items())[0]
        best_scene = {int(key): round(value, 3)}
        current_scene = Scene(list(best_scene.keys())[0], list(best_scene.values())[0],
                              scan.configurator)
        current_scene.best_car = scan.bestCar
        scan.best_scene = current_scene
        scan.bestCar.update_virtual_parts(scan.best_scene, None, None)
        for key, v_parts in scan.bestCar.virtualPartMap.mapData.items():
            for part in v_parts:
                print(part.box, configurator.get_classes().get(key, {}).get('name', ''))
        parts_roi = scan.find_roi()
        rect = [370, 770, 100, 52]
        obj, iou = scan.bestCar.get_rectangle_location(rect)
        print(obj.class_id, obj.box)
    scan_save_data.save_scan_data(scan)
    print(best_scene)
    print(scenes)
    print(parts_roi)


def find_roi_in_view(view):
    ai_scans_processing = AIScansProcessing()
    scan_metadata = ai_scans_processing.processing_view(view)
    parts_roi = scan_metadata.find_roi()
    print(parts_roi)


def virtual_parts_analysis(file_data, model_file_name=None):
    modelling = StatAnalysis(file_data,
                             model_full_file_name=model_file_name)
    modelling.analysis_virtual_parts(False)


def analysis_full_event(file_data, model_file_name):
    analysis = StatAnalysis(file_data, model_file_name)
    analysis.analysis_full_event(False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get_best_scene(r'65_27.27.jpg.csv', r'/Users/innadaymand'
    #                                     r'/PycharmProjects/service-ravin'
    #                                     r'-monorepo/apps/algo/src/assets'
    #                                     r'/algo-test/tfs-488', r'/Users/innadaymand/PycharmProjects/service-ravin'
    #                                                            r'-monorepo/apps/algo/src'
    #                                                            r'/assets/algo-test/tfs-488/csv')

    # get_best_scene(r'/Users/innadaymand/Documents/download/to_inna/metadata.json',
    #                r'/Users/innadaymand/Documents/download/to_inna/65_0.650000.jpg')
    # get_best_scene1('65_20.20.jpg.csv',
    #                 '/Users/innadaymand/PycharmProjects/service-ravin-monorepo/apps/algo/src/assets/algo-test/tfs-488',
    #                 '/Users/innadaymand/PycharmProjects/service-ravin-monorepo/'
    #                 'apps/algo/src/assets/algo-test/tfs-488/csv')

    # get_beauty_shots_data('/Users/innadaymand/Downloads/1885622/detection_boxes.csv',
    #                       '/Users/innadaymand/PycharmProjects/service-ravin-monorepo/models/proxy/proxy.json',
    #                       '/Users/innadaymand/PycharmProjects/scene_analysis/model/7angles.h5')

    # get_damage_data('/Users/innadaymand/Projects/ravin-monorepo/apps/algo/src'
    #                 '/assets/algo-test/prod150181/list_files.txt',
    #                 '/Users/innadaymand/Projects/scene-configurator/models/proxy/proxy.json',
    #                 '/Users/innadaymand/Projects/scene-analysis-lib/scene_analysis_lib/model/P.2.3.1.h5',
    #                 'csv5.90'
    #                 )

    remarketing_analysis('/Users/innadaymand/Documents/2327976_dev_original.csv', #sompo_605_614.csv',#   dev_2314395.csv',  # qa_openlane1320489_495_497_and_more.csv',
                         # 'dev_2306689.csv',   # qa_bidacar_remarketing_2510.csv', # mtn_530594_260623.csv', #mtn_200623.csv',#qa_bidacar_remarketing_0211.csv',
                         model_file_name=pkg_resources.resource_filename('scene_analysis_lib',
                                                                         os.path.join('model', os.environ.get(
                                                                             'ANGLE_MODEL',
                                                                             'E_9_3.4.2.h5'))),
                         )

    # virtual_parts_analysis('/Users/innadaymand/Documents/dev_2218837.csv',
    #                        model_file_name=pkg_resources.resource_filename('scene_analysis_lib',
    #                                                                        os.path.join('model', os.environ.get(
    #                                                                            'ANGLE_MODEL',
    #                                                                            'E_9_3.4.2.h5'))),
    #                        )

    # analysis_full_event('/Users/innadaymand/Documents/2191621_boxes.csv',
    #                     '/Users/innadaymand/Projects/scene-configurator/models/proxy/proxy.json',
    #                     '/Users/innadaymand/Documents/Data/P.2.3.1/P.2.3.1.h5')

    # get_virtual_parts(r'/Users/innadaymand/Projects/scene-analysis/output/2191621/08_51.80051.jpgvia_region_data.csv',
    #                   r'/Users/innadaymand/Projects/scene-analysis/output/2191621/08_51.80051.jpg')

    # get_virtual_parts('/Users/innadaymand/Documents/download/Gilad_test/65_114.114.jpg.csv')

    # find_roi_for_picture('/Users/innadaymand/Documents/download/Gilad_test/65_114.114.jpg.csv')

    # find_roi_in_view()
    # get_virtual_parth_from_via_file('/Users/innadaymand/',
    #                                  '/Users/innadaymand/output/detect_annotations_160_only_parts/64_70.70.csv')
