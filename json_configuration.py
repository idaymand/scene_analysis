import json


class Configurator:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file_configurator = open(self.file_name)
        self.data_configurator = json.load(self.file_configurator)

    def get_object_detection_constants(self):
        ret_data = self.data_configurator['ObjectDetectionConstants']
        return ret_data

    def get_object_relates_to(self):
        ret_data = self.data_configurator['objectRelatesTo']
        return ret_data

    def get_object_constructed_from(self):
        ret_data = self.data_configurator['objectConstructedFrom']
        return ret_data

    def get_view_directions(self):
        ret_data = self.data_configurator['viewDirections']
        return ret_data

    def get_angle_lut(self):
        ret_data = self.data_configurator['ANGLE_LUT']
        return ret_data

    def get_index_lut(self):
        ret_data = self.data_configurator['INDEX_LUT']
        return ret_data

    def get_rev_index_lut(self):
        ret_data = self.data_configurator['REV_INDEX_LUT']
        return ret_data

    def get_classes(self):
        ret_data = self.data_configurator['CLASSES']
        return ret_data

    def get_scene_parts_consts(self):
        ret_data = self.data_configurator['scenePartsConsts']
        return ret_data

    def get_angle_models(self):
        ret_data = self.data_configurator['ObjectDetectionConstants']['angleModels']
        return ret_data

    def get_view_type(self):
        ret_data = self.data_configurator['ViewDescription']
        return ret_data

    def get_parts(self):
        ret_data = self.data_configurator['PARTS']
        return ret_data
