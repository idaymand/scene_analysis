from json_configuration import Configurator


class DetectedObject:
    def __init__(self, class_id, box, score):
        self.box = []
        for number in box:
            self.box.append(number)
        self.class_id = class_id
        self.score = score
        self.area = self.box[2] * self.box[3]
        self.center = {'x': (self.box[0] * 2 + self.box[2]) / 2, 'y': (self.box[1] * 2 + self.box[3]) / 2}

    def intersect(self, obj):
        return (min((self.box[2] + self.box[0]), (obj.box[2] + obj.box[0])) - max(self.box[0], obj.box[0])) * \
               (min((self.box[3] + self.box[1]), (obj.box[3] + obj.box[1])) - max(self.box[1], obj.box[1]))

    def iou(self, obj):
        intersect = self.intersect(obj)
        return intersect / (self.area + obj.area - intersect)

    def ios(self, obj):
        intersect = self.intersect(obj)
        return intersect / max(self.area, 1)


class DetectionMap:
    def __init__(self, configurator):
        self.mapData = {}
        self.configurator = configurator
        self.classes = configurator.get_classes()

    def add_detection(self, obj):
        if self.mapData.get(str(obj.class_id), 0) == 0:
            self.mapData[str(obj.class_id)] = []

        candidates = self.mapData.get(str(obj.class_id))

        # if (CLASSES[detection.classId].merges_with) {
        #   CLASSES[detection.classId].merges_with.forEach(clsId => {
        #     if (this[clsId]) {
        #       candidates = [...candidates, ...this[clsId]];
        #     }
        #   });
        # }

        for element in candidates:
            if obj.intersect(element) > 0 and obj.score * obj.area < element.score * element.area:
                return False

        candidates.append(obj)
        return True

    def mean_score(self):
        mean_score = 0
        count = 0
        for elements in self.mapData.items():
            for item in elements:
                mean_score = item.score
                count = count + 1
        mean_score = mean_score / max(count, 1)
        return mean_score


def sort_views(view):
    for key in view.keys():
        return view[key]


class DetectionContainer(DetectedObject):
    def __init__(self, class_id, box, score, view_type, config_file_name):
        super().__init__(class_id, box, score)
        self.configurator = Configurator(config_file_name)
        self.scenePartScores = {}
        self.sceneAngleScores = {}
        self.memberParts = []
        self.partMap = DetectionMap(self.configurator)
        self.damageMap = DetectionMap(self.configurator)
        self.viewType = view_type
        object_detection_constants = self.configurator.get_object_detection_constants()
        self.list_objects = object_detection_constants['angleModels'][self.viewType]['boxes']
        self.object_constructed_from = self.configurator.get_object_constructed_from()
        self.object_relates_to = self.configurator.get_object_relates_to()
        self.scene_parts_consts = self.configurator.get_scene_parts_consts()

    def includes(self, obj):
        return self.list_objects.get(str(obj.class_id), 0)

    def add_detection(self, obj):
        if self.includes(obj):
            self.partMap.add_detection(obj)
            self.register_part(obj)

    def resister_part(self, obj):
        for scene in self.object_relates_to.get(str(obj.class_id)):
            list_parts = self.object_constructed_from[scene].parts
            if list_parts.get(obj.class_id, 0):
                valid_part = (obj.score > self.object_relates_to[obj.classId][scene].minConfidence) and \
                             (obj.box[2] / self.box[2] > self.object_relates_to[obj.classId][scene].minWidth) \
                             and (obj.box[3] / self.box[3] > self.object_relates_to[obj.classId][scene].minHeight)
                if valid_part:
                    i_unit = list_parts[obj.class_id].get('iUnit', 0.0)
                    i_max = list_parts[obj.class_id].get('iMax', 1.0)
                    try:
                        self.scenePartScores[scene][obj.classId] = \
                            min((self.scenePartScores[scene][obj.classId] + i_unit), i_max)
                    except Exception as e:
                        print(e)

    def get_scene_part_score(self, scene):
        score = 0
        for class_id in self.scenePartScores[scene]:
            score = score + self.scenePartScores[scene][class_id]
        return score

    def get_scene_angle_score(self, scene):
        return self.sceneAngleScores[scene][self.scene_parts_consts['ANGLES_TOTAL']]

    def get_top_n_scene_candidates(self, n):
        views = []
        for scene in self.scenePartScores:
            scene_score = self.get_scene_angle_score(scene) * self.get_scene_part_score(scene)
            element = {'{}'.format(scene): scene_score}
            views.append(element)
        views.sort(key=sort_views, reverse=True)

    def get_top_n_scene_candidates_only_angle(self, n):
        views = []
        for scene in self.sceneAngleScores:
            scene_score = self.get_scene_angle_score(scene)
            if 0 < scene_score < 1:
                element = {'{}'.format(scene): scene_score}
                views.append(element)
        views.sort(key=sort_views, reverse=True)
        return views[0], views[0:n]

    def register_angle(self, angle):
        angle_exists = angle.__len__()
        object_constructed_from = self.configurator.get_object_constructed_from()
        list_scenes = self.configurator.get_object_detection_constants()['angleModels'][self.viewType]['scenes']
        view_directions = self.configurator.get_view_directions()
        ANGLE_LUT = self.configurator.get_angle_lut()
        REV_INDEX_LUT = self.configurator.get_rev_index_lut()
        scene_parts_consts = self.configurator.get_scene_parts_consts()
        for scene_i in list_scenes:
            scene = str(scene_i)
            if self.sceneAngleScores.get(scene, None) is None:
                self.sceneAngleScores[scene] = {'0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
                                                scene_parts_consts['ANGLES_TOTAL']: scene_parts_consts['ANGLES_FACTOR']}
            max_conf = angle_exists + object_constructed_from[scene]['angles'].__len__()
            if object_constructed_from[scene]['angles'].__len__():
                if angle_exists:
                    for direction in object_constructed_from[scene]['angles']:
                        direction = str(direction)
                        p_conf = 1
                        n_conf = 1
                        for angle_ix_s in view_directions[direction]:
                            angle_ix_i = int(angle_ix_s)
                            p_item = 0
                            n_item = 0  # number of positive and negative data points
                            p_conf_item = 0
                            n_conf_item = 0  # integral of positive and negative data points
                            for i in range(0, angle[angle_ix_i].__len__()):
                                if view_directions[direction][angle_ix_s]['min'] > \
                                        view_directions[direction][angle_ix_s]['max']:
                                    if (ANGLE_LUT[REV_INDEX_LUT[angle_ix_i + 1]][i] <= view_directions[direction][
                                        angle_ix_s]['max'] or
                                            ANGLE_LUT[REV_INDEX_LUT[angle_ix_i + 1]][i] >= view_directions[direction][
                                                angle_ix_s]['min']):
                                        p_item += 1
                                        p_conf_item += angle[angle_ix_i][i] * angle[angle_ix_i].__len__()
                                    else:
                                        n_item += 1
                                        n_conf_item += angle[angle_ix_i][i] * angle[angle_ix_i].__len__()
                                if view_directions[direction][angle_ix_s]['min'] < \
                                        view_directions[direction][angle_ix_s]['max']:
                                    if (ANGLE_LUT[REV_INDEX_LUT[angle_ix_i + 1]][i] <= view_directions[direction][
                                        angle_ix_s]['max']) and (
                                            ANGLE_LUT[REV_INDEX_LUT[angle_ix_i + 1]][i] >=
                                            view_directions[direction][angle_ix_s]['min']):
                                        p_item += 1
                                        p_conf_item += angle[angle_ix_i][i] * angle[angle_ix_i].__len__()
                                    else:
                                        n_item += 1
                                        n_conf_item += angle[angle_ix_i][i] * angle[angle_ix_i].__len__()
                            p_conf_item /= p_item
                            n_conf_item /= n_item
                            p_conf *= p_conf_item / (p_conf_item + n_conf_item)
                            n_conf *= n_conf_item / (p_conf_item + n_conf_item)
                            nrm = p_conf + n_conf
                            p_conf /= nrm
                            n_conf /= nrm
                            self.sceneAngleScores[scene][angle_ix_s] = min(p_conf,
                                                                           self.sceneAngleScores[scene][angle_ix_s])
                        max_conf = min(max_conf, p_conf)
            self.sceneAngleScores[scene][scene_parts_consts['ANGLES_TOTAL']] = max_conf


class DetectionHistMap(DetectedObject):
    def __init__(self, class_id, box, score, car, angle_model_config):
        super.__init__(class_id, box, score)
        self.horizontal_features = None
        self.vertical_features = None
        self.car_box = car
        self.angle_model_config = angle_model_config

    def feature_directional_hist(self):
        pass

    def diff(self):
        pass
