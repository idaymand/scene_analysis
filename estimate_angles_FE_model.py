import tensorflow as tf

import itertools

import numpy as np
from algo_lib import rect
from keras import layers, Input, Model
from sklearn.preprocessing import normalize

from metadataHandler import View, car

NUM_VALIDS = 3
NUM_FEATURES = 7
NUM_FEATURES_WIDTH = 10
NUM_FEATURES_HEIGHT = 30
NUM_FEATURES_HEIGHTD = NUM_FEATURES_HEIGHT - 1
NUM_FEATURES_INIT = NUM_FEATURES * (NUM_FEATURES_HEIGHTD + NUM_FEATURES_WIDTH)
NUM_FEATURES_ALL = NUM_FEATURES * (NUM_FEATURES_HEIGHTD + NUM_FEATURES_WIDTH)
NUM_ANGLES_Z = 6
NUM_ANGLES_Y = 4
NUM_ANGLES_X = 36
NUM_SHIFTS_Z = 7
NUM_SHIFTS_Y = 7
NUM_SHIFTS_X = 13
NUM_MODELS = 19


def norm1d(x):
    return x / np.linalg.norm(x)


def vertical_1D_features_vec(data: dict, margin: float = 0.05, grid_size: int = 10, grid_size_y: int = 10) -> np.array:
    grid_size_y = 30
    grid_area = grid_size * grid_size_y

    playing_features = dict(zip(car.detection_parts_list, range(7)))
    car_rect = rect.Rectangle(data['car'][0]['rect'])
    feature_vec = np.zeros((len(car.detection_parts_list) * grid_area, 1))
    _vertical_1D_features_vec = np.zeros((len(playing_features) * (grid_size_y - 1), 1))
    # self._vertical_1D_features_vec = np.zeros((len(playing_features) * grid_area, 1))
    _vertical_1D_features_vec_conf = np.zeros((len(playing_features), 1))

    # steps_y = car_rect.up_extend(margin)+car_rect.down_extend(margin)
    #         - np.geomspace(start=car_rect.down_extend(margin), stop=car_rect.up_extend(margin), num=grid_size_y + 1)
    steps_y = np.linspace(start=car_rect.up_extend(margin), stop=car_rect.down_extend(margin), num=grid_size_y + 1)
    steps_x = np.linspace(start=car_rect.left_extend(margin), stop=car_rect.right_extend(margin), num=grid_size + 1)
    cat_rects: Dict[Tuple[int, int], rect.NumVec] = {}
    for i, j in itertools.product(range(grid_size_y), range(grid_size)):
        cat_rects[(i, j)] = [steps_y[i], steps_x[j], steps_y[i + 1], steps_x[j + 1]]

    for fType in playing_features:  # self.data:
        i = playing_features[fType]
        i1 = i * grid_area
        i2 = i * (grid_size_y - 1)
        for element in data.get(fType, []):
            r = element['rect']
            if element.get('Score', '') == '':
                element['Score'] = '1'
            _vertical_1D_features_vec_conf[i, 0] += float(element['Score']) / len(data[fType])
            for j, p in cat_rects:
                feature_vec[i1 + j * grid_size + p, 0] += rect.rectIntersect(cat_rects[(j, p)], r)[1]
        _vertical_1D_features_vec[i2:i2 + grid_size_y - 1, 0] = np.diff(
            feature_vec[i1:i1 + grid_area, 0].reshape((grid_size_y, grid_size)).sum(axis=1), axis=0)
        # self._vertical_1D_features_vec[i1:i1 + grid_area, 0] = feature_vec[i1:i1 + grid_area, 0]
        # #.reshape((10, 10)).sum(axis=1)
        # if feature_vec[i1:i1 + 100, 0].reshape((10, 10)).sum() == 0:
        #     # for cases where part is outside of the car (TODO: consider raising a flag in this case somehow)
        #     self._vertical_1D_features_vec_conf[i, 0] = 0

    # self._vertical_1D_features_vec = np.diff(self._vertical_1D_features_vec, axis=0)
    _vertical_1D_features_vec = normalize(_vertical_1D_features_vec, axis=0, norm='l2')
    return _vertical_1D_features_vec


def horizontal_1D_features_vec(data: dict, margin: float = 0.05, grid_size: int = 10) -> np.array:
    playing_features = dict(zip(car.detection_parts_list, range(7)))
    car_rect = rect.Rectangle(data['car'][0]['rect'])

    feature_vec = np.zeros((len(car.detection_parts_list) * 100, 1))
    _horizontal_1D_features_vec = np.zeros((len(playing_features) * 10, 1))
    _horizontal_1D_features_vec_conf = np.zeros((len(playing_features), 1))

    steps_y = np.linspace(start=car_rect.up_extend(margin), stop=car_rect.down_extend(margin), num=grid_size + 1)
    steps_x = np.linspace(start=car_rect.left_extend(margin), stop=car_rect.right_extend(margin), num=grid_size + 1)
    cat_rects: Dict[Tuple[int, int], rect.NumVec] = {}
    for i, j in itertools.product(range(grid_size), range(grid_size)):
        cat_rects[(i, j)] = [steps_y[i], steps_x[j], steps_y[i + 1], steps_x[j + 1]]

    for fType in playing_features:  # self.data:
        i = playing_features[fType]
        i1 = i * 100
        i2 = i * 10
        for element in data.get(fType, []):
            r = element['rect']
            if element.get('Score', '') == '':
                element['Score'] = '1'
            _horizontal_1D_features_vec_conf[i, 0] += float(element['Score']) / len(data[fType])
            for j, p in cat_rects:
                feature_vec[i1 + j * 10 + p, 0] += rect.rectIntersect(cat_rects[(j, p)], r)[1]
        _horizontal_1D_features_vec[i2:i2 + 10, 0] = feature_vec[i1:i1 + 100, 0].reshape((10, 10)).sum(axis=0)
        if feature_vec[i1:i1 + 100, 0].reshape((10, 10)).sum() == 0:
            # for cases where part is outside of the car (TODO: consider raising a flag in this case somehow)
            _horizontal_1D_features_vec_conf[i, 0] = 0

    _horizontal_1D_features_vec = normalize(_horizontal_1D_features_vec, axis=0, norm='l2')
    return _horizontal_1D_features_vec


def get_score_weights(data: dict):
    playing_features = dict(zip(car.detection_parts_list, range(7)))
    score_weights = np.zeros((len(playing_features), 1))
    for fType in playing_features:  # self.data:
        i = playing_features[fType]
        for element in data.get(fType, []):
            if element.get('Score', '') == '':
                element['Score'] = '1'
            score_weights[i, 0] += float(element['Score']) / len(data[fType])
    return score_weights


def interpret_angels(hv_normed):
    angs = [np.argmax(x) for x in hv_normed]
    yaw = angs[0] * 10
    pitch = angs[1] * 10
    roll = (angs[2] - 3) * 10
    elevation = 1 + angs[1] * .5
    return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 'elevation': elevation}


def angles_from_model(v: View):
    return angles_from_model_(v, model)


def angles_from_model_(v: View, angle_model):
    h1 = norm1d(horizontal_1D_features_vec(v.data).ravel())
    h2 = norm1d(vertical_1D_features_vec(v.data).ravel())
    f_hv = norm1d(np.hstack([h1, h2]))
    out_hv = angle_model.predict(f_hv.reshape((1, 273)))
    angle_dict = interpret_angels(out_hv)
    v.projection_params.yaw = angle_dict.get('yaw')
    v.projection_params.pitch = angle_dict.get('pitch')
    v.projection_params.roll = angle_dict.get('roll')
    return v, out_hv


# model1 = tf.keras.models.load_model(r'C:\Python\raven\event_parser\detectors\weights\7angles.h5')


inputs = Input(shape=(NUM_FEATURES_ALL,), name='box_features')
x1 = layers.Dense(3600, activation='relu', name='dense_1')(inputs)
outputs = (layers.Dense(NUM_MODELS, name='model')(x1),
           layers.Dense(NUM_ANGLES_X, name='yaw')(x1),
           layers.Dense(NUM_ANGLES_Y, name='pitch')(x1),
           layers.Dense(NUM_ANGLES_Z, name='roll')(x1),
           layers.Dense(NUM_SHIFTS_X, name='translation')(x1),
           layers.Dense(NUM_SHIFTS_Y, name='elevation')(x1),
           layers.Dense(NUM_SHIFTS_Z, name='distance')(x1))

model = Model(inputs=inputs, outputs=outputs)

model.load_weights(r'/Users/innadaymand/PycharmProjects/sceneAnalysis/model/7angles.h5')

# the boxes could be read directly into the v.data
