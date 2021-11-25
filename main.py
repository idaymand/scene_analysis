from ai_scan_metadata import AIScanMetadata
from json_configuration import Configurator


def get_best_scene(input_file_name, path_to_image, path_to_csv):
    configurator = Configurator('/Users/innadaymand/PycharmProjects/service-ravin-monorepo/models/proxy/proxy.json')
    scan = AIScanMetadata(configurator)
    scan.process_parts_from_file(input_file_name, path_to_image, path_to_csv)


def get_best_scene(input_file_name):
    configurator = Configurator('/Users/innadaymand/PycharmProjects/service-ravin-monorepo/models/proxy/proxy.json')
    scan = AIScanMetadata(configurator)
    scan.process_parts_from_view_metadata(input_file_name)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get_best_scene(r'65_27.27.jpg.csv', r'/Users/innadaymand'
    #                                     r'/PycharmProjects/service-ravin'
    #                                     r'-monorepo/apps/algo/src/assets'
    #                                     r'/algo-test/tfs-488', r'/Users/innadaymand/PycharmProjects/service-ravin'
    #                                                            r'-monorepo/apps/algo/src'
    #                                                            r'/assets/algo-test/tfs-488/csv')

    get_best_scene(r'/Users/innadaymand/PycharmProjects/sceneAnalysis/input_csv/0.pngvia_region_data.csv')