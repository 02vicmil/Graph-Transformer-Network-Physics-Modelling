import pathlib

def get_directory_path(__file__in, up_directories=0):
    return str(
        pathlib.Path(__file__in)
        .parents[up_directories]
        .resolve()
    ).replace("\\", "/")

class NodeFeatureIndices:
    is_pump = 0
    is_tube = 1
    is_suction_cup = 2
    is_tube_connector = 3
    is_virtual_node = 4
    tube_length = 5
    time = 6
    pump_model1 = 7
    pump_model2 = 8
    pump_model3 = 9
    pump_model4 = 10
    pump_model5 = 11
    tube_diameter1 = 12
    tube_diameter2 = 13
    tube_diameter3 = 14
    tube_diameter4 = 15
    tube_diameter5 = 16
    suction_cup_model1 = 17
    suction_cup_model2 = 18
    suction_cup_model3 = 19
    suction_cup_model4 = 20
    suction_cup_model5 = 21
    suction_cup_material1 = 22
    suction_cup_material2 = 23
    counter = 24 # Just to keep track of length of the vector

    component_type_features = [is_pump, is_tube, is_suction_cup, is_tube_connector, is_virtual_node]
    pump_model_features = [pump_model1, pump_model2, pump_model3, pump_model4, pump_model5]
    tube_diameter_features = [tube_diameter1, tube_diameter2, tube_diameter3, tube_diameter4, tube_diameter5]
    suction_cup_model_features = [suction_cup_model1, suction_cup_model2, suction_cup_model3, suction_cup_model4, suction_cup_model5]
    suction_cup_material_features = [suction_cup_material1, suction_cup_material2]


def get_enabled_feature(features, available_indices):
    index = None
    index_count = 0
    for i in range(len(available_indices)):
        if features[available_indices[i]]:
            index = i
            index_count += 1

    if index_count > 1:
        index = None
    return index