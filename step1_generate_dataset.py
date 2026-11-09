import numpy as np
import os
import json
from pathlib import Path

from util import NodeFeatureIndices, get_directory_path
if os.path.exists(get_directory_path(__file__) + "/ovm_pro.py"):
    from ovm_pro import generate_simulation_datapoint

def sample_tube_connector_features():
    node_features = [0] * NodeFeatureIndices.counter
    node_features[NodeFeatureIndices.is_tube_connector] = 1
    return node_features

def sample_suction_cup_features():
    node_features = [0] * NodeFeatureIndices.counter
    node_features[NodeFeatureIndices.is_suction_cup] = 1
    node_features[NodeFeatureIndices.suction_cup_model_features[np.random.randint(low=0, high=len(NodeFeatureIndices.suction_cup_model_features))]] = 1
    node_features[NodeFeatureIndices.suction_cup_material_features[np.random.randint(low=0, high=len(NodeFeatureIndices.suction_cup_material_features))]] = 1
    return node_features

def sample_pump_features():
    node_features = [0] * NodeFeatureIndices.counter
    node_features[NodeFeatureIndices.is_pump] = 1
    node_features[NodeFeatureIndices.pump_model_features[np.random.randint(low=0, high=len(NodeFeatureIndices.pump_model_features))]] = 1
    return node_features

def sample_tube_features():
    node_features = [0] * NodeFeatureIndices.counter
    node_features[NodeFeatureIndices.is_tube] = 1
    node_features[NodeFeatureIndices.tube_diameter_features[np.random.randint(low=0, high=len(NodeFeatureIndices.tube_diameter_features))]] = 1
    node_features[NodeFeatureIndices.tube_length] = 5 * (10 ** (np.random.rand() * 2 - 2))
    return node_features

# Generate a random vacuum system, based on random vacuum system generation
def sample_random_vacuum_system(seed):
    np.random.seed(seed=seed)

    # Generate the number of components
    tube_connector_count = np.random.randint(low=1, high=3+1)
    suction_cup_count = np.random.randint(low=1, high=3+1)
    pump_count = np.random.randint(low=1, high=2+1)

    # Assign ids to the components
    component_id_counter = 0
    tube_connector_ids = []
    for _ in range(tube_connector_count):
        tube_connector_ids.append(component_id_counter)
        component_id_counter += 1

    suction_cup_ids = []
    for _ in range(suction_cup_count):
        suction_cup_ids.append(component_id_counter)
        component_id_counter += 1

    pump_ids = []
    for _ in range(pump_count):
        pump_ids.append(component_id_counter)
        component_id_counter += 1


    # Connect all the tube connectors with tubes
    tube_ids = list()
    edges = []
    for i in range(1, tube_connector_count):
        # Create a tube
        tube_id = component_id_counter
        tube_ids.append(tube_id)
        component_id_counter += 1

        # Connect tube connectors
        other_tube_connector_index = np.random.randint(low=0, high=i)
        edges.append([tube_id, tube_connector_ids[i]])
        edges.append([tube_id, tube_connector_ids[other_tube_connector_index]])

    # Connect all the suction cups with tubes
    for i in range(0, suction_cup_count):
        # Create a tube
        tube_id = component_id_counter
        tube_ids.append(tube_id)
        component_id_counter += 1

        # Connect suction cup
        tube_connector_index = np.random.randint(low=0, high=tube_connector_count)
        edges.append([tube_id, tube_connector_ids[tube_connector_index]])
        edges.append([tube_id, suction_cup_ids[i]])

    # Connect all the pumps with tubes
    for i in range(0, pump_count):
        # Create a tube
        tube_id = component_id_counter
        tube_ids.append(tube_id)
        component_id_counter += 1

        # Connect pump
        tube_connector_index = np.random.randint(low=0, high=tube_connector_count)
        edges.append([tube_id, tube_connector_ids[tube_connector_index]])
        edges.append([tube_id, pump_ids[i]])
    

    # Assign node features to all the components
    node_features = [[]] * component_id_counter
    for id in tube_connector_ids:
        node_features[id] = sample_tube_connector_features()

    for id in suction_cup_ids:
        node_features[id] = sample_suction_cup_features()

    for id in pump_ids:
        node_features[id] = sample_pump_features()

    for id in tube_ids:
        node_features[id] = sample_tube_features()

    return node_features, edges


def get_data_sample_count(file_path):
    if not os.path.exists(file_path):
        return 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
        return line_count
    

def generate_datapoints(file_path, target_count, seed):
    sample_count = get_data_sample_count(file_path=file_path)

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "a+", encoding="utf-8") as f:
        while(sample_count < target_count):
            system_seed = sample_count + seed + 1
            all_node_features, edges = sample_random_vacuum_system(seed=system_seed)
            simulation_datapoint = generate_simulation_datapoint(all_node_features, edges)
            simulation_datapoint["seed"] = system_seed
            f.write(json.dumps(simulation_datapoint) + "\n")
            sample_count += 1
            print(sample_count , "/", target_count, file_path)


total_simulation_count = 100_000
train_split = 0.8
eval_split = 0.1
test_split = 0.1

training_file_path = get_directory_path(__file__) + "/data/training_simulations.jsonl"
eval_file_path = get_directory_path(__file__) + "/data/eval_simulations.jsonl"
test_file_path = get_directory_path(__file__) + "/data/test_simulations.jsonl"


def run_step1_generate_dataset():
    generate_datapoints(file_path=training_file_path, target_count=total_simulation_count * train_split, seed=10_000_000)
    generate_datapoints(file_path=test_file_path, target_count=total_simulation_count * test_split, seed=20_000_000)
    generate_datapoints(file_path=eval_file_path, target_count=total_simulation_count * eval_split, seed=30_000_000)

if __name__ == "__main__":
    run_step1_generate_dataset()


    