import step1_generate_dataset
import step2_train_ml_model
from step2_train_ml_model import DataSampler, get_normalization, get_model_data, data_graph_pre_processing, get_directory_path
from util import NodeFeatureIndices, get_enabled_feature
import numpy as np
import torch
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import math

def undo_target_pre_processing(normalization, target: torch.Tensor, device):
    target_mean = torch.tensor(normalization["target_mean"], device=device)
    target_std = torch.tensor(normalization["target_std"], device=device)

    target = target * target_std
    target = target + target_mean
    target = torch.exp(target) - 1
    return target


def make_model_predictions(model_directory_path: str, samples_node_features, samples_edges, model, device):
    normalization = get_normalization(model_directory_path=model_directory_path, training_file_path=None)
    data_graph_pre_processing(samples_node_features, samples_edges, normalization)
    sample_count = len(samples_node_features)

    predictions = list()
    for sample in range(sample_count):
        node_features = samples_node_features[sample]
        edges = samples_edges[sample]
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0) # Make sure node features are passed in batch format

        prediction: torch.Tensor = model.forward(node_features, edges)
        prediction = prediction.squeeze(0)
        prediction = undo_target_pre_processing(normalization, prediction, device)
        predictions.append(prediction.tolist())

    return predictions


def compute_average_simulation_runtime(eval_file_path: str):
    with open(eval_file_path, "r") as f:
        runtimes = [
            json.loads(line)["computation_time"]
            for line in f
            if line.strip()
        ]

    print("average simulation runtime:", np.mean(runtimes))


def compute_average_model_runtime(model_directory_path: str, eval_file_path: str, device):
    best_checkpoint_path = model_directory_path + "/best_model.checkpoint"
    model, optimizer, criterion, interval_train_loss, interval_eval_loss, best_eval_loss = get_model_data(best_checkpoint_path, device)
    normalization = get_normalization(model_directory_path=model_directory_path, training_file_path=None)

    with open(eval_file_path, "r") as f:
        samples_node_features = [
            torch.tensor(json.loads(line)["node_features"], device=device)
            for line in f
            if line.strip()
        ]
    with open(eval_file_path, "r") as f:
        samples_edges = [
            torch.tensor(json.loads(line)["edges"], device=device)
            for line in f
            if line.strip()
        ]

    target_time = 0.1 * 10**(np.random.random() * math.log10(50))
    for sample in range(len(samples_node_features)):
        samples_node_features[sample][:, NodeFeatureIndices.time] = target_time
    data_graph_pre_processing(samples_node_features, samples_edges, normalization)

    runtimes = []
    for sample in range(len(samples_node_features)):
        node_features = samples_node_features[sample]
        edges = samples_edges[sample]
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0) # Make sure node features are passed in batch format

        start_time = time.time()
        prediction: torch.Tensor = model.forward(node_features, edges)
        runtime = time.time() - start_time
        runtimes.append(runtime)

    print("average model runtime:", np.mean(runtimes))


def compute_error_percentages(
    targets: List[float],
    predictions: List[float],
    min_target_value # Have targets have a minimum value to avoid undefined percentages
) -> List[float]:
    return [
        abs(target - prediction) / max(abs(target), min_target_value) * 100 
        for target, prediction in zip(targets, predictions)
    ]


def _save_or_show(
    path: Optional[str],
):
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_sorted_predictions(
    targets: List[float],
    predictions: List[float],
    label: str,
    path: str=None,
):
    """
    Sorts by target and plots target and prediction together.

    This makes systematic under/over-prediction very easy to see.
    """

    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    order = np.argsort(targets)

    plt.figure(figsize=(12, 6))

    plt.plot(
        targets[order],
        label="Target",
        linewidth=0,
        marker="o",
        markersize=3,
    )

    plt.plot(
        predictions[order],
        label="Prediction",
        linewidth=0,
        marker="o",
        markersize=3,
    )

    plt.xlabel("Samples sorted by target")
    plt.ylabel("Value")
    plt.title(
        f"Sorted Target vs Prediction — {label}"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    _save_or_show(path)


def plot_percent_error_distribution(
    lift_error_percentages,
    air_cons_error_percentages,
    path,
):
    """
    Shows the cumulative distribution of absolute relative error
    for both lift and air conditioners.

    For each threshold (5%, 10%, ..., 100%), the plot shows the
    percentage of cases whose absolute percentage error is <= that threshold.
    """

    lift_error_percentages = np.array(lift_error_percentages, dtype=float)
    air_cons_error_percentages = np.array(air_cons_error_percentages, dtype=float)

    # Absolute percentage error
    abs_lift_error_percentages = np.abs(lift_error_percentages)
    abs_air_cons_error_percentages = np.abs(air_cons_error_percentages)

    thresholds = np.arange(5, 101, 5)

    # Percentage of cases under each threshold
    lift_percentages = [
        np.mean(abs_lift_error_percentages <= threshold) * 100
        for threshold in thresholds
    ]

    air_cons_percentages = [
        np.mean(abs_air_cons_error_percentages <= threshold) * 100
        for threshold in thresholds
    ]

    for threshold, lift_pct, air_pct in zip(
        thresholds, lift_percentages, air_cons_percentages
    ):
        print(
            f"Threshold <= {threshold}%: "
            f"Lift = {lift_pct:.2f}%, "
            f"Air consumption = {air_pct:.2f}%"
        )

    plt.figure(figsize=(9, 6))

    plt.plot(
        thresholds,
        lift_percentages,
        marker="o",
        linewidth=2,
        label="Lift",
    )

    plt.plot(
        thresholds,
        air_cons_percentages,
        marker="o",
        linewidth=2,
        label="Air Cons",
    )

    plt.xlabel("Absolute Error Threshold (%)")
    plt.ylabel("Cases Within Threshold (%)")
    plt.title(f"Cumulative Percent Error")

    plt.xticks(thresholds)
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim(0, 100)

    plt.grid(True, alpha=0.3)
    plt.legend()

    _save_or_show(path)


def compute_evaluation_metrics_for_best_model(dataset_file_path: str, model_directory_path: str, image_directory_path: str, sample_count: int, device):
    eval_sampler = DataSampler(dataset_file_path)
    best_checkpoint_path = model_directory_path + "/best_model.checkpoint"
    model, optimizer, criterion, interval_train_loss, interval_eval_loss, best_eval_loss = get_model_data(best_checkpoint_path, device)
    np.random.seed(42)
    samples_node_features, samples_edges, samples_targets = eval_sampler.sample_data(count=sample_count, device=device)
    model_predictions = make_model_predictions(model_directory_path, samples_node_features, samples_edges, model, device)
    target_lift: List[float] = [min(max(1, target[0].tolist()), 500) for target in samples_targets]
    target_air_cons: List[float] = [max(0.01, target[1].tolist()) for target in samples_targets]
    predicted_lift: List[float] = [prediction[0] for prediction in model_predictions]
    predicted_air_cons: List[float] = [prediction[1] for prediction in model_predictions]
    lift_error_percentages = compute_error_percentages(target_lift, predicted_lift, 1)
    air_cons_error_percentages = compute_error_percentages(target_air_cons, predicted_air_cons, 0.01)

    plot_sorted_predictions(target_lift, predicted_lift, "Lift", image_directory_path + "/sorted_predictions_lift")
    plot_sorted_predictions(target_air_cons, predicted_air_cons, "Air Consumption", image_directory_path + "/sorted_predictions_air_cons")
    plot_percent_error_distribution(lift_error_percentages, air_cons_error_percentages, image_directory_path + "/cumulative_percent_error")


import torch
def model_estimate_graph(nodes: torch.Tensor, edges: torch.Tensor, model_directory_path, model, device):
    model_time_data = [i * 0.01 for i in range(10, 500)]
    # (model_time_data)
    normalization = get_normalization(model_directory_path=model_directory_path, training_file_path=None)

    samples_node_features = list()
    samples_edges = list()
    for i in range(len(model_time_data)):
        node_features_copy = nodes.clone()
        edges_copy = edges.clone()
        node_features_copy[:, NodeFeatureIndices.time] = model_time_data[i]
        samples_node_features.append(node_features_copy)
        samples_edges.append(edges_copy)

    predictions = make_model_predictions(model_directory_path=model_directory_path, samples_node_features=samples_node_features, samples_edges=samples_edges, model=model, device=device)
    
    predicted_lift = [prediction[0] for prediction in predictions]
    predicted_energy_cons = [prediction[1] for prediction in predictions]

    return model_time_data, predicted_lift, predicted_energy_cons


import dataset_plot
def evaluate_best_model_on_graph(graph_index: int, model_directory_path: str, image_directory_path: str, device):
    line = dataset_plot.read_line_in_file(
        step1_generate_dataset.eval_file_path,
        graph_index,
    )
    datapoint = json.loads(line)

    print(datapoint["seed"])
    nodes = datapoint["node_features"]
    edges = datapoint["edges"]

    labels = ["", "", "", ""]

    node_label = [
        labels[
            get_enabled_feature(
                node,
                NodeFeatureIndices.component_type_features,
            )
        ]
        for node in nodes
    ]

    named_edges = [
        (
            f"{a}{node_label[a]}",
            f"{b}{node_label[b]}",
        )
        for a, b in edges
    ]

    positions = dataset_plot.tree_positions(
        named_edges,
        root=f"0{node_label[0]}",
        x_spacing=2.0,   # Increase this for more horizontal space
        y_spacing=1.5,
    )

    information = dataset_plot.create_node_information(
        nodes,
        node_label,
    )

    best_checkpoint_path = model_directory_path + "/best_model.checkpoint"
    model, optimizer, criterion, interval_train_loss, interval_eval_loss, best_eval_loss = get_model_data(best_checkpoint_path, device)

    model_time_data, predicted_lift, predicted_energy_cons = model_estimate_graph(torch.tensor(nodes, device=device), torch.tensor(edges, device=device), model_directory_path, model, device)
    idx = next((i for i, x in enumerate(datapoint["time_data"]) if x > 0.1), None)
    dataset_plot.plot_network_and_data(
        positions,
        named_edges,
        information,
        [("Target", datapoint["time_data"][idx:], datapoint["lift_data"][idx:]), ("Prediction", model_time_data, predicted_lift)],
        [("Target", datapoint["time_data"][idx:], datapoint["cumulative_air_cons"][idx:]), ("Prediction", model_time_data, predicted_energy_cons)],
        save_path=image_directory_path + f"/system_plot_eval_graph{graph_index}",
        graph_header=f"Graph {graph_index} (eval dataset)"
    )


def run_step3_evaluate_model():
    print("Generate system examples with model evaluations")
    np.random.seed(42)
    eval_sample_count = step1_generate_dataset.get_data_sample_count(step1_generate_dataset.eval_file_path)
    graph_indices = [np.random.randint(0, eval_sample_count) for i in range(1)]
    for graph_index in graph_indices:
        evaluate_best_model_on_graph(graph_index, step2_train_ml_model.model_directory_path, image_directory_path=step2_train_ml_model.image_directory_path, device=step2_train_ml_model.device)

    print("Evaluate model performance")
    compute_evaluation_metrics_for_best_model(
        dataset_file_path=step1_generate_dataset.test_file_path,
        model_directory_path=step2_train_ml_model.model_directory_path, 
        image_directory_path=step2_train_ml_model.image_directory_path, 
        sample_count=100_000, 
        device=step2_train_ml_model.device,
    )

    print("Compute runtime")
    compute_average_simulation_runtime(
        eval_file_path=step1_generate_dataset.eval_file_path
    )

    compute_average_model_runtime(
        model_directory_path=step2_train_ml_model.model_directory_path, 
        eval_file_path=step1_generate_dataset.eval_file_path,
        device=step2_train_ml_model.device
    )

    compute_average_model_runtime(
        model_directory_path=step2_train_ml_model.model_directory_path, 
        eval_file_path=step1_generate_dataset.eval_file_path,
        device=torch.device("cpu")
    )

if __name__ == "__main__":
    run_step3_evaluate_model()