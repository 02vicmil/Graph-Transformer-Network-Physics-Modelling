import json

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

import step1_generate_dataset
import step2_train_ml_model
from util import NodeFeatureIndices, get_enabled_feature
from step2_train_ml_model import get_normalization, data_graph_pre_processing, get_model_data, get_directory_path
from step3_evaluate_model import make_model_predictions
import textwrap
import numpy as np


def tree_positions(edges, root=None, x_spacing=2.0, y_spacing=1.5):
    """Position tree nodes by depth, with configurable spacing."""
    graph = nx.Graph()
    graph.add_edges_from(edges)

    if root is None:
        root = next(iter(graph.nodes))

    depths = nx.single_source_shortest_path_length(graph, root)
    levels = {}

    for node, depth in depths.items():
        levels.setdefault(depth, []).append(node)

    positions = {}

    for depth, nodes in levels.items():
        offset = (len(nodes) - 1) * x_spacing / 2

        for i, node in enumerate(nodes):
            positions[node] = (
                i * x_spacing - offset,
                -depth * y_spacing,
            )

    return positions


def draw_graph_panel(ax, positions, connections, information, graph_header = "Graph"):
    """Draw graph nodes with text that is guaranteed to stay inside boxes."""

    graph = nx.Graph()
    graph.add_edges_from(connections)

    nx.draw_networkx_edges(
        graph,
        pos=positions,
        ax=ax,
        edge_color="black",
        width=1.5,
    )

    # Maximum width of a node box in graph coordinates.
    MAX_WIDTH = 3.0

    # Minimum width.
    MIN_WIDTH = 1.2

    # Text settings.
    fontsize = 8
    fontweight = "bold"

    # First draw the text temporarily so Matplotlib can measure it.
    fig = ax.figure

    renderer = fig.canvas.get_renderer()

    node_data = {}

    for node, (x, y) in positions.items():

        raw_lines = [str(node)] + [
            str(line)
            for line in information.get(node, [])
            if line
        ]

        # Start with a reasonable wrapping width.
        chars_per_line = 30

        wrapped_lines = []

        for line in raw_lines:
            wrapped_lines.extend(
                textwrap.wrap(
                    line,
                    width=chars_per_line,
                    break_long_words=True,
                    break_on_hyphens=False,
                ) or [""]
            )

        text_string = "\n".join(wrapped_lines)

        # Temporary invisible text.
        temp = ax.text(
            0,
            0,
            text_string,
            fontsize=fontsize,
            fontweight=fontweight,
            ha="center",
            va="center",
        )

        bbox = temp.get_window_extent(renderer=renderer)

        temp.remove()

        # Convert pixel dimensions to graph coordinates.
        inv = ax.transData.inverted()

        p0 = inv.transform((0, 0))
        p1 = inv.transform((bbox.width, bbox.height))

        text_width = p1[0] - p0[0]
        text_height = p1[1] - p0[1]

        # Add padding.
        width = text_width + 0.25
        height = text_height + 0.20

        width = max(width, MIN_WIDTH)
        width = min(width, MAX_WIDTH)

        node_data[node] = {
            "lines": wrapped_lines,
            "width": width,
            "height": height,
        }

    # Draw nodes.
    for node, (x, y) in positions.items():

        data = node_data[node]

        width = data["width"]
        height = data["height"]

        ax.add_patch(
            FancyBboxPatch(
                (
                    x - width / 2,
                    y - height / 2,
                ),
                width,
                height,
                boxstyle="round,pad=0.05",
                facecolor="lightblue",
                edgecolor="black",
                linewidth=1.5,
                zorder=2,
            )
        )

        ax.text(
            x,
            y,
            "\n".join(data["lines"]),
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            zorder=3,
        )

    # Plot limits.
    if positions:
        xs, ys = zip(*positions.values())

        max_width = max(
            data["width"]
            for data in node_data.values()
        )

        max_height = max(
            data["height"]
            for data in node_data.values()
        )

        ax.set_xlim(
            min(xs) - max_width / 2 - 0.5,
            max(xs) + max_width / 2 + 0.5,
        )

        ax.set_ylim(
            min(ys) - max_height / 2 - 0.5,
            max(ys) + max_height / 2 + 0.5,
        )

    ax.set_title(
        graph_header,
        fontsize=14,
        fontweight="bold",
    )

    ax.set_aspect("equal")
    ax.axis("off")


def draw_scatter_panel(ax, values, title, x_label, y_label):
    # Single dataset: (x, y)
    if len(values) == 2 and not isinstance(values[0], tuple):
        values = [("Data", values[0], values[1])]

    for label, x, y in values:
        ax.scatter(x, y, label=label)

    ax.set(
        title=title,
        xlabel=x_label,
        ylabel=y_label,
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

from pathlib import Path

def plot_network_and_data(
    node_positions,
    node_connections,
    node_information,
    plot1_values,
    plot2_values,
    save_path=get_directory_path(__file__) + "/network_and_data.png",
    graph_header = "Graph"
):
    fig, (ax_graph, ax_lift, ax_air) = plt.subplots(
        1,
        3,
        figsize=(22, 6),
        gridspec_kw={"width_ratios": [1.5, 1, 1]},
    )

    draw_graph_panel(
        ax_graph,
        node_positions,
        node_connections,
        node_information,
        graph_header
    )

    draw_scatter_panel(
        ax_lift,
        plot1_values,
        "Lift",
        "Time [Seconds]",
        "Lift [Newton]",
    )

    draw_scatter_panel(
        ax_air,
        plot2_values,
        "Cumulative Air Consumption",
        "Time [Seconds]",
        "Air Consumption [Normal Liter]",
    )

    plt.tight_layout()

    # Save the complete figure 
    save_path = Path(save_path) 
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def read_line_in_file(file_path, line_nr):
    with open(file_path) as file:
        for number, line in enumerate(file, start=1):
            if number == line_nr:
                return line


def create_node_information(node_features, node_label):
    information = {}

    for i, node in enumerate(node_features):
        name = f"{i}{node_label[i]}"

        if node[NodeFeatureIndices.is_tube_connector]:
            information[name] = ["Connector"]

        elif node[NodeFeatureIndices.is_tube]:
            diameter = get_enabled_feature(
                node,
                NodeFeatureIndices.tube_diameter_features,
            )
            length = node[NodeFeatureIndices.tube_length]

            information[name] = [
                "Tube",
                f"Diameter: {[2, 4, 6, 9, 12][diameter]}mm",
                f"Length: {length:.2f}m",
            ]

        elif node[NodeFeatureIndices.is_pump]:
            model = get_enabled_feature(
                node,
                NodeFeatureIndices.pump_model_features,
            )

            information[name] = [
                "Pump",
                f"Model: {["AG", "BV", "AB", "BX", "BY"][model]}",
            ]

        elif node[NodeFeatureIndices.is_suction_cup]:
            model = get_enabled_feature(
                node,
                NodeFeatureIndices.suction_cup_model_features,
            )
            material = get_enabled_feature(
                node,
                NodeFeatureIndices.suction_cup_material_features,
            )

            information[name] = [
                "Suction Cup",
                f"Model: {["ABG", "AAZ", "ABA", "AFE", "AHB"][model]}",
                f"Material: {["METAL", "CARDBOARD"][material]}",
            ]

        else:
            information[name] = ["Unknown"]

    return information


def plot_datapoint(datapoint, save_path):
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

    positions = tree_positions(
        named_edges,
        root=f"0{node_label[0]}",
        x_spacing=2.0,   # Increase this for more horizontal space
        y_spacing=1.5,
    )

    information = create_node_information(
        nodes,
        node_label,
    )

    plot_network_and_data(
        positions,
        named_edges,
        information,
        (datapoint["time_data"], datapoint["lift_data"]),
        (datapoint["time_data"], datapoint["cumulative_air_cons"]),
        save_path=save_path
    )


if __name__ == "__main__":
    sample_count = step1_generate_dataset.get_data_sample_count(
        step1_generate_dataset.eval_file_path
    )

    for i in range(1):
        np.random.seed(42)
        sample_index = np.random.randint(1, sample_count)

        line = read_line_in_file(
            step1_generate_dataset.eval_file_path,
            sample_index,
        )

        plot_datapoint(json.loads(line), step2_train_ml_model.image_directory_path + f"/system_plot_eval_graph{sample_index}")