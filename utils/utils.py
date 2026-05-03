import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def compare_model_results(answer_maps: list[list[int]],
                          model_names: list[str],
                          save_path: str ="comparison_result.png") -> None:
    """
    Visualize the differences in the answers provided by multiple models.

    :param answer_maps: A list containing multiple 0/1 arrays (e.g., [arr1, arr2, ...]).
    :param model_names: A list of model names (e.g., ['Model_A', 'Model_B', ...]).
    :param save_path: The path to save the generated image.
    """
    num_models = len(answer_maps)
    num_questions = len(answer_maps[0])

    data_matrix = np.array(answer_maps)

    fig = plt.figure(figsize=(18, 6 + num_models))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5])

    ax0 = fig.add_subplot(gs[0])

    cmap_res = sns.color_palette(["#e74c3c", "#2ecc71"])
    sns.heatmap(data_matrix, cmap=cmap_res, cbar=False, ax=ax0,
                yticklabels=model_names, xticklabels=False)
    ax0.set_title('Item-level Comparison (Green=Correct, Red=Incorrect)', fontsize=14)
    ax0.set_xlabel(f'Questions (1 to {num_questions})')

    agreement_matrix = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(num_models):
            agreement_matrix[i, j] = np.mean(answer_maps[i] == answer_maps[j])

    ax1 = fig.add_subplot(gs[1])
    sns.heatmap(agreement_matrix, annot=True, fmt=".1%", cmap="Blues", ax=ax1,
                xticklabels=model_names, yticklabels=model_names)
    ax1.set_title('Pairwise Agreement Rate (Do they answer the same?)', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data.append(json.loads(line))

    return data