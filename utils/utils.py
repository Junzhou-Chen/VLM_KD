import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def compare_model_results(answer_maps: list[list[int]],
                          model_names: list[str],
                          save_path: str = "./comparison_result.png") -> None:
    """
    Compare the prediction results (correct/incorrect) of two models
    and plot an agreement/confusion matrix, saving it as an image.
    """
    if len(answer_maps) < 2 or len(model_names) < 2:
        raise ValueError("At least two models' data are required for comparison.")

    # Extract data for the first two models and convert to numpy arrays
    model_a_results = np.array(answer_maps[0])
    model_b_results = np.array(answer_maps[1])

    name_a = model_names[0]
    name_b = model_names[1]

    a_1_b_1 = np.sum((model_a_results == 1) & (model_b_results == 1))
    a_1_b_0 = np.sum((model_a_results == 1) & (model_b_results == 0))
    a_0_b_1 = np.sum((model_a_results == 0) & (model_b_results == 1))
    a_0_b_0 = np.sum((model_a_results == 0) & (model_b_results == 0))

    matrix = np.array([
        [a_1_b_1, a_1_b_0],
        [a_0_b_1, a_0_b_0]
    ])


    plt.figure(figsize=(15, 4))
    xticklabels = [f'{name_b} Correct', f'{name_b} Incorrect']
    yticklabels = [f'{name_a} Correct', f'{name_a} Incorrect']
    ax = sns.heatmap(matrix,
                     annot=True,
                     fmt='d',
                     cmap='Blues',
                     xticklabels=xticklabels,
                     yticklabels=yticklabels)
    plt.title('Agreement / Confusion Matrix (0 and 1)', fontsize=12)
    plt.yticks(rotation=90, va='center')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data.append(json.loads(line))

    return data