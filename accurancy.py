import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Generate 100 review data
categories = ["Delivery", "Quality", "Price", "Packaging", "Service"]
true_data = []
for _ in range(100):
    true_data.append({
        "category": random.choice(categories),
        "sentiment": random.choice(["positive", "neutral", "negative"])
    })

# Step 2: Generate predictions for GPT-2 and GPT-4o

def generate_predictions(true_labels, accuracy):
    predictions = []
    for label in true_labels:
        if random.random() < accuracy:  # Correct prediction
            predictions.append(label)  # Append the correct label
        else:  # Random incorrect prediction
            predictions.append(random.choice(["positive", "neutral", "negative"]))
    return predictions

# Simulate GPT-2 predictions with extremely low accuracy
category_accuracies_gpt2 = {
    "Delivery": 0.05,
    "Quality": 0.10,
    "Price": 0.15,
    "Packaging": 0.05,
    "Service": 0.05
}
predictions_gpt2 = {}
for category in categories:
    true_labels = [entry["sentiment"] for entry in true_data if entry["category"] == category]
    predictions_gpt2[category] = generate_predictions(true_labels, category_accuracies_gpt2[category])

# Simulate GPT-4o predictions with high accuracy
category_accuracies_gpt4o = {
    "Delivery": 0.98,
    "Quality": 0.99,
    "Price": 0.97,
    "Packaging": 0.98,
    "Service": 0.95
}
predictions_gpt4o = {}
for category in categories:
    true_labels = [entry["sentiment"] for entry in true_data if entry["category"] == category]
    predictions_gpt4o[category] = generate_predictions(true_labels, category_accuracies_gpt4o[category])

# Step 3: Calculate performance metrics
metrics_gpt2 = {}
metrics_gpt4o = {}

for category in categories:
    true_labels = [entry["sentiment"] for entry in true_data if entry["category"] == category]
    pred_labels_gpt2 = predictions_gpt2[category]
    pred_labels_gpt4o = predictions_gpt4o[category]

    metrics_gpt2[category] = {
        "Accuracy": accuracy_score(true_labels, pred_labels_gpt2),
        "Precision": precision_score(true_labels, pred_labels_gpt2, average="weighted", zero_division=0),
        "Recall": recall_score(true_labels, pred_labels_gpt2, average="weighted", zero_division=0),
        "F1": f1_score(true_labels, pred_labels_gpt2, average="weighted", zero_division=0)
    }

    metrics_gpt4o[category] = {
        "Accuracy": accuracy_score(true_labels, pred_labels_gpt4o),
        "Precision": precision_score(true_labels, pred_labels_gpt4o, average="weighted", zero_division=0),
        "Recall": recall_score(true_labels, pred_labels_gpt4o, average="weighted", zero_division=0),
        "F1": f1_score(true_labels, pred_labels_gpt4o, average="weighted", zero_division=0)
    }

# Extract accuracy values for visualization
accuracy_gpt2 = [metrics_gpt2[category]["Accuracy"] * 100 for category in categories]
accuracy_gpt4o = [metrics_gpt4o[category]["Accuracy"] * 100 for category in categories]

# Step 4: Visualization
x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, accuracy_gpt2, width, label='GPT-2', color='orange')
rects2 = ax.bar(x + width / 2, accuracy_gpt4o, width, label='GPT-4o', color='green')

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Categories')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Performance Comparison: GPT-2 vs GPT-4o')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Add accuracy values on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, 110)  # Set y-axis limit to 110% for better visualization
plt.tight_layout()
plt.show()

# Print detailed metrics
print("GPT-2 Performance Metrics by Category:")
for category, metrics in metrics_gpt2.items():
    print(f"{category}: {metrics}")

print("\nGPT-4o Performance Metrics by Category:")
for category, metrics in metrics_gpt4o.items():
    print(f"{category}: {metrics}")
