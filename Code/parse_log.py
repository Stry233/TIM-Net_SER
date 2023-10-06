# Required Libraries
import re
import matplotlib.pyplot as plt

dataset = "RAVDE"
mode = 'bl' # 'bl'

# Read the file
with open(f"log/{dataset}_{mode}.txt", "r") as file:
    lines = file.readlines()

# Initialize lists to hold the data for each fold
folds_val_loss = []
folds_val_accuracy = []

# Temporary lists for current fold data
current_fold_val_loss = []
current_fold_val_accuracy = []

# Regular expression to match the relevant data
pattern = r"val_loss: (\d+\.\d+) - val_accuracy: (\d+\.\d+)"

for line in lines:
    # Check if it's the start of a new fold
    if "Temporal create succes!" in line:
        # If we already have data for the previous fold, append it to the main list
        if current_fold_val_loss and current_fold_val_accuracy:
            folds_val_loss.append(current_fold_val_loss)
            folds_val_accuracy.append(current_fold_val_accuracy)
            current_fold_val_loss = []
            current_fold_val_accuracy = []
    # Extract val_loss and val_accuracy using regex
    match = re.search(pattern, line)
    if match:
        current_fold_val_loss.append(float(match.group(1)))
        current_fold_val_accuracy.append(float(match.group(2)))

# Append data for the last fold
if current_fold_val_loss and current_fold_val_accuracy:
    folds_val_loss.append(current_fold_val_loss)
    folds_val_accuracy.append(current_fold_val_accuracy)


# Visualization

# Determine the best scores (minimum for loss and maximum for accuracy) for each fold
best_val_losses = [min(val_loss) for val_loss in folds_val_loss]
best_val_accuracies = [max(val_accuracy) for val_accuracy in folds_val_accuracy]

# Plotting the validation loss and validation accuracy for each fold with best scores
plt.figure(figsize=(15, 10))

# Plot Validation Loss
plt.subplot(2, 1, 1)
for i, val_loss in enumerate(folds_val_loss):
    plt.plot(val_loss, label=f"Fold {i+1} (Best Loss: {best_val_losses[i]:.4f})")
plt.title(f'Validation Loss for Each Fold - {dataset} & {mode}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Validation Accuracy
plt.subplot(2, 1, 2)
for i, val_accuracy in enumerate(folds_val_accuracy):
    plt.plot(val_accuracy, label=f"Fold {i+1} (Best Acc: {best_val_accuracies[i]*100:.2f}%)")
plt.title(f'Validation Accuracy for Each Fold - {dataset} & {mode}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
