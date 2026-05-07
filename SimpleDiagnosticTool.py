import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import os

class SimpleDiagnosticTool:

    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.output_dir = "simple_diagnostic_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_training_history(self, history):

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Test Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Test Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '1_training_history.png'), dpi=150)
        plt.close()
        print(f"Saved: {self.output_dir}/1_training_history.png")

        # Check for overfitting
        if len(history['test_loss']) > 10:
            if history['test_loss'][-1] > history['test_loss'][-5] and history['train_loss'][-1] < \
                    history['train_loss'][-5]:
                print("Warning: Possible overfitting detected (test loss increasing)")

    def visualize_conv_filters(self):

        weights = self.model.conv1.weight.data.cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle('First Layer Convolutional Filters (8 filters)', fontsize=14)

        for i in range(8):
            ax = axes[i // 4, i % 4]
            kernel = weights[i, 0]

            vmin, vmax = kernel.min(), kernel.max()
            im = ax.imshow(kernel, cmap='viridis', vmin=vmin, vmax=vmax)

            ax.set_title(f'Filter {i + 1}')
            ax.axis('off')

        plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '2_conv_filters.png'), dpi=150)
        plt.close()
        print(f"Saved: {self.output_dir}/2_conv_filters.png")

        filter_mean = np.mean(np.abs(weights))
        filter_std = np.std(weights)
        print(f"Filter statistics: Mean(abs)={filter_mean:.4f}, Std={filter_std:.4f}")

        if filter_mean < 0.01:
            print("Warning: Filters may be too small (possible gradient vanishing)")
        if filter_mean > 1.0:
            print("Warning: Filters may be too large (possible gradient explosion)")

    def visualize_feature_maps(self, sample_data):

        device = next(self.model.parameters()).device
        self.model.eval()

        with torch.no_grad():
            _ = self.model(sample_data.to(device))

        activations = self.model.activations

        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        fig.suptitle('Feature Map Visualization Analysis', fontsize=14)

        input_img = sample_data[0, 0].cpu().numpy()
        axes[0, 0].imshow(input_img, cmap='gray')
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')

        if 'conv1' in activations:
            conv1_acts = activations['conv1'][0].cpu().numpy()
            for i in range(3):
                axes[0, i + 1].imshow(conv1_acts[i], cmap='hot')
                axes[0, i + 1].set_title(f'Layer 1 Channel {i + 1}')
                axes[0, i + 1].axis('off')

        if 'conv2' in activations:
            conv2_acts = activations['conv2'][0].cpu().numpy()
            for i in range(4):
                axes[1, i].imshow(conv2_acts[i], cmap='hot')
                axes[1, i].set_title(f'Layer 2 Channel {i + 1}')
                axes[1, i].axis('off')

        if 'conv3' in activations:
            conv3_acts = activations['conv3'][0].cpu().numpy()
            for i in range(4):
                axes[2, i].imshow(conv3_acts[i], cmap='hot')
                axes[2, i].set_title(f'Layer 3 Channel {i + 1}')
                axes[2, i].axis('off')

        axes[2, 3].axis('off')
        plt.text(0.5, 0.5, 'Feature Map Analysis:\n'
                           '- Layer 1: Edges/Textures\n'
                           '- Layer 2: Shape Parts\n'
                           '- Layer 3: High-level Features',
                 transform=axes[2, 3].transAxes,
                 ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '3_feature_maps.png'), dpi=150)
        plt.close()
        print(f"Saved: {self.output_dir}/3_feature_maps.png")

        if 'conv1' in activations and 'conv3' in activations:
            conv1_act_mean = np.mean(np.abs(conv1_acts))
            conv3_act_mean = np.mean(np.abs(conv3_acts))
            print(f"Activation statistics: Layer1={conv1_act_mean:.4f}, Layer3={conv3_act_mean:.4f}")

    def visualize_feature_space(self, data_loader, num_samples=200):

        device = next(self.model.parameters()).device
        self.model.eval()

        features = []
        labels = []

        with torch.no_grad():
            count = 0
            for data, target in data_loader:
                if count >= num_samples:
                    break

                data = data.to(device)
                feature = self.model.extract_features(data)
                feature = feature.cpu().numpy()

                batch_size = data.size(0)
                features.append(feature[:batch_size])
                labels.append(target.numpy()[:batch_size])
                count += batch_size

        features = np.vstack(features)
        labels = np.hstack(labels)

        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # PCA visualization
        scatter1 = axes[0].scatter(features_pca[:, 0], features_pca[:, 1],
                                   c=labels, cmap='tab10', alpha=0.7, s=30)
        axes[0].set_xlabel('PCA Dimension 1')
        axes[0].set_ylabel('PCA Dimension 2')
        axes[0].set_title('PCA Feature Space Visualization')
        axes[0].grid(True, alpha=0.3)

        unique_labels = np.unique(labels)
        class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            class_points = features_pca[labels == label]
            centroid = np.mean(class_points, axis=0)

            axes[1].scatter(centroid[0], centroid[1],
                            color=class_colors[i], s=200,
                            marker='*', label=f'Class {label} Center')

            # Calculate intra-class distance
            if len(class_points) > 1:
                distances = np.linalg.norm(class_points - centroid, axis=1)
                avg_distance = np.mean(distances)

                circle = plt.Circle(centroid, avg_distance,
                                    color=class_colors[i], alpha=0.2)
                axes[1].add_patch(circle)

        axes[1].set_xlabel('PCA Dimension 1')
        axes[1].set_ylabel('PCA Dimension 2')
        axes[1].set_title('Class Separation Analysis')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.colorbar(scatter1, ax=axes[0], label='Class')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '4_feature_space.png'), dpi=150)
        plt.close()
        print(f"Saved: {self.output_dir}/4_feature_space.png")

        if len(unique_labels) > 1:
            intra_distances = []
            inter_distances = []

            for label in unique_labels:

                class_points = features[labels == label]
                if len(class_points) > 1:
                    centroid = np.mean(class_points, axis=0)
                    distances = np.linalg.norm(class_points - centroid, axis=1)
                    intra_distances.append(np.mean(distances))

                for other_label in unique_labels:
                    if other_label > label:
                        other_points = features[labels == other_label]
                        if len(other_points) > 0:
                            centroid1 = np.mean(class_points, axis=0)
                            centroid2 = np.mean(other_points, axis=0)
                            distance = np.linalg.norm(centroid1 - centroid2)
                            inter_distances.append(distance)

            if intra_distances and inter_distances:
                separation_ratio = np.mean(inter_distances) / np.mean(intra_distances)
                print(f"Class separation ratio: {separation_ratio:.2f}")

                if separation_ratio < 1.5:
                    print("Warning: Poor class separation in feature space")
                elif separation_ratio > 3.0:
                    print("Good: Clear class separation in feature space")

    def visualize_confusion_matrix(self, data_loader):
        device = next(self.model.parameters()).device
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(device)
                output = self.model(data)
                _, predicted = output.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.numpy())

        cm = confusion_matrix(all_targets, all_preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Circle', 'Rectangle', 'Triangle', 'Cross'],
                    yticklabels=['Circle', 'Rectangle', 'Triangle', 'Cross'])

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix Analysis')

        accuracy = np.trace(cm) / np.sum(cm) * 100

        class_accuracies = []
        for i in range(len(cm)):
            if np.sum(cm[i]) > 0:
                class_acc = cm[i, i] / np.sum(cm[i]) * 100
                class_accuracies.append(class_acc)

        confusion_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append((i, j, cm[i, j]))

        confusion_pairs.sort(key=lambda x: x[2], reverse=True)

        info_text = f'Overall Accuracy: {accuracy:.1f}%\n'
        info_text += f'Class Accuracies: {", ".join([f"{acc:.1f}%" for acc in class_accuracies])}'

        if confusion_pairs:
            label_names = ['Circle', 'Rectangle', 'Triangle', 'Cross']
            top_confusion = confusion_pairs[0]
            info_text += f'\nMost confused: {label_names[top_confusion[0]]}→{label_names[top_confusion[1]]} ({top_confusion[2]} times)'

        plt.figtext(0.5, 0.01, info_text,
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '5_confusion_matrix.png'), dpi=150)
        plt.close()
        print(f"Saved: {self.output_dir}/5_confusion_matrix.png")

        return cm, accuracy, class_accuracies

    def generate_summary_report(self, history, cm, accuracy, class_accuracies):

        report_path = os.path.join(self.output_dir, 'summary_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DEEP LEARNING MODEL DIAGNOSTIC REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("1. MODEL INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model Architecture: SimpleDiagnosticCNN\n")
            f.write(f"Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n")
            f.write(f"Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n\n")

            f.write("2. TRAINING RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Training Epochs: {len(history['train_loss'])}\n")
            f.write(f"Final Training Accuracy: {history['train_acc'][-1]:.1f}%\n")
            f.write(f"Final Test Accuracy: {history['test_acc'][-1]:.1f}%\n")
            f.write(f"Confusion Matrix Accuracy: {accuracy:.1f}%\n")
            f.write(f"Class-wise Accuracies: {', '.join([f'{acc:.1f}%' for acc in class_accuracies])}\n\n")

            f.write("3. DIAGNOSTIC VISUALIZATIONS\n")
            f.write("-" * 40 + "\n")
            f.write("The following diagnostic visualizations have been generated:\n\n")
            f.write("  1. training_history.png - Training process monitoring\n")
            f.write("     - Shows loss and accuracy curves over epochs\n")
            f.write("     - Helps detect overfitting/underfitting\n\n")

            f.write("  2. conv_filters.png - Convolutional filter visualization\n")
            f.write("     - Shows what patterns the first layer learns\n")
            f.write("     - Helps diagnose weight initialization issues\n\n")

            f.write("  3. feature_maps.png - Feature map analysis\n")
            f.write("     - Shows activations at different network depths\n")
            f.write("     - Reveals hierarchical feature learning\n\n")

            f.write("  4. feature_space.png - Feature space visualization\n")
            f.write("     - Shows how data is represented internally\n")
            f.write("     - Helps understand class separability\n\n")

            f.write("  5. confusion_matrix.png - Classification error analysis\n")
            f.write("     - Shows which classes are confused\n")
            f.write("     - Identifies systematic errors\n\n")

            f.write("4. DIAGNOSTIC INSIGHTS\n")
            f.write("-" * 40 + "\n")

            if len(history['test_loss']) > 5:
                if history['test_loss'][-1] > history['test_loss'][-5]:
                    f.write("- Possible overfitting: Test loss increasing\n")
                else:
                    f.write("- Good generalization: Test loss decreasing\n")

            if accuracy > 90:
                f.write("- Excellent classification performance\n")
            elif accuracy > 75:
                f.write("- Good classification performance\n")
            else:
                f.write("- Classification performance needs improvement\n")

            if len(class_accuracies) > 0:
                min_acc = min(class_accuracies)
                max_acc = max(class_accuracies)
                if max_acc - min_acc > 20:
                    f.write("- Significant class imbalance detected\n")
                else:
                    f.write("- Relatively balanced class performance\n")

            f.write("5. TECHNICAL DETAILS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n")
            f.write(f"Output directory: {os.path.abspath(self.output_dir)}\n")
            f.write("Library versions:\n")
            f.write(f"  - PyTorch: {torch.__version__}\n")
            f.write(f"  - NumPy: {np.__version__}\n")
            f.write(f"  - Matplotlib: {matplotlib.__version__}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF DIAGNOSTIC REPORT\n")
            f.write("=" * 60 + "\n")

        print(f"Saved: {report_path}")
