import matplotlib
matplotlib.use('Agg')
import SimpleDiagnosticCNN # class for simple CNN
import SimpleDiagnosticTool # class for simple tools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

def set_seed(seed=114514):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(114514)

print("=" * 60)
print("Deep Learning Diagnostic Toolkit - Initialized")
print("=" * 60)

def train_simple_model(model, train_loader, test_loader, num_epochs=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }

    print("Starting model training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)

        print(f'Epoch {epoch + 1:02d}: '
              f'Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.1f}% | '
              f'Test Loss: {avg_test_loss:.4f}, Acc: {test_acc:.1f}%')

        print(f"Training completed! Final test accuracy: {test_acc:.1f}%")
    return history, model


def main():
    print("\n" + "=" * 60)
    print("DEEP LEARNING DIAGNOSTIC DEMONSTRATION")
    print("=" * 60)

    train_loader, test_loader, sample_data, sample_label = SimpleDiagnosticCNN.create_simple_dataset(
        num_samples=800, img_size=28
    )

    print("\n" + "=" * 30)
    print("1. MODEL CREATION AND TRAINING")
    print("=" * 30)

    model = SimpleDiagnosticCNN.SimpleDiagnosticCNN(num_classes=4)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history, trained_model = train_simple_model(
        model, train_loader, test_loader, num_epochs=15
    )

    print("\n" + "=" * 30)
    print("2. MODEL DIAGNOSTIC ANALYSIS")
    print("=" * 30)

    diagnostic_tool = SimpleDiagnosticTool.SimpleDiagnosticTool(trained_model)

    print("\n1) Visualizing training history...")
    diagnostic_tool.visualize_training_history(history)

    print("\n2) Visualizing convolutional filters...")
    diagnostic_tool.visualize_conv_filters()

    print("\n3) Visualizing feature maps...")
    diagnostic_tool.visualize_feature_maps(sample_data)

    print("\n4) Visualizing feature space...")
    diagnostic_tool.visualize_feature_space(test_loader, num_samples=200)

    print("\n5) Visualizing confusion matrix...")
    cm, accuracy, class_accuracies = diagnostic_tool.visualize_confusion_matrix(test_loader)

    print("\n" + "=" * 30)
    print("3. GENERATING DIAGNOSTIC REPORT")
    print("=" * 30)

    diagnostic_tool.generate_summary_report(history, cm, accuracy, class_accuracies)

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 60)

    print(f"\nGenerated visualizations and report saved in: {diagnostic_tool.output_dir}")
    print("\nFile listing:")
    print(f"  1. {diagnostic_tool.output_dir}/1_training_history.png")
    print(f"  2. {diagnostic_tool.output_dir}/2_conv_filters.png")
    print(f"  3. {diagnostic_tool.output_dir}/3_feature_maps.png")
    print(f"  4. {diagnostic_tool.output_dir}/4_feature_space.png")
    print(f"  5. {diagnostic_tool.output_dir}/5_confusion_matrix.png")
    print(f"  6. {diagnostic_tool.output_dir}/summary_report.txt")

if __name__ == "__main__":
    main()