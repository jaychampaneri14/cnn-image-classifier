"""
CNN Image Classifier
Custom CNN for multi-class image classification with Grad-CAM visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import warnings
warnings.filterwarnings('ignore')

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


class CNN(nn.Module):
    """Custom CNN with 3 conv blocks + global avg pooling."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.3)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
        )
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        # Grad-CAM hook
        self.gradients = None
        self.activations = None
        self.block3[-2].register_forward_hook(self._save_activations)
        self.block3[-2].register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

    def grad_cam(self, x, class_idx=None):
        """Generate Grad-CAM heatmap."""
        self.eval()
        x.requires_grad_(True)
        output = self(x)
        if class_idx is None:
            class_idx = output.argmax(1).item()
        self.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        pooled_grads = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (self.activations * pooled_grads).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam, class_idx


def get_loaders(batch_size=128):
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    try:
        tr = datasets.CIFAR10('./data', train=True,  download=True, transform=train_t)
        te = datasets.CIFAR10('./data', train=False, download=True, transform=test_t)
        return (DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0),
                DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=0), True)
    except:
        Xtr = torch.randn(2000, 3, 32, 32); ytr = torch.randint(0, 10, (2000,))
        Xte = torch.randn(400,  3, 32, 32); yte = torch.randint(0, 10, (400,))
        return (DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True),
                DataLoader(TensorDataset(Xte, yte), batch_size=batch_size), False)


def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item() * len(y)
        correct  += (out.argmax(1) == y).sum().item()
        total    += len(y)
    return loss_sum/total, correct/total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss = criterion(out, y)
        preds = out.argmax(1)
        loss_sum += loss.item() * len(y)
        correct  += (preds == y).sum().item()
        total    += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return loss_sum/total, correct/total, np.array(all_preds), np.array(all_labels)


def plot_gradcam(model, images, labels, device, save_path='gradcam.png'):
    """Visualize Grad-CAM for sample images."""
    model.eval()
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])
    for i in range(6):
        img = images[i:i+1].to(device)
        cam, pred_cls = model.grad_cam(img)
        img_np = images[i].permute(1,2,0).cpu().numpy()
        img_np = (img_np * std + mean).clip(0,1)

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f'True: {CLASSES[labels[i]]}', fontsize=8)
        axes[0, i].axis('off')

        axes[1, i].imshow(cam, cmap='hot')
        axes[1, i].set_title(f'Pred: {CLASSES[pred_cls]}', fontsize=8)
        axes[1, i].axis('off')

        overlay = img_np.copy()
        heatmap = plt.cm.jet(cam)[:, :, :3]
        axes[2, i].imshow(0.6 * img_np + 0.4 * heatmap)
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel('Image', rotation=90, fontsize=9)
    axes[1, 0].set_ylabel('Grad-CAM', rotation=90, fontsize=9)
    axes[2, 0].set_ylabel('Overlay', rotation=90, fontsize=9)
    plt.suptitle('Grad-CAM Visualization', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Grad-CAM saved to {save_path}")


def main():
    print("=" * 60)
    print("CNN IMAGE CLASSIFIER WITH GRAD-CAM")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, test_loader, real_data = get_loaders()
    model     = CNN().to(device)
    params    = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    n_epochs  = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    history = {'tr_acc': [], 'va_acc': []}
    best_acc = 0

    print(f"\n--- Training for {n_epochs} epochs ---")
    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        history['tr_acc'].append(tr_acc)
        history['va_acc'].append(va_acc)
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), 'best_cnn_model.pth')
        print(f"Epoch {epoch:2d}: Train={tr_acc:.4f} | Val={va_acc:.4f}")

    print(f"\nBest accuracy: {best_acc:.4f}")

    # Grad-CAM on test samples
    if real_data:
        test_images, test_labels = next(iter(test_loader))
        model.load_state_dict(torch.load('best_cnn_model.pth', map_location=device))
        plot_gradcam(model, test_images, test_labels.numpy(), device)

    # Training plot
    plt.figure(figsize=(8, 5))
    plt.plot(history['tr_acc'], label='Train Acc')
    plt.plot(history['va_acc'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('CNN Training History'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()

    print("Model saved to best_cnn_model.pth")
    print("\n✓ CNN Image Classifier complete!")


if __name__ == '__main__':
    main()
