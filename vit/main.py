import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as T

import os
import argparse

from models import ViT, ViT_Small
from utils.loss_fn import CrossEntropyLS

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument(
    '--resume', '-r', action='store_true', help='Resume from checkpoint'
)
parser.add_argument(
    '--net',
    choices=['vit', 'vit_small', 'vit_timm'],
    default='vit',
    type=str,
    help="Which model to use",
)
parser.add_argument('--lr', default=8e-4, type=float, help="Learning rate")
parser.add_argument(
    '--cos', action='store_true', help='Train with cosine annealing scheduling'
)
parser.add_argument('--label-smooth', default=0.2, type=float, help="Label smoothing")
parser.add_argument('--optim', default='adam', type=str, help="adam, sgd")
parser.add_argument('--weight-decay', default=1e-4, type=float, help="Weight decay")
parser.add_argument('--batch-size', default=256, type=int, help="Batch size")
parser.add_argument('--imsize', default=32, type=int, help="Image size")
parser.add_argument('--aug', action='store_true', help="Use random aug")
parser.add_argument('--ra-n', default=2, type=int, help="Random augmentation n")
parser.add_argument('--ra-m', default=14, type=int, help="Random augmentation m")
parser.add_argument('--amp', action='store_true', help="Enable AMP training")
parser.add_argument('--max-epoch', default=200, type=int, help="Max training epochs")
parser.add_argument('--patch-size', default=4, type=int, help="Patch size")
parser.add_argument('--dim', default=512, type=int, help="Dimension")

args = parser.parse_args()

use_amp = args.amp
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_acc_5 = 0  # best top-5 test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data...')
if args.net == "vit_timm":
    imsize = 224
else:
    imsize = args.imsize
transform_train = T.Compose(
    [
        T.RandomCrop(32, padding=4),
        T.Resize(imsize),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
)

transform_test = T.Compose(
    [
        T.Resize(imsize),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
)

# Random augmentation
if args.aug:
    transform_train.transforms.insert(0, T.RandAugment(args.ra_n, args.ra_m))

train_set = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train
)
train_batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=train_batch_size, shuffle=True, num_workers=8
)

test_set = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test
)
test_batch_size = 256
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=test_batch_size, shuffle=False, num_workers=8
)

# Model
print(f'==> Building model for {args.net}...')
if args.net == 'vit':
    net = ViT(
        image_size=imsize,
        patch_size=args.patch_size,
        num_classes=100,
        dim=args.dim,
        depth=6,
        heads=args.dim // 64,
        mlp_dim=args.dim * 2,
        dropout=0.1,
        emb_dropout=0.1,
    )
elif args.net == 'vit_small':
    net = ViT_Small(
        image_size=imsize,
        patch_size=args.patch_size,
        num_classes=100,
        dim=args.dim,
        depth=6,
        heads=args.dim // 64,
        mlp_dim=args.dim * 2,
        dropout=0.1,
        emb_dropout=0.1,
    )
elif args.net == "vit_timm":
    import timm

    net = timm.create_model("vit_base_patch16_224", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 100)

if device == 'cuda':
    net = torch.nn.DataParallel(net)  # make parallel
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint...')
    assert os.path.isdir('checkpoints'), 'Error: no checkpoints directory found!'
    checkpoint = torch.load(f'./checkpoints/{args.net}-ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    scaler.load_state_dict(checkpoint['scaler'])
    best_acc = checkpoint['acc']
    best_acc_5 = checkpoint['acc_5']
    start_epoch = checkpoint['epoch']

# Citerion and optimizer
criterion = CrossEntropyLS(args.label_smooth)
if args.optim == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optim == "sgd":
    optimizer = optim.SGD(
        net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )

# Use cosine or reduce LR on Plateau scheduling
if not args.cos:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, verbose=True, min_lr=1e-3 * 1e-5, factor=0.1
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)


# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    correct_5 = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        total += targets.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        _, predicted_5 = outputs.topk(5, 1, True, True)
        correct_5 += (
            predicted_5.eq(targets.view(-1, 1).expand_as(predicted_5)).sum().item()
        )

    epoch_loss = train_loss / ((batch_idx + 1) * train_batch_size)
    epoch_acc = 100.0 * correct / total
    epoch_acc_5 = 100.0 * correct_5 / total

    return epoch_loss, epoch_acc, epoch_acc_5


# Validation
def test(epoch):
    global best_acc
    global best_acc_5
    net.eval()
    test_loss = 0
    correct = 0
    correct_5 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            _, predicted_5 = outputs.topk(5, 1, True, True)
            correct_5 += (
                predicted_5.eq(targets.view(-1, 1).expand_as(predicted_5)).sum().item()
            )

    epoch_loss = test_loss / ((batch_idx + 1) * test_batch_size)
    epoch_acc = 100.0 * correct / total
    epoch_acc_5 = 100.0 * correct_5 / total

    # Save checkpoint
    if epoch_acc > best_acc:
        state = {
            'net': net.state_dict(),
            'scaler': scaler.state_dict(),
            'acc': epoch_acc,
            'acc_5': epoch_acc_5,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, f'./checkpoints/{args.net}-{args.patch_size}-ckpt.t7')
        best_acc = epoch_acc
        best_acc_5 = epoch_acc_5

    return epoch_loss, epoch_acc, epoch_acc_5


# Main loop
if not os.path.isdir('runs'):
    os.mkdir('runs')
writer = SummaryWriter(log_dir='./runs')

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
lr_schedule = []
for epoch in range(start_epoch, start_epoch + args.max_epoch):
    train_loss, train_acc, train_acc_5 = train(epoch)
    test_loss, test_acc, test_acc_5 = test(epoch)
    if args.cos:
        scheduler.step()
    else:
        scheduler.step(test_loss)

    # Print log info
    print("============================================================")
    print(f"Epoch: {epoch + 1}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Acc@5: {train_acc_5:.2f}%"
    )
    print(
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Test Acc@5: {test_acc_5:.2f}%"
    )
    print(f"Best Acc: {best_acc:.2f}% | Best Acc@5: {best_acc_5:.2f}%")
    print("============================================================")

    # Logging
    # train_losses.append(train_loss)
    # train_accuracies.append(train_acc)
    # test_losses.append(test_loss)
    # test_accuracies.append(test_acc)
    # lr_schedule.append(optimizer.param_groups[0]['lr'])
    writer.add_scalars('Loss', {'train': train_loss, 'test': test_loss}, epoch)
    writer.add_scalars('Accuracy', {'train': train_acc, 'test': test_acc}, epoch)
    writer.add_scalars('Accuracy@5', {'train': train_acc_5, 'test': test_acc_5}, epoch)


# Plot
# plt.figure(figsize=(16, 8))

# plt.subplot2grid((2, 4), (0, 0), colspan=3)
# plt.plot(train_losses, label='train')
# plt.plot(test_losses, label='test')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')

# plt.subplot2grid((2, 4), (1, 0), colspan=3)
# plt.plot(train_accuracies, label='train')
# plt.plot(test_accuracies, label='test')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')

# plt.subplot2grid((2, 4), (0, 3), rowspan=2)
# plt.plot(lr_schedule, label='lr')
# plt.legend()
# plt.title('Learning Rate')
# plt.xlabel('Epoch')

# if not os.path.isdir('logs'):
#     os.mkdir('logs')
# plt.savefig(f'./logs/{args.net}.png')
