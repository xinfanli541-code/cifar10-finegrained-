"""CIFAR-10 data loading for the fine-grained hard-class project.

Reuses the HuggingFace parquet data from the previous project
(data/ -> symlink to ~/workspace/pytorch-image-classification/data).
45k train / 5k val split, fixed seed 42, same as before.

Augmentation variants (single variable under test, ResNet18 / Adam):
  base   : RandomCrop(32, padding=4) + RandomHorizontalFlip   (prior resnet18 run)
  none   : no augmentation
  strong : base + ColorJitter + RandomRotation(15)
"""

import io
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

plain_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
test_transform = plain_transform

AUG_TRANSFORMS = {
    'base': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]),
    'none': plain_transform,
    'strong': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]),
}


class ParquetCIFAR10(Dataset):
    def __init__(self, parquet_path, transform=None):
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        self.labels = table.column('label').to_pylist()
        img_col = table.column('img').to_pylist()
        self.blobs = [x['bytes'] if isinstance(x, dict) else x for x in img_col]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.blobs[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _parquet_paths(data_root):
    d = os.path.join(data_root, 'parquet')
    train_p = os.path.join(d, 'train-00000-of-00001.parquet')
    test_p = os.path.join(d, 'test-00000-of-00001.parquet')
    if os.path.exists(train_p) and os.path.exists(test_p):
        return train_p, test_p
    return None, None


def get_loaders(data_root='./data', batch_size=128, num_workers=2,
                seed=42, aug='base', train_limit=None):
    tf_train = AUG_TRANSFORMS[aug]
    train_p, test_p = _parquet_paths(data_root)
    if train_p:
        print(f'loading CIFAR-10 from parquet: {train_p}')
        train_aug = ParquetCIFAR10(train_p, tf_train)
        train_plain = ParquetCIFAR10(train_p, plain_transform)
        test_set = ParquetCIFAR10(test_p, test_transform)
    else:
        train_aug = datasets.CIFAR10(root=data_root, train=True,
                                     download=True, transform=tf_train)
        train_plain = datasets.CIFAR10(root=data_root, train=True,
                                       download=True, transform=plain_transform)
        test_set = datasets.CIFAR10(root=data_root, train=False,
                                    download=True, transform=test_transform)
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_aug), generator=gen)
    train_idx = indices[:45000].tolist()
    if train_limit:
        train_idx = train_idx[:train_limit]
    train_set = Subset(train_aug, train_idx)
    val_set = Subset(train_plain, indices[45000:].tolist())
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
