import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_transforms(train_augment = False):
    """
    ---return---
    valid_trainsform , train_transform
    """
    #mobilenet normalize
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    valid_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(), 
                                        normalize])
    # train augment
    if train_augment:
        train_transform = transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    # train no augment
    else:
        train_transform = transforms.Compose([transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])

    return valid_transform, train_transform


def get_train_valid_loader(data_dir, batch_size, valid_size = 0.2, augment = True, shuffle = True, num_workers = 4):
    """
    ---return---
    train_dataloader , valid_dataloader
    """

    valid_transform, train_transform = get_transforms(train_augment=augment)
    
    #데이터 셋 로드
    train_dataset = datasets.ImageFolder(
        root=data_dir, transform=train_transform,
    )

    valid_dataset = datasets.ImageFolder(
        root=data_dir, transform=valid_transform,
    )

    class_names = train_dataset.classes


    #split 설정
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        # np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle = False, drop_last = True,
        num_workers=num_workers
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler, shuffle = False, drop_last = True,
        num_workers=num_workers
    )

    return train_loader, valid_loader