import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, LSUN
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.adv_cifar10 import AdvCifar10, GradientMatchingTargetCifar10, PoisonCIFAR10
from torch.utils.data import Subset
import numpy as np

def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        tran_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])

    if config.data.dataset == 'CIFAR10':
        if args.adv:
            dataset = AdvCifar10(os.path.join(args.exp, 'datasets', 'cifar10'), train=True, download=True,
                            transform=tran_transform, args=args, config=config)
            test_dataset = AdvCifar10(os.path.join(args.exp, 'datasets', 'cifar10_test'), train=False, download=True,
                            transform=test_transform, args=args, config=config)
            target_dataset = GradientMatchingTargetCifar10(os.path.join(args.exp, 'datasets', 'cifar10'), train=True, download=True,
                            transform=tran_transform, args=args, config=config)
            target_test_dataset = GradientMatchingTargetCifar10(os.path.join(args.exp, 'datasets', 'cifar10_test'), train=False, download=True,
                            transform=test_transform, args=args, config=config)
            
            return dataset, test_dataset, target_dataset, target_test_dataset
        elif args.poison:
            dataset = PoisonCIFAR10(os.path.join(args.exp, 'datasets', 'cifar10'), train=True, download=True,
                            transform=tran_transform, args=args, config=config)
            test_dataset = PoisonCIFAR10(os.path.join(args.exp, 'datasets', 'cifar10_test'), train=False, download=True,
                            transform=test_transform, args=args, config=config)
        else:
            dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10'), train=True, download=True,
                            transform=tran_transform)
            test_dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10_test'), train=False, download=True,
                               transform=test_transform)
            if config.data.sub_dataset:
                if config.data.subset_number == 0:
                    sub_class = [2, 7, 9]
                    single_class_num = 3000
                    np_targets = np.array(dataset.targets)
                    new_data_list = []
                    new_targets_list = []
                    for i in range(len(sub_class)):
                        sub_class_idx = np_targets == sub_class[i]
                        new_data_list.append(dataset.data[sub_class_idx][:single_class_num])
                        new_targets_list.append(np_targets[sub_class_idx][:single_class_num])
                    dataset.data = np.concatenate(new_data_list, axis=0)
                    dataset.targets = np.concatenate(new_targets_list, axis=0)
                    
                    for i in range(len(sub_class)):
                        sub_class_idx = dataset.targets == sub_class[i]
                        dataset.targets[sub_class_idx] = i

                    # print(dataset.targets.shape)
                    # print(dataset.data.shape)
                    # input("check")

                else:
                    raise("unknown subset_number")

    elif config.data.dataset == 'CELEBA':
        if config.data.random_flip:
            dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), download=True)
        else:
            dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

        test_dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba_test'), split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), download=True)


    elif config.data.dataset == 'LSUN':
        train_folder = '{}_train'.format(config.data.category)
        val_folder = '{}_val'.format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                             ]))
        else:
            dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

        test_dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[val_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ]), resolution=config.data.image_size)
        else:
            dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.ToTensor(),
                           resolution=config.data.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X

def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)
