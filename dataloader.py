import torch, torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random, copy
import argparse
import numpy as np
import pandas as pd

class VisionDataset(object):

    def __init__(self, opt, class_order=None):
        self.kwargs = {
        'num_workers': opt.workers,
        'batch_size': opt.batch_size,
        'shuffle': True,
        'pin_memory': True}
        self.opt = opt
        # Sets parameters of the dataset. For adding new datasets, please add the dataset details in `get_statistics` function.
        mean, std, opt.total_num_classes, opt.inp_size, opt.in_channels = get_statistics(opt.dataset)
        self.class_order = class_order
        # Generates the standard data augmentation transforms
        train_augment, test_augment = get_augment_transforms(dataset=opt.dataset, inp_sz=opt.inp_size)
        self.train_transforms = torchvision.transforms.Compose(train_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
        self.test_transforms = torchvision.transforms.Compose(test_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

        # Creates the supervised baseline dataloader (upper bound for continual learning methods)
        self.supervised_trainloader = self.get_loader(indices=None, transforms=self.train_transforms, train=True)
        self.supervised_testloader = self.get_loader(indices=None, transforms=self.test_transforms, train=False)
        self.kwargs['shuffle'] = False
        self.n = 1
        self.trainidx = []
        self.testidx = []
        self.encountered_classes = []
        self.total_tasks = []


    def get_loader(self, indices, transforms, train, shuffle=True, target_transforms=None):
        sampler = None
        if indices is not None: sampler = SubsetRandomSampler(indices) if (shuffle and train) else SubsetSequentialSampler(indices)

        # Support for *some* pytorch default loaders is provided. Code is made such that adding new datasets is super easy, given they are in ImageFolder format.
        if self.opt.dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'KMNIST', 'FashionMNIST']:
            return DataLoader(getattr(torchvision.datasets, self.opt.dataset)(root=self.opt.data_dir, train=train, download=True, transform=transforms, target_transform=target_transforms), sampler=sampler, **self.kwargs)
        elif self.opt.dataset=='SVHN':
            split = 'train' if train else 'test'
            return DataLoader(getattr(torchvision.datasets, self.opt.dataset)(root=self.opt.data_dir, split=split, download=True, transform=transforms, target_transform=target_transforms), sampler=sampler, **self.kwargs)
        else:
            subfolder = 'train' if train else 'test' # ImageNet 'val' is labled as 'test' here.
            return DataLoader(torchvision.datasets.ImageFolder(self.opt.data_dir+self.opt.dataset+'/'+subfolder, transform=transforms, target_transform=target_transforms), sampler=sampler, **self.kwargs)

    def gen_cl_mapping(self, task_id, tasks_list=None):
        # Get the label -> idx mapping dictionary
        train_class_labels_dict, test_class_labels_dict = classwise_split(targets=self.supervised_trainloader.dataset.targets), classwise_split(targets=self.supervised_testloader.dataset.targets)

        if self.opt.scenario == 'realistic':
            print('Realistic scenario')
            self.n, self.encountered_classes, self.trainidx, self.testidx = RealCL(self.opt, train_class_labels_dict, test_class_labels_dict, task_id, self.encountered_classes, self.trainidx, self.testidx,self.total_tasks, self.n)
            continual_target_transform = ReorderTargets(self.encountered_classes)
            assert (len(self.trainidx) <= self.opt.memory_size), "ERROR: Cannot exceed max. memory samples!"
            print("Total Memory Samples: {}".format(len(self.trainidx)))
            print("Total Test Samples: {}".format(len(self.testidx)))

            self.cltrain_loader = self.get_loader(indices=self.trainidx, transforms=self.train_transforms, train=True,
                                                  target_transforms=continual_target_transform)
            self.cltest_loader = self.get_loader(indices=self.testidx, transforms=self.test_transforms, train=False,
                                                target_transforms=continual_target_transform)


        elif self.opt.scenario == 'semirealistic':
            print('Semi-Realistic scenario')
            self.total_tasks = tasks_list
            flattened_list = flatten_list(tasks_list)
            self.n, self.encountered_classes, self.trainidx, self.testidx = semireal(self.opt, train_class_labels_dict, test_class_labels_dict, task_id, self.encountered_classes, self.trainidx, self.testidx,self.total_tasks, self.n)
            continual_target_transform = ReorderTargets(flattened_list)
            assert (len(self.trainidx) <= self.opt.memory_size), "ERROR: Cannot exceed max. memory samples!"
            print("Total Memory Samples: {}".format(len(self.trainidx)))
            print("Total Test Samples: {}".format(len(self.testidx)))

            self.cltrain_loader = self.get_loader(indices=self.trainidx, transforms=self.train_transforms, train=True,
                                                  target_transforms=continual_target_transform)
            self.cltest_loader = self.get_loader(indices=self.testidx, transforms=self.test_transforms, train=False,
                                     target_transforms=continual_target_transform)

        else:
            print('Unrealistic scenario')
            # Sets classes to be 0 to n-1 if class order is not specified, else sets it to class order. To produce different effects tweak here.
            class_list = self.class_order if self.class_order is not None else list(range(self.opt.total_num_classes))
            cl_class_list = class_list[:self.opt.num_classes_per_task * self.opt.num_tasks]
            assert (self.opt.num_tasks * self.opt.num_classes_per_task <= self.opt.total_num_classes), "num_classes lesser than classes_per_task * num_tasks"
            if self.class_order is None: random.shuffle(cl_class_list)
            self.total_tasks = [cl_class_list[i * self.opt.num_classes_per_task:(i + 1) * self.opt.num_classes_per_task] for i in range((len(cl_class_list) + self.opt.num_classes_per_task - 1) // self.opt.num_classes_per_task)]
            if task_id in range(len(self.total_tasks)):
                self.trainidx = []
                self.testidx = []
                num_samp = self.opt.memory_size // (self.opt.num_classes_per_task * self.n)
                for tsk in range(0, task_id + 1):
                    for class_name in self.total_tasks[tsk]:
                        self.trainidx += train_class_labels_dict[class_name][:num_samp]
                        self.testidx += test_class_labels_dict[class_name][:]
                        if class_name not in self.encountered_classes:
                            self.encountered_classes.append(class_name)

                self.n += 1
            continual_target_transform = ReorderTargets(cl_class_list)  # Remaps the class order to a 0-n order, required for crossentropy loss using class list
            assert(len(self.trainidx) <= self.opt.memory_size), "ERROR: Cannot exceed max. memory samples!"
            print("Total Memory Samples: {}".format(len(self.trainidx)))
            print("Total Test Samples: {}".format(len(self.testidx)))
            self.cltrain_loader = self.get_loader(indices=self.trainidx, transforms=self.train_transforms, train=True, target_transforms=continual_target_transform)
            self.cltest_loader = self.get_loader(indices=self.testidx, transforms=self.test_transforms, train=False, target_transforms=continual_target_transform)


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class ReorderTargets(object):
    """
    Converts the class-orders to 0 -- (n-1) irrespective of order passed.
    """
    def __init__(self, class_order):
        self.class_order = np.array(class_order)

    def __call__(self, target):
        return np.where(self.class_order==target)[0][0]

def get_augment_transforms(dataset, inp_sz):
    """
    Returns appropriate augmentation given dataset size and name
    Arguments:
        indices (sequence): a sequence of indices
    """
    train_augment = []
    test_augment = []
    if inp_sz == 32 or inp_sz == 28 or inp_sz == 64:
        if inp_sz == 64:
            train_augment.append(torchvision.transforms.Resize(224))
            test_augment.append(torchvision.transforms.Resize(224))
        else:
            train_augment.append(torchvision.transforms.RandomCrop(inp_sz, padding=4))
    else:
        train_augment.append(torchvision.transforms.RandomResizedCrop(inp_sz))
        test_augment.append(torchvision.transforms.Resize(inp_sz + 32))
        test_augment.append(torchvision.transforms.CenterCrop(inp_sz))

    if dataset not in ['MNIST', 'SVHN', 'KMNIST']:
        train_augment.append(torchvision.transforms.RandomHorizontalFlip())

    return train_augment, test_augment


def classwise_split(targets):
    """
    Returns a dictionary with classwise indices for any class key given labels array.
    Arguments:
        indices (sequence): a sequence of indices
    """
    targets = np.array(targets)
    indices = targets.argsort()
    class_labels_dict = dict()

    for idx in indices:
        if targets[idx] in class_labels_dict: class_labels_dict[targets[idx]].append(idx)
        else: class_labels_dict[targets[idx]] = [idx]

    return class_labels_dict

def get_statistics(dataset):
    '''
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    '''
    assert(dataset in ['MNIST', 'KMNIST', 'EMNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'CINIC10', 'ImageNet100', 'ImageNet', 'TinyImagenet', 'CUB200'])
    mean = {
            'MNIST':(0.1307,),
            'KMNIST':(0.1307,),
            'EMNIST':(0.1307,),
            'FashionMNIST':(0.1307,),
            'SVHN':  (0.4377,  0.4438,  0.4728),
            'CIFAR10':(0.4914, 0.4822, 0.4465),
            'CIFAR100':(0.5071, 0.4867, 0.4408),
            'CINIC10':(0.47889522, 0.47227842, 0.43047404),
            'TinyImagenet':(0.4802, 0.4481, 0.3975),
            'ImageNet100':(0.485, 0.456, 0.406),
            'ImageNet':(0.485, 0.456, 0.406),
            'CUB200': (0.485, 0.456, 0.406),
        }

    std = {
            'MNIST':(0.3081,),
            'KMNIST':(0.3081,),
            'EMNIST':(0.3081,),
            'FashionMNIST':(0.3081,),
            'SVHN': (0.1969,  0.1999,  0.1958),
            'CIFAR10':(0.2023, 0.1994, 0.2010),
            'CIFAR100':(0.2675, 0.2565, 0.2761),
            'CINIC10':(0.24205776, 0.23828046, 0.25874835),
            'TinyImagenet':(0.2302, 0.2265, 0.2262),
            'ImageNet100':(0.229, 0.224, 0.225),
            'ImageNet':(0.229, 0.224, 0.225),
            'CUB200': (0.229, 0.224, 0.225),
        }

    classes = {
            'MNIST': 10,
            'KMNIST': 10,
            'EMNIST': 49,
            'FashionMNIST': 10,
            'SVHN': 10,
            'CIFAR10': 10,
            'CIFAR100': 100,
            'CINIC10': 10,
            'TinyImagenet':200,
            'ImageNet100':100,
            'ImageNet': 1000,
            'CUB200': 200,
        }

    in_channels = {
            'MNIST': 1,
            'KMNIST': 1,
            'EMNIST': 1,
            'FashionMNIST': 1,
            'SVHN': 3,
            'CIFAR10': 3,
            'CIFAR100': 3,
            'CINIC10': 3,
            'TinyImagenet':3,
            'ImageNet100':3,
            'ImageNet': 3,
            'CUB200': 3,
        }

    inp_size = {
            'MNIST': 28,
            'KMNIST': 28,
            'EMNIST': 28,
            'FashionMNIST': 28,
            'SVHN': 32,
            'CIFAR10': 224,
            'CIFAR100': 224,
            'CINIC10': 32,
            'TinyImagenet':64,
            'ImageNet100':224,
            'ImageNet': 224,
            'CUB200': 224,
        }
    return mean[dataset], std[dataset], classes[dataset],  inp_size[dataset], in_channels[dataset]


def replace_elements(list_a, list_b, m):
    # Select random samples from the memory and replace them with new task samples
    if m >= len(list_a):
        raise ValueError("m cannot be greater than or equal to the length of List A.")
    indices = random.sample(range(len(list_a)), m)
    copied_list_a = list(list_a)
    for index in sorted(indices, reverse=True):
        del copied_list_a[index]
    selected_elements = list_b[:m]
    for i, element in zip(indices, selected_elements):
        copied_list_a.insert(i, element)
    return copied_list_a

def flatten_list(input_list):
    return [item for sublist in input_list for item in sublist]


def task_groups_realistic(train_class_labels_dict, num_tasks):
    tasks_list = []
    tasks_classes = []
    total_samples = sum(len(samples) for samples in train_class_labels_dict.values())
    samples_per_task = total_samples // num_tasks
    class_labels = list(train_class_labels_dict.keys())
    for _ in range(num_tasks):
        task_samples = []
        selected_classes = []
        while len(task_samples) < samples_per_task:
            cls = random.choice(class_labels)
            if train_class_labels_dict[cls]:
                sample = random.choice(train_class_labels_dict[cls])
                task_samples.append(sample)
                selected_classes.append(cls)
                train_class_labels_dict[cls].remove(sample)
            else:
                # If a class has no more samples, remove it from the list of available classes
                class_labels.remove(cls)
        tasks_list.append(task_samples)
        tasks_classes.append(selected_classes)
    return tasks_list, tasks_classes


def RealCL(opt, train_class_labels_dict, test_class_labels_dict, task_id, encountered_classes, trainidx, testidx,
           total_tasks, n):
    tasks_list, task_class_list = task_groups_realistic(train_class_labels_dict, opt.num_tasks)
    current_task_classes = task_class_list[task_id]
    current_task_samples = tasks_list[task_id]

    if task_id == 0:
        get_task = []
        for class_name in current_task_classes:
            if class_name not in encountered_classes:
                encountered_classes.append(class_name)
                testidx.extend(test_class_labels_dict[class_name])
            get_task.append(class_name)

        if len(current_task_samples) > opt.memory_size:
            trainidx.extend(random.sample(current_task_samples, opt.memory_size))
        else:
            trainidx.extend(current_task_samples)

        total_tasks.append(get_task)
        n += 1
    else:
        get_task = []
        for class_name in current_task_classes:
            if class_name not in encountered_classes:
                encountered_classes.append(class_name)
                testidx.extend(test_class_labels_dict[class_name])
            get_task.append(class_name)

        remaining_space = opt.memory_size - len(trainidx)

        if remaining_space > 0:
            if len(current_task_samples) <= remaining_space:
                trainidx.extend(current_task_samples)
            else:
                unique_samples = random.sample(current_task_samples, remaining_space)
                trainidx.extend(unique_samples)
                remaining_samples = random.sample(current_task_samples, opt.memory_size // n)
                trainidx = replace_elements(trainidx, remaining_samples, len(remaining_samples))
        else:
            unique_samples = random.sample(current_task_samples, opt.memory_size // n)
            trainidx = replace_elements(trainidx, unique_samples, len(unique_samples))

        total_tasks.append(get_task)
        n += 1

    assert len(trainidx) <= opt.memory_size, "ERROR: Cannot exceed max. memory samples!"

    return n, encountered_classes, trainidx, testidx


def semireal(opt, train_class_labels_dict, test_class_labels_dict, task_id, encountered_classes, trainidx, testidx,
             tasks_list, n):
    task_classes = len(tasks_list[task_id])
    num_samp = opt.memory_size // (task_classes * n)
    remaining_space = opt.memory_size - len(trainidx)

    def process_class(class_name, remaining_space, num_samp):
        encountered_classes.append(class_name)
        testidx.extend(test_class_labels_dict[class_name])

        current_num_samp = min(num_samp, len(train_class_labels_dict[class_name]))

        if remaining_space >= current_num_samp:
            trainidx.extend(random.sample(train_class_labels_dict[class_name], current_num_samp))
            return remaining_space - current_num_samp, []
        elif remaining_space > 0:
            trainidx.extend(random.sample(train_class_labels_dict[class_name], remaining_space))
            remain_samp = current_num_samp - remaining_space
            return 0, random.sample(train_class_labels_dict[class_name], remain_samp)
        else:
            return remaining_space, random.sample(train_class_labels_dict[class_name], current_num_samp)

    mem_smp_2 = []

    if task_id == 0:
        for class_name in tasks_list[task_id]:
            remaining_space, class_mem_smp_2 = process_class(class_name, remaining_space, num_samp)
            mem_smp_2.extend(class_mem_smp_2)

        if remaining_space > 0:
            last_class = tasks_list[task_id][-1]
            population = train_class_labels_dict.get(last_class, [])
            if population:
                items_to_add = min(remaining_space, len(population))
                trainidx.extend(random.sample(population, items_to_add))
    else:
        for class_name in tasks_list[task_id]:
            remaining_space, class_mem_smp_2 = process_class(class_name, remaining_space, num_samp)
            mem_smp_2.extend(class_mem_smp_2)
        if mem_smp_2:
            trainidx = replace_elements(trainidx, mem_smp_2, len(mem_smp_2))
    n += 1
    return n, encountered_classes, trainidx, testidx
