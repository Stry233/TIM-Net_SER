import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms

from DeepSVDD.base import base_dataset


def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


class SERDataset(base_dataset.BaseADDataset):

    def __init__(self, x_train, y_train, x_test, y_test, normal_class=0, random_state=42):
        super().__init__(root="")

        self.train_set = np.array(x_train)
        self.test_set = np.array(x_test)
        self.train_labels = np.argmax(np.array(y_train), axis=1)
        self.test_labels = np.argmax(np.array(y_test), axis=1)

        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        self.normal_class = normal_class

        self.train_set = MyMFCCDataset({'inputs': self.train_set, 'labels': self.train_labels})
        self.test_set = MyMFCCDataset({'inputs': self.test_set, 'labels': self.test_labels},
                                      target_transform=self.target_transform)

        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(self.train_labels, self.normal_class)
        self.train_set = Subset(self.train_set, train_idx_normal)
        # print(len(self.train_set))

    def __len__(self):
        return len(self.train_set)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) \
            -> (DataLoader, DataLoader):

        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)

        return train_loader, test_loader


class MyMFCCDataset(Dataset):
    def __init__(self, data, target_transform=None):
        self.data = data['inputs']
        self.targets = data['labels']
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, targets = self.data[idx], self.targets[idx]
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return data, targets, idx
