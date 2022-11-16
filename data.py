import os
import h5py
import numpy as np
from torch.utils.data import Dataset
import transforms as transforms

from torchvision import transforms as T


class PointCloudDataset(Dataset):
    def __init__(self, dataDir="./data",  num_points = 1024, partition = "Training", transforms = T.Compose([transforms.Normalize()]), data_type = ["synthetic", "real"], binary_data = False, data_ratio = 1.0):
        self.transforms = transforms
        self.dataDir = dataDir
        self.num_points = num_points
        self.partition = partition
        self.data_type = data_type
        self.binary_data = binary_data
        self.data_ratio = data_ratio

        np.random.seed(1234567890)

        self.data, self.labels = self.load_data()

        if self.binary_data:
            self.labels[self.labels > 0] = 1
            self.print_data_info(self.partition, self.data, self.labels)

        if self.data_ratio < 1.0:
            data = None
            labels = None
            for class_label in np.unique(self.labels):
                data_class = self.data[self.labels==class_label]
                labels_class = self.labels[self.labels==class_label]


                sample_size = int(labels_class.shape[0]*self.data_ratio)
                idx = np.random.choice(labels_class.shape[0], sample_size, replace=False)
                data_tmp = data_class[idx]
                labels_tmp = labels_class[idx]

                if data is None:
                    data = data_tmp
                    labels = labels_tmp
                else:
                    data = np.concatenate((data, data_tmp))
                    labels = np.concatenate((labels, labels_tmp))
            self.data = data
            self.labels = labels

        self.print_data_info(self.partition, self.data, self.labels)
        self.class_weights = self.calculate_label_weights()


    def __getitem__(self, item):
        data = self.data[item][:self.num_points]
        labels = self.labels[item]
        if self.partition == 'Training':
            np.random.shuffle(data)

        if self.transforms:
            data = self.transforms(data)

        return data, labels

    def __len__(self):
        return self.data.shape[0]

    def load_data(self):
        if(self.partition == 'Training' or self.partition == 'Validation'):
            hdf5File = "training_pointcloud_hdf5"
        elif self.partition == "Testing":
            hdf5File = "testing_pointcloud_hdf5"

        data = None
        labels = None
        for datatype in self.data_type:
            print(self.data_type, datatype)
            path = os.path.join(self.dataDir, "{}_{}.h5".format(hdf5File, datatype))
            print(path)

            with h5py.File(path, 'r') as hdf:
                data_tmp = np.asarray(hdf[f'{self.partition}/PointClouds'][:])
                labels_tmp = np.asarray(hdf[f'{self.partition}/Labels'][:])

                print(type(np.asarray(data_tmp)))

                self.print_data_info(self.partition+"-"+datatype, data_tmp, labels_tmp)
                if data is None:
                    data = data_tmp
                    labels = labels_tmp
                else:
                    data = np.concatenate((data, data_tmp))
                    labels = np.concatenate((labels, labels_tmp))

            self.print_data_info(self.partition, data, labels)
        return data, labels


    def print_data_info(self, partition, data, labels):

        unique_labels, unique_counts = np.unique(labels, return_counts = True)
        print(f'\n[{partition} Data]')
        print(f'Data Shape: {data.shape} | Type: {data[0].dtype}')
        print(f'Label Shape: {labels.shape} | Type: {labels[0].dtype}')
        print(f'Labels: {unique_labels} | Counts: {unique_counts}\n')

    
    def calculate_label_weights(self):

        unique_labels, unique_counts = np.unique(self.labels, return_counts = True)
        largest_count = np.max(unique_counts)
        class_weights = largest_count / unique_counts

        return class_weights




