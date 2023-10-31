import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import pandas as pd
import pickle

def get_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type="connect",data_type=None) -> np.array:
    """
    construct adjacent matrix by csv file
    :param distance_file: path of csv file to save the distances between nodes
    :param num_nodes: number of nodes in the graph
    :param id_file: path of txt file to save the order of the nodes
    :param graph_type: ["connect", "distance"] if use weight, please set distance
    :return:
    """
    try:
        with open(distance_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(distance_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', distance_file, ':', e)
        raise
    sensor_ids, sensor_id_to_ind, adj_mx = pickle_data

    return adj_mx


def get_flow_data(flow_file: str) -> np.array:
    """
    parse npz to get flow data
    :param flow_file: (N, T, D)
    :return:
    """
    df = pd.read_hdf(flow_file)
    data = np.expand_dims(df.values, axis=-1)
    flow_data = data.transpose([1, 0, 2])  # [N, T, D]  D = 1
    return flow_data


class PEMSBAYDataset(Dataset):
    def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, train_mode,data_type=None):
        """
        load processed data
        :param data_path: ["graph file name" , "flow data file name"], path to save the data file names
        :param num_nodes: number of nodes in graph
        :param divide_days: [ days of train data, days of test data], list to divide the original data
        :param time_interval: time interval between two traffic data records (mins)
        :param history_length: length of history data to be used
        :param train_mode: ["train", "test"]
        """
        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0]
        self.test_days = divide_days[1]
        self.history_length = history_length  # 6
        self.time_interval = time_interval  # 5 min
        self.one_day_length = int(24 * 60 / self.time_interval)
        self.graph = get_adjacent_matrix(distance_file=data_path[0], num_nodes=num_nodes,data_type=data_type)
        self.flow_norm, self.flow_data = self.pre_process_data(data=get_flow_data(data_path[1]), norm_dim=1)

    def __len__(self):
        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):  # (x, y), index = [0, L1 - 1]
        if self.train_mode == "train":
            index = index
        elif self.train_mode == "test":
            index += self.train_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = PEMSBAYDataset.slice_data(self.flow_data, self.history_length, index, self.train_mode)
        data_x = PEMSBAYDataset.to_tensor(data_x)  # [N, H, D]
        data_y = PEMSBAYDataset.to_tensor(data_y).unsqueeze(1)  # [N, 1, D]
        return {"graph": PEMSBAYDataset.to_tensor(self.graph), "flow_x": data_x, "flow_y": data_y}

    @staticmethod
    def slice_data(data, history_length, index, train_mode):
        """
        :param data: np.array, normalized traffic data.
        :param history_length: int, length of history data to be used.
        :param index: int, index on temporal axis.
        :param train_mode: str, ["train", "test"].
        :return:
            data_x: np.array, [N, H, D].
            data_y: np.array [N, D].
        """
        if train_mode == "train":
            start_index = index
            end_index = index + history_length
        elif train_mode == "test":
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError("train model {} is not defined".format(train_mode))
        # print("index:%d",index)
        # print("start_index:%d",start_index)
        # print("end_index:%d",end_index)
        data_x = data[:, start_index: end_index]
        data_y = data[:, end_index]

        return data_x, data_y

    @staticmethod
    def pre_process_data(data, norm_dim):
        """
        :param data: np.array, original traffic data without normalization.
        :param norm_dim: int, normalization dimension.
        :return:
            norm_base: list, [max_data, min_data], data of normalization base.
            norm_data: np.array, normalized traffic data.
        """
        norm_base = PEMSBAYDataset.normalize_base(data, norm_dim)  # find the normalize base
        norm_data = PEMSBAYDataset.normalize_data(norm_base[0], norm_base[1], data)  # normalize data

        return norm_base, norm_data

    @staticmethod
    def normalize_base(data, norm_dim):
        """
        :param data: np.array, original traffic data without normalization.
        :param norm_dim: int, normalization dimension.
        :return:
            max_data: np.array
            min_data: np.array
        """
        max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D]
        min_data = np.min(data, norm_dim, keepdims=True)
        return max_data, min_data

    @staticmethod
    def normalize_data(max_data, min_data, data):
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, original traffic data without normalization.
        :return:
            np.array, normalized traffic data.
        """
        mid = min_data
        base = max_data - min_data
        normalized_data = (data - mid) / base

        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data):
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, normalized data.
        :return:
            recovered_data: np.array, recovered data.
        """
        mid = min_data
        base = max_data - min_data

        recovered_data = data * base + mid

        return recovered_data

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)


def pemsbay_get_loader(ds_name="pems-bay"):
    
    if ds_name == 'pems-bay':
        num_nodes = 325
   

    train_data = PEMSBAYDataset(data_path=["../dataset/PEMS/{}/adj_mx_bay.pkl".format(ds_name), "../dataset/PEMS/{}/pems-bay.h5".format(ds_name)], num_nodes=num_nodes,
                             divide_days=[45, 14],
                             time_interval=5, history_length=6,
                             train_mode="train",data_type=ds_name)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    test_data = PEMSBAYDataset(data_path=["../dataset/PEMS/{}/adj_mx_bay.pkl".format(ds_name), "../dataset/PEMS/{}/pems-bay.h5".format(ds_name)], num_nodes=num_nodes,
                            divide_days=[45, 14],
                            time_interval=5, history_length=6,
                            train_mode="test",data_type=ds_name)

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader=pemsbay_get_loader()
    for data in train_loader:
        g = data['graph']
        f = data['flow_x']
        y = data['flow_y']
        
