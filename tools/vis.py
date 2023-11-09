import pandas as pd
import pickle
import numpy as np
import csv
from folium.plugins import HeatMap
import folium 
import os

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

def get_metr_la_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type="connect",data_type=None) -> np.array:
    """
    construct adjacent matrix by csv file
    :param distance_file: path of csv file to save the distances between nodes
    :param num_nodes: number of nodes in the graph
    :param id_file: path of txt file to save the order of the nodes
    :param graph_type: ["connect", "distance"] if use weight, please set distance
    :return:
    """
    A = np.zeros([int(num_nodes), int(num_nodes)])

    if id_file:
        with open(id_file, "r") as f_id:
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(f_id.read().strip().split("\n"))}

            with open(distance_file, "r") as f_d:
                f_d.readline()
                reader = csv.reader(f_d)
                for item in reader:
                    if len(item) != 3:
                        continue
                    i, j, distance = int(item[0]), int(item[1]), float(item[2])
                    if graph_type == "connect":
                        A[node_id_dict[i], node_id_dict[j]] = 1.
                        A[node_id_dict[j], node_id_dict[i]] = 1.
                    elif graph_type == "distance":
                        A[node_id_dict[i], node_id_dict[j]] = 1. / distance
                        A[node_id_dict[j], node_id_dict[i]] = 1. / distance
                    else:
                        raise ValueError("graph type is not correct (connect or distance)")
        return A

    if data_type=='metr-la':
        ids_path = os.path.join('./assets/metr-la/','sensor_ids_la.txt')
        with open(ids_path) as f:
            ids = f.read().strip().split(',')
        sensor_to_idx = {int(sensor_id): i for i, sensor_id in enumerate(ids)}
            
        with open(distance_file, "r") as f_d:
            f_d.readline()
            reader = csv.reader(f_d)
            for item in reader:
                if len(item) != 3:
                    continue
                i, j, distance = int(item[0]), int(item[1]), float(item[2])
                if i not in sensor_to_idx or j not in sensor_to_idx:
                    continue
                # print(i,j,distance)
                i = sensor_to_idx[i]
                j = sensor_to_idx[j]
                if graph_type == "connect":
                    A[i, j], A[j, i] = 1., 1.
                elif graph_type == "distance":
                    A[i, j] = 1. / distance
                    A[j, i] = 1. / distance
                else:
                    raise ValueError("graph type is not correct (connect or distance)")
    
    else:
        with open(distance_file, "r") as f_d:
            f_d.readline()
            reader = csv.reader(f_d)
            for item in reader:
                if len(item) != 3:
                    continue
                i, j, distance = int(item[0]), int(item[1]), float(item[2])

                if graph_type == "connect":
                    A[i, j], A[j, i] = 1., 1.
                elif graph_type == "distance":
                    A[i, j] = 1. / distance
                    A[j, i] = 1. / distance
                else:
                    raise ValueError("graph type is not correct (connect or distance)")

    return A



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

def draw_metr_la():
    # adj_mx = get_adjacent_matrix('/home/drx/workshop/traffic_predict/tp_code/dataset/PEMS/pems-bay/adj_mx_bay.pkl',num_nodes=325)
    adj_mx = get_metr_la_adjacent_matrix('./assets/metr-la/distances_la.csv',num_nodes=207,data_type='metr-la')
    adj_mx_bi = []
    for i in range(adj_mx.shape[0]):
        for j in range(adj_mx.shape[0]):
            # print(adj_mx[i][j])
            if i==j:
                continue
            if adj_mx[i][j]!=0:
                adj_mx_bi.append((i,j))


    data_single_car=[]
    with open("./assets/graph_sensor_locations.csv", "r") as f_d:
        f_d.readline()
        reader = csv.reader(f_d)
  
        for index, item in enumerate(reader):
            x, y = float(item[2]), float(item[3])
            # print(x,y)
            data_single_car.append([x, y,0])
    # print(len(data_single_car))

    map_h = folium.Map(location=(data_single_car[0][0],data_single_car[0][1]),zoom_start= 12,tiles="CartoDB positron")
    data =get_flow_data("./assets/metr-la/metr_la.h5")
    data = data.transpose([1, 0, 2]) 
    clip_data = data[0,:,:].flatten()

    for i,ele in enumerate(clip_data):
        data_single_car[i][2]=float(ele)

    heatmap = HeatMap(data_single_car)
    heatmap.add_to(map_h)
    # Save the map:
    hm_name="metr-la.html"
    map_h.save(hm_name)

def draw_pems_bay():
    adj_mx = get_adjacent_matrix('./assets/pems-bay/adj_mx_bay.pkl',num_nodes=325)
    adj_mx_bi = []
    for i in range(adj_mx.shape[0]):
        for j in range(adj_mx.shape[0]):
            # print(adj_mx[i][j])
            if i==j:
                continue
            if adj_mx[i][j]!=0:
                adj_mx_bi.append((i,j))


    data_single_car=[]
    with open("./assets/graph_sensor_locations_bay.csv", "r") as f_d:
        # f_d.readline()
        reader = csv.reader(f_d)
  
        for item in reader:
            x, y = float(item[1]), float(item[2])
            data_single_car.append([x, y,0])


    # print(len(data_single_car))

    map_h = folium.Map(location=(data_single_car[0][0],data_single_car[0][1]),zoom_start= 12,tiles="CartoDB positron")
    data =get_flow_data("./assets/pems-bay/pems-bay.h5")
    data = data.transpose([1, 0, 2]) 
    clip_data = data[0,:,:].flatten()

    for i,ele in enumerate(clip_data):
        data_single_car[i][2]=float(ele)

    heatmap = HeatMap(data_single_car)
    heatmap.add_to(map_h)
    # Save the map:
    hm_name="pems-bay.html"
    map_h.save(hm_name)


if __name__ == "__main__":
    # draw graph metr_la
    draw_metr_la()
    print("Plot metr_la graph finished")
    # draw graph pems-bay
    draw_pems_bay()
    print("Plot pems_bay graph finished")




