# traffic_predcition_GCN
The repository implements a traffic prediction algorithm, and it includes an improved GCN (Graph Convolutional Network) algorithm.

## Dependencies

- matplotlib==3.7.3
- numpy==1.22.4
- pandas==1.3.5
- torch==1.10.0+cu111


## Datasets

PEMS03、PEMS04、PEMS07、PEMS08、PEMS-BAY、METR-LA
```
├── dataset
│   └── PEMS
│       ├── PEMS03
│       │   ├── PEMS03.csv
│       │   ├── PEMS03_data.csv
│       │   ├── PEMS03.npz
│       │   └── PEMS03.txt
│       ├── PEMS04
│       │   ├── PEMS04.csv
│       │   └── PEMS04.npz
│       ├── PEMS07
│       │   ├── PEMS07.csv
│       │   └── PEMS07.npz
│       └── PEMS08
│           ├── PEMS08.csv
│           └── PEMS08.npz

```

## Start training
```shell
cd src
python traffic_prediction.py
```
## Plot traffic flow graph
```shell
cd tools
python vis.py
```