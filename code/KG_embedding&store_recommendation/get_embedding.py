import argparse
import json
import logging
import os
import torch
import torch.optim
import pandas as pd
import numpy as np
import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models

DATA_PATH = './data'

parser = argparse.ArgumentParser(
    description="Urban Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="NYC_final", choices=["NYC", "CHI"],
    help="Urban Knowledge Graph dataset"
)
parser.add_argument(
    "--model", default="GIE", choices=all_models, help='"TransE", "CP", "MurE", "RotE", "RefE", "AttE",'
                                                        '"ComplEx", "RotatE",'
                                                        '"RotH", "RefH", "AttH"'
                                                        '"GIE"'
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adagrad",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=150, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=3, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--rank", default=32, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--batch_size", default=4096, type=int, help="Batch size"
)
parser.add_argument(
    "--learning_rate", default=1e-3, type=float, help="Learning rate"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--multi_c", action="store_false", help="Multiple curvatures per relation"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"],
    help="Bias type (none for no bias)"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)


def get_embeddings(args):
    # create model
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    model = getattr(models, args.model)(args)
    model.load_state_dict(torch.load(
        os.path.join("")))
    entity_embeddings = model.entity.weight.detach().numpy()

    idx = pd.read_csv(DATA_PATH + '/' + args.dataset + "/entity_idx_embedding.csv", header=None)
    entity_idx = np.array(idx)

    entity_final_embedddings = np.zeros([entity_embeddings.shape[0], entity_embeddings.shape[1]])
    for i in range(entity_embeddings.shape[0]):
        entity_final_embedddings[int(entity_idx[i])] = entity_embeddings[i]

    return entity_final_embedddings


def get_region_embeddings(grid_KG_id_path, entity_final_embedddings, save_path):
    grid = pd.read_csv(grid_KG_id_path)
    grid_KG_id = grid[["region_id", "KG_id"]].values
    grid_embeddings = np.zeros([len(grid_KG_id), 32])

    for i in range(grid_embeddings.shape[0]):
        grid_embeddings[i] = entity_final_embedddings[int(grid_KG_id[i][1])]

    print(grid_embeddings)
    np.save(save_path, grid_embeddings)


def get_POI_embedding(grid_KG_id_path, entity_final_embedddings, save_path):
    poi = pd.read_csv(grid_KG_id_path)
    poi_KG_id = poi[["poi_id", "KG_id", "Region_id"]].values
    poi_embeddings = np.zeros([len(poi_KG_id), 33])
    for i in range(poi_embeddings.shape[0]):
        poi_embeddings[i][0:32] = entity_final_embedddings[int(poi_KG_id[i][1])]
        poi_embeddings[i][32] = int(poi_KG_id[i][2])

    print(poi_embeddings)
    np.save(save_path, poi_embeddings)


def get_Road_embedding_considerGeo(grid_KG_id_path, entity_final_embedddings, geo_file_path, save_path):
    geo = pd.read_csv(geo_file_path)
    valid_road_ids = set(geo['geo_id'].unique())

    road = pd.read_csv(grid_KG_id_path)
    road_KG_id = road[["road_id", "KG_id", "Region_id"]].values

    filtered_embeddings = []
    count = 0

    for i in range(len(road_KG_id)):
        road_id = int(road_KG_id[i][0])
        kg_id = int(road_KG_id[i][1])
        region_id = int(road_KG_id[i][2])

        if road_id in valid_road_ids:
            emb = np.zeros(33, dtype=np.float16)
            emb[0:32] = entity_final_embedddings[kg_id].astype(np.float16)
            emb[32] = np.float16(region_id)
            filtered_embeddings.append(emb)
            count += 1

    road_embeddings = np.array(filtered_embeddings, dtype=np.float16)
    np.save(save_path, road_embeddings)


def get_Iso_embedding_considerGeo(grid_KG_id_path, entity_final_embedddings, geo_file_path, save_path):
    geo = pd.read_csv(geo_file_path)
    valid_iso_ids = set(geo['geo_id'].unique())

    iso = pd.read_csv(grid_KG_id_path)
    iso_KG_id = iso[["iso_id", "KG_id", "Region_id"]].values

    filtered_embeddings = []
    count = 0

    for i in range(len(iso_KG_id)):
        iso_id = int(iso_KG_id[i][0])
        kg_id = int(iso_KG_id[i][1])
        region_id = int(iso_KG_id[i][2])

        if iso_id in valid_iso_ids:
            emb = np.zeros(33, dtype=np.float16)
            emb[0:32] = entity_final_embedddings[kg_id].astype(np.float16)
            emb[32] = np.float16(region_id)
            filtered_embeddings.append(emb)
            count += 1

    iso_embeddings = np.array(filtered_embeddings, dtype=np.float16)
    np.save(save_path, iso_embeddings)


# def get_Area_embedding(grid_KG_id_path, entity_final_embedddings, save_path):
#     area = pd.read_csv(grid_KG_id_path)
#     area_KG_id = area[["area_id", "KG_id", "Region_id"]].values
#     area_embeddings = np.zeros([len(area_KG_id), 33])
#     for i in range(area_embeddings.shape[0]):
#         area_embeddings[i][0:32] = entity_final_embedddings[int(area_KG_id[i][1])]
#         area_embeddings[i][32] = int(area_KG_id[i][2])
#         print(int(area_KG_id[i][2]))
#     # exit(0)
#     print(area_embeddings)
#     np.save(save_path, area_embeddings)


def get_Area_embedding(grid_KG_id_path, entity_final_embedddings, save_path):
    area = pd.read_csv(grid_KG_id_path)

    area = area[~area["Region_id"].isin([1, 103, 104])]

    area_KG_id = area[["area_id", "KG_id", "Region_id"]].values
    area_embeddings = np.zeros([len(area_KG_id), 33])

    for i in range(area_embeddings.shape[0]):
        area_embeddings[i][0:32] = entity_final_embedddings[int(area_KG_id[i][1])]
        area_embeddings[i][32] = int(area_KG_id[i][2])

    np.save(save_path, area_embeddings)

# entity_final_embedddings, pca_final_embeddings, tsne_dim2_embeddings = get_embeddings(parser.parse_args())
# entity_final_embedddings = get_embeddings(parser.parse_args())
# np.save('./data/NYC/all_entity_embeddings.npy', entity_final_embedddings)


# get_Area_embedding("./data/NYC/Area2KG_NYC.csv", entity_final_embedddings,
#                    './data/NYC/bike_Area_embedding.npy')


