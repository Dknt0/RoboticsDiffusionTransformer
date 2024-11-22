import sys
sys.path.append("../")

import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch

from configs.state_vec import STATE_VEC_IDX_MAPPING

embeddings_path = "/home/dknt/Project/rdt/RoboticsDiffusionTransformer/out/earn_money.pt"

embeddings = torch.load(embeddings_path)
print(embeddings['instruction'])
