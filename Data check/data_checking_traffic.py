import os
import pandas as pd
import numpy as np
from featurewiz import featurewiz
from matplotlib import pyplot as plt
import json



folder = os.path.join("/", "RanD", "CREMEv2_Result", "20230207", "logs", "label_traffic")
if os.path.exists(folder):
    print("Path is exist!!!")
    filename_label = 'extract_traffic_mirai_nfstream.csv'
    label_technique = 'labels_technique.json'
    label_lifecycle = 'labels_lifecycle.json'
else:
    print("Path is not exist!!!")
    
df = pd.read_csv(os.path.join(folder, "nfstream", filename_label))
df.info()
df.describe()
    
    