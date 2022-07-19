
from dataclasses import dataclass
from pandas.core import base

import heapq
import collections
import numpy as np
import json
from math import exp
from typing import *

from poke_env.environment.pokemon import Pokemon
from poke_env.environment.battle import Battle

import pandas as pd
import math
import os 
import glob
import ndjson
import random

class AgentBuffer:
    def __init__(self, path) -> None:
        
        self._models_path_dict = {}
        self._keys = {}
        self.make_models_list(path)
        self.called_times = 0
        self.sampled_max_index = 10
    
    def make_models_list(self, recorded_path: str) -> None:
        if os.path.exists(os.path.join(recorded_path)):
            for f in os.listdir(os.path.join(recorded_path)):
                
                self._models_path_dict[
                    int(f.split("_")[0])
                ] = os.path.join(recorded_path, f)
                
            self._keys = sorted(self._models_path_dict)
    
    def add_models_path(self, path):
        
        
        self._models_path_dict[int(os.path.basename(path).split("_")[0])] = path
        self._keys = sorted(self._models_path_dict)

    def get_latest_model(self):
        
        return self._models_path_dict[self._keys[-1]]

    def __len__(self):
        return len(self._models_path_dict)
    
    def random_sample(self):
        
        return random.choice(list(self._models_path_dict.values()))

    

     