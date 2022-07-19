
import pfrl
import os

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch
import concurrent
import asyncio
from functools import partial

import numpy as np
from multiprocessing import Pool
from tabulate import tabulate

from poke_env.player_configuration import PlayerConfiguration
from poke_env.utils import to_id_str
from poke_env.player.env_player import (
    Gen8EnvSinglePlayer,
)
import time
from typing import Any, Callable, List, Dict, Optional, Tuple, Union
import argparse
from poke_env.player.random_player import RandomPlayer

from players import base_env, OppoPlayer
from utils import (
    Parse, Config, GlobalStatistics, Population, DDAPopulation
    
    )

from battles import (
    env_algorithm_wrapper, 
    multi_launch_battles, launch_battles, wrap_launch_battles
    )



# Set a random seed used in PFRL.
pfrl.utils.set_random_seed(2021)

if __name__=='__main__':

    n=30
    #out_dir = r"C:\Users\taimo\Desktop\ppo_and_rnd_multi_env_run\PPO_DDA_results\PPO_DDA_2022-1-27_17-22-36-301183"
    out_dir = r"PPO_DDA_results\PPO_DDA_2022-1-30_11-58-24-781235"
    config = Config(


        num_env = n,
        steps=int(5e9), 
        lr = 5e-5, 
        gamma=0.999,
        vf_coef=0.5,
        update_interval=35000,
        minibatch_size=2000, 
        epochs=20, 
        checkpoint_freq=1000,
        is_dda=True,
        oppo_model_path=r"PPO_results\PPO_2022-1-25_18-11-12-703036/trained_models", 
        # for resuming training   
        out_dir=out_dir
    )
    
    dda_population = DDAPopulation(config)
    dda_players, dda_opponents = dda_population.create_n_agent(n)
    

    
        
    # create loop 
    loop = asyncio.get_event_loop()    
    total_collect_time = 0
    prev_dda_cum_num_battle = 0
    prev_cum_num_battle = 0
    prev_dda_cum_num_step = 0
    prev_cum_num_step = 0
    start = time.perf_counter()   
    num = 1
    for i in range(100000000000000):



        collect_start = time.perf_counter()    
        
        loop.run_until_complete(
            multi_launch_battles(dda_players, dda_opponents)
            )
      
        collect_time = time.perf_counter()-collect_start
        total_collect_time += collect_time
       
        train_start = time.perf_counter()
        dda_do_train= dda_population.update_if_dataset_is_ready()
        train_time = time.perf_counter()-train_start
        
        if dda_do_train:

            
            config.lg.debug(f"{num}")
            this_dda_step = dda_population.statistics.curr_step-prev_dda_cum_num_step
            total_step = this_dda_step
            
            time_per_step = total_collect_time/total_step
            
            config.lg.debug(
                f"number of battles and steps(dda)| Cumlative:{dda_population.statistics.cum_num_battle}, {dda_population.statistics.curr_step} "
                f", This step:{dda_population.statistics.cum_num_battle-prev_dda_cum_num_battle}, {this_dda_step}"
                )
            config.lg.debug(f"win rate(dda): {dda_population.statistics.cum_win_rate()}")
         
            
            config.lg.debug(f"total_step:{total_step}, total_time/total_step: {time_per_step}")
            #config.lg.debug(f"len of local buffer: {len(population.learner.memory)}, {len(dda_population.learner.memory)}, {len(population.get_agent(0).memory)}")
            config.lg.debug(f"cum time: {time.perf_counter()-start},collect_time: {total_collect_time/num} , total_collect_time: {total_collect_time}")
            config.lg.debug(f"train_time: {train_time}\n")
            
            
            total_collect_time = 0
            prev_dda_cum_num_battle = dda_population.statistics.cum_num_battle
            prev_dda_cum_num_step = dda_population.statistics.curr_step
           
            dda_population.reset_battles()
            num = 0
        num += 1
    exit()