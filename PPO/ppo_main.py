
import pfrl
import os
import time
from typing import Any, Callable, List, Dict, Optional, Tuple, Union


import asyncio


from players import base_env, OppoPlayer
from utils import (
    Config, GlobalStatistics, Population, DDAPopulation
    )

from battles import (
    env_algorithm_wrapper, 
    multi_launch_battles, 
    launch_battles, 
    wrap_launch_battles
    )



# Set a random seed used in PFRL.
pfrl.utils.set_random_seed(2021)

def train_loop():
    n=1
    out_dir = r""
    config = Config(
        num_env = n,
        steps=int(5e9), 
        lr = 1e-4, 
        gamma=0.99,
        vf_coef=0.5, 
        update_interval=50000,
        minibatch_size=4000, 
        epochs=20, 
        checkpoint_freq=1000,

        # for resuming training   
        #out_dir=out_dir
    )
    
    
    population = Population(config)
    players, opponents = population.create_n_agent(n)
    
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
            launch_battles(players[0], opponents[0])
            )
    
        collect_time = time.perf_counter()-collect_start
        total_collect_time += collect_time
       
        train_start = time.perf_counter()
        do_train = population.update_if_dataset_is_ready()
        train_time = time.perf_counter()-train_start
        
        if do_train:

            # 
            config.lg.debug(f"{num}")
            
            this_step = population.statistics.curr_step-prev_cum_num_step
            total_step = this_step 
            
            time_per_step = total_collect_time/total_step
            
            config.lg.debug(
                f"number of battles and steps| Cumlative:{population.statistics.cum_num_battle}, {population.statistics.curr_step} "
                f", This step:{population.statistics.cum_num_battle-prev_cum_num_battle}, {this_step}"
                )
            config.lg.debug(f"win rate: {population.statistics.cum_win_rate()}")
            
            config.lg.debug(f"total_step:{total_step}, total_time/total_step: {time_per_step}")
            
            config.lg.debug(f"cum time: {time.perf_counter()-start},collect_time: {total_collect_time/num} , total_collect_time: {total_collect_time}")
            config.lg.debug(f"train_time: {train_time}\n")
            
            
            total_collect_time = 0
            
            prev_cum_num_battle = population.statistics.cum_num_battle
            prev_cum_num_step = population.statistics.curr_step
            
            num = 0
        num += 1
    exit()

if __name__=='__main__':
    train_loop()