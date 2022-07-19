
import collections
import logging
import pfrl
from pfrl.experiments.evaluator import Evaluator, save_agent
import asyncio
from asyncio.events import get_event_loop
import concurrent.futures

from typing import Any, Callable, List, Dict, Optional, Tuple, Union
import time
from poke_env.player.env_player import EnvPlayer
from poke_env.utils import to_id_str
import numpy as np
from players import base_env

def timeit(func):
    async def process(func, *args, **params):
        if asyncio.iscoroutinefunction(func):
            print('this function is a coroutine: {}'.format(func.__name__))
            return await func(*args, **params)
        else:
            print('this is not a coroutine')
            return func(*args, **params)

    async def helper(*args, **params):
        print('{}.time'.format(func.__name__))
        start = time.time()
        result = await process(func, *args, **params)

        # Test normal function route...
        # result = await process(lambda *a, **p: print(*a, **p), *args, **params)

        print('>>>', time.time() - start)
        return result

    return helper

def train_agent(
    env:base_env,
    agent: pfrl.agents.PPO,
    steps,
    outdir,
    max_episode_len=None,
    step_offset=0,
    evaluator=None,
    successful_score=None,
    step_hooks=(),
    eval_during_episode=False,
    logger=None,
):

    logger = logger or logging.getLogger(__name__)

    episode_r = 0
    episode_idx = 0
    
    # o_0, r_0
    obs = env.reset()

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    turn_freq = collections.defaultdict(int)
    episode_len = 0
    try:
        for t in range(1, steps):
            
            # a_t
            action = agent.act(obs)
    
            # o_{t+1}, r_{t+1}
            obs, r, done, info = env.step(action)
            episode_r += r
            episode_len += 1
            reset = episode_len == max_episode_len or info.get("needs_reset", False)
            

            is_win = env._current_battle.won
            agent.observe(
                obs, r, done, reset, is_win
                )
            episode_end = done or reset or t == steps
            turn_freq[env._current_battle.turn] += 1

            if episode_end:
                
                episode_idx += 1
                
                if t == steps:
                    break
                
                # Start a new episode
                episode_r = 0
                episode_len = 0
                turn_freq = collections.defaultdict(int)
                obs = env.reset()
            
    except (Exception, KeyboardInterrupt):
        
        raise

    

@timeit
async def multi_launch_battles(players, opponents):
    
    battles_coroutines = []
    for i in range(len(players)):
        battles_coroutines.append(
            asyncio.gather(
            players[i].send_challenges(
                opponent=to_id_str(opponents[i].username),
                n_challenges=1,
                to_wait=opponents[i].logged_in,
            ),
            opponents[i].accept_challenges(
                opponent=to_id_str(players[i].username), 
                n_challenges=1
                )
            )
            
        )
    
    await asyncio.gather(*battles_coroutines)


#@timeit   
async def multi_launch_battles(players, opponents):
    battles_coroutines = []
    for i in range(len(players)):
        battles_coroutines.append(
                asyncio.gather(
                players[i].send_challenges(
                    opponent=to_id_str(opponents[i].username),
                    n_challenges=1,
                    to_wait=opponents[i].logged_in,
                ),
                opponents[i].accept_challenges(
                    opponent=to_id_str(players[i].username), 
                    n_challenges=1
                    )
                )
            )
    await asyncio.gather(*battles_coroutines)


    
async def launch_battles(player, opponent):
    
    battles_coroutine = asyncio.gather(
        player.send_challenges(
            opponent=to_id_str(opponent.username),
            n_challenges=1,
            to_wait=opponent.logged_in,
        ),
        opponent.accept_challenges(
            opponent=to_id_str(player.username), 
            n_challenges=1
            ),
        
    )
    
    await battles_coroutine
    

      

def wrap_launch_battles(func: Callable, *args: Any):
    
    loop = get_event_loop()
    result = loop.run_until_complete(func(*args))
    loop.close()
    return result

def env_algorithm_wrapper(player, number,kwargs):

    train_agent(player, **kwargs)
    
    player._start_new_battle = False
    while True:
        try:
            player.complete_current_battle()
            player.reset()
        except OSError:
            break
