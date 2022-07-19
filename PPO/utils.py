
from logging import NOTSET
from posixpath import dirname
from re import I
import re
import time
import os
import asyncio
import argparse
import logging
import datetime
import random
from tkinter import N
from tkinter.messagebox import NO
import numpy as np
import multiprocessing
import ndjson
from pfrl import agent, agents
import joblib
from torch._C import device

from players import base_env, OppoPlayer, dda_env

from torch_win_rate import PokeBattle, BattleBuffer, OppoInfoBuffer, train_win_predictor, LSTM_Predictor
from asyncio import Queue
import torch
import pfrl
from pfrl.nn import EmpiricalNormalization
import json
import copy
from threading import Thread
import glob
from torch.utils.tensorboard import SummaryWriter

from battles import wrap_launch_battles, env_algorithm_wrapper
from explorer import RNDModel
from rnd_utils import RunningMeanStd
from agent import PPO, DDA_PPO
from memories import LocalMemory
from model import MainModel
from features import PokeDataBase
from agents_pool import AgentBuffer


def save_agent(agent, t, outdir, logger, suffix=""):
    dirname = os.path.join(outdir, "{}{}".format(t, suffix))
    agent.save(dirname)
    logger.info("Saved the agent to %s", dirname)
    model_path = os.path.join(dirname, "model.pt")
    return dirname



    
    

class GlobalStatistics:
    
    def __init__(self, start_step = 0, start_num_battle = 0, cum_num_win_battle = 0) -> None:
        
        self.cum_num_battle = start_num_battle
        self.curr_step = start_step
        self.cum_num_win_battle = cum_num_win_battle
                
    
    
    def cum_win_rate(self):
        
        return 0 if self.cum_num_battle == 0 else self.cum_num_win_battle/self.cum_num_battle
    

def get_logger(
    logger_name, log_file,
    s_fmt='%(message)s', f_fmt='%(message)s'):
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter(s_fmt))

    logger.addHandler(stream_handler)


    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(f_fmt))

    logger.addHandler(file_handler)

    return logger
    
class Config:
    
    def __init__(self,
                num_env,
                lr, 
                update_interval, 
                gamma,
                vf_coef, 
                minibatch_size, 
                epochs, 
                steps, 
                checkpoint_freq,
                is_dda=False,
                oppo_model_path=None,
                out_dir=None,
                ) -> None:
        # observation info
        self.poke_database = PokeDataBase()
        features_dim_dict = self.poke_database.get_features_dims()
        
        self.total_features_dim = features_dim_dict["total_features_dim"]
        
        self.battle_obs_dim = features_dim_dict["battle_obs_dim"]
        self.oppo_obs_dim = features_dim_dict["oppo_obs_dim"]
        self.player_obs_dim = features_dim_dict["player_obs_dim"]
        self.total_features_dim = features_dim_dict["total_features_dim"]
            
        self.poke_obs_dim = features_dim_dict["poke_obs_dim"]
        self.moves_obs_dim  = features_dim_dict["moves_obs_dim"]
        self.ability_obs_dim = features_dim_dict["ability_obs_dim"]
        self.item_obs_dim = features_dim_dict["item_obs_dim"]
        self.others_obs_dim = features_dim_dict["others_obs_dim"]
        self.total_poke_dim = features_dim_dict["total_poke_dim"]
        
        
        # agent model(PPO)
        self.num_env = num_env
        self.max_recurrent_sequence_len = None
        self.lr = lr
        self.update_interval =update_interval
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.steps = steps
        self.checkpoint_freq = checkpoint_freq

        # dda
        self.use_win_predictor = True
        self.prediction_steps = 60
        self.win_predictor_lr =5e-5
        self.use_oracle = True
        
        # define path for saving
        self.oppo_model_path = oppo_model_path
        
        # make log dir
        if out_dir is None: # start to train models from scratch
            dt_now = datetime.datetime.now()
            cwd = os.path.dirname(__file__)
            if is_dda == False:
                dir_name = f"PPO_{dt_now.year}-{dt_now.month}-{dt_now.day}_{dt_now.hour}-{dt_now.minute}-{dt_now.second}-{dt_now.microsecond}"
                self.out_dir = os.path.join(cwd, f"PPO_results/{dir_name}")
            else:
                dir_name = f"PPO_DDA_{dt_now.year}-{dt_now.month}-{dt_now.day}_{dt_now.hour}-{dt_now.minute}-{dt_now.second}-{dt_now.microsecond}"
                self.out_dir = os.path.join(cwd, f"PPO_DDA_results/{dir_name}")
            os.makedirs(self.out_dir, exist_ok=True)
            self.start_step=0
            self.start_num_battle=0
            self.n_updates = 0
            self.win_predictor_path = None
            self.trained_model = None
            self.dda_trained_model = None
            self.rnd_path = None
            self.rnd_normalizer_path = None
            self.reward_normalizer_path = None
            self.env_steps = None
            self.env_oracle_param = None
            self.cum_num_win_battle = 0
            self.entropy_coef = 0.001
            self.int_rewems = None

            self.model_path = os.path.join(self.out_dir, "trained_models")
            #self.agentbuffer = AgentBuffer(self.model_path)

            # dda
            self.dda_n_updates = 0

        
        else: # resume training
            print("resume training")
            with open(
                os.path.join(out_dir, "train_record.ndjson")
                ) as f:
                data = ndjson.load(f)
            
            latest_data = data[-1]

            self.out_dir = out_dir
            if is_dda:
                self.model_path = os.path.join(self.out_dir, "dda_trained_models")
            else:
                self.model_path = os.path.join(self.out_dir, "trained_models")
            self.agentbuffer = AgentBuffer(self.model_path)
            

            self.lr = latest_data["curr_lr"]
            self.start_step = latest_data["curr_step"]
            self.start_num_battle = latest_data["curr_num_battle"]
            self.n_updates = latest_data["n_updates"]
            self.cum_num_win_battle = latest_data["curr_cum_num_win_battle"]
            self.entropy_coef = latest_data["entropy_coef"]
            
            
            self.trained_model = self.agentbuffer.get_latest_model()
            print(self.trained_model)
            #self.dda_trained_model = dda_trained_model
            self.rnd_path = os.path.join(self.out_dir, "rnd_model.pt")
            self.rnd_normalizer_path = os.path.join(self.out_dir, "rnd_normalizer.jb")
            self.reward_normalizer_path = os.path.join(
            self.out_dir, "reward_normalizer.jb")
            self.int_rewems = latest_data["int_rewems"]
            
            self.env_steps = {}
            self.env_oracle_param = {}
            for id in range(self.num_env):
                with open(
                    os.path.join(out_dir, f"env{id}_record.ndjson")
                ) as f:
                    env_data = ndjson.load(f)
                self.env_steps[id] = env_data[-1]["curr_step"]
                self.env_oracle_param[id] = env_data[-1]["oracle_param"]
            
            # if trainig the models toward dda
            if is_dda:
                with open(
                os.path.join(out_dir, "train_record.ndjson")
                ) as f:
                    dda_data = ndjson.load(f)
                self.win_predictor_path = os.path.join(self.out_dir, "win_predictor.pt")
                dda_latest_data = dda_data[-1]
                self.win_predictor_lr = dda_latest_data["curr_lr"]
                self.dda_n_updates = dda_latest_data["dda_n_updates"]

            # printing for debug
            print(self.out_dir)
            print(self.lr)
            print(self.start_step, self.cum_num_win_battle, self.start_num_battle)

            
        # log
        self.lg = get_logger(__name__, os.path.join(self.out_dir,"log.txt"))

        # replay
        self.replay_outdir = os.path.join(
                self.out_dir, f"replays")
        os.makedirs(self.replay_outdir, exist_ok=True)
        
        # dumping config to json
        self.dump()
        


    def return_obses_dim(self):
        return[
            self.total_features_dim, 
            self.battle_obs_dim,
            self.player_obs_dim,
            self.oppo_obs_dim,
            self.moves_obs_dim,
            self.total_poke_dim,
            self.others_obs_dim
        ] 
    def set_seed(self, seed):
        # PFRL depends on random
        random.seed(seed)
        # PFRL depends on numpy.random
        np.random.seed(seed)
        # torch.manual_seed is enough for the CPU and GPU
        torch.manual_seed(seed)

        os.environ["PYTHONHASHSEED"] = str(seed)
        
    def dump(self):
        reserved_config = {k : v for k, v in self.__dict__.items() if k not in ["hook_list", "lg", "poke_database", "agentbuffer"]}
        with open(os.path.join(self.out_dir,"config.json"), "w") as f:
            json.dump(reserved_config, f, indent=2)
        return
    

class Learner:
    
    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.is_cuda = torch.cuda.is_available() 
        self.rnd_normalizer_outdir = os.path.join(
            config.out_dir, "rnd_normalizer.jb")
        self.rnd_outdir = os.path.join(
            config.out_dir, "rnd_model.pt")
        self.reward_normalizer_outdir = os.path.join(
            config.out_dir, "reward_normalizer.jb")
        self.statistics = GlobalStatistics(
            start_step=self.config.start_step,
            start_num_battle=self.config.start_num_battle,
            cum_num_win_battle=self.config.cum_num_win_battle
            )
        
        self.memory = LocalMemory()
        self.obs_nomalizer = None
        self.unique_index = format(os.getpid(), f"08d")
        self.rnd_model=RNDModel(
            *self.config.return_obses_dim()
            )
        self.model = MainModel(
            *self.config.return_obses_dim()
        )
        self.learner = self.create_learner()
        print("number of params", self.num_model_params())
        
        
        self.logger = logging.getLogger(__name__)
        
        self.outdir = os.path.join(config.out_dir,
                                    "trained_models")
        
        
        self.writer = SummaryWriter(
            os.path.join(config.out_dir,"tensorboard")
            )
        
    def create_learner(self):
        
        optim = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        learner = PPO(
            gpu = 0 if self.is_cuda else None,
            obs_normalizer=self.obs_nomalizer,
            rnd_model=self.rnd_model,
            rnd_normalizer=RunningMeanStd(),
            #reward_normalizer=RunningMeanStd(),
            max_recurrent_sequence_len=self.config.max_recurrent_sequence_len,
            recurrent=True,
            model=self.model,
            optimizer=optim,
            value_func_coef=self.config.vf_coef,
            gamma=self.config.gamma,
            update_interval=self.config.update_interval,
            minibatch_size=self.config.minibatch_size,
            epochs=self.config.epochs,
            memory=self.memory,
            entropy_coef=self.config.entropy_coef
        )
        learner.save_rnd_normalizer(self.rnd_normalizer_outdir)
        learner.load_rnd_normalizer(self.rnd_normalizer_outdir)
        
        learner.int_reward_fileter.rewems = self.config.int_rewems

        # loading the model in advance
        if self.config.trained_model:
            learner.load(self.config.trained_model)
        
        if self.config.rnd_normalizer_path:
            learner.load_rnd_normalizer(self.config.rnd_normalizer_path)
        
        if self.config.reward_normalizer_path and learner.reward_normalizer is not None:
            learner.load_reward_normalizer(self.config.reward_normalizer_path)
        
        if self.config.rnd_path:
            learner.load_rnd(self.config.rnd_path)
        
        
        if self.config.n_updates is not None:
            learner.n_updates = self.config.n_updates

            
        return learner
    
    def save(self):
       
        dirname = save_agent(
                    self.learner, 
                    self.statistics.curr_step, 
                    self.outdir, 
                    self.logger, suffix="_checkpoint")
        
        return dirname
    
    
    def update_if_dataset_is_ready(self)->bool:
        
        dirname = None
        do_train = self.learner._update_if_dataset_is_ready()
        
        
        if do_train:
            self.learner.save_rnd(self.rnd_outdir)
            self.learner.save_rnd_normalizer(self.rnd_normalizer_outdir)
            if self.learner.reward_normalizer is not None:
                self.learner.save_reward_normalizer(self.reward_normalizer_outdir)
            dirname = self.save()
            
            self.writer_hook( self.learner)
            
        return do_train, dirname
    
    
    def num_model_params(self):
        print(self.learner.model)
        params = 0
        for p in self.learner.model.parameters():
            if p.requires_grad:
                params += p.numel()
        return params

    
    def writer_hook(self, agent: PPO):      
        step = self.statistics.curr_step
        stats = dict(agent.get_statistics())
        
        self.writer.add_scalar(
            'learner/total_loss', 
            stats['average_value_loss'] + stats['average_policy_loss'], 
            step
        )
        self.writer.add_scalar(
            'learner/policy_loss', 
            stats['average_policy_loss'], step)

        self.writer.add_scalar(
            'learner/approx_kl_div', 
            stats['average_approx_kl_div'], step)

        self.writer.add_scalar(
            'learner/value_loss', 
            stats['average_value_loss'], step)

        self.writer.add_scalar(
            'learner/int_value_loss', 
            stats['average_int_value_loss'], step)
        self.writer.add_scalar(
            'learner/rnd_loss', 
            stats['average_rnd_loss'], step)

        self.writer.add_scalar(
            'learner/rnd_mean', 
            stats['average_rnd_mean'], step)
        self.writer.add_scalar(
            'learner/rnd_var', 
            stats['average_rnd_var'], step)
        self.writer.add_scalar(
            "learner/int_explained_variance", 
            stats["int_explained_variance"], step)
        self.writer.add_scalar(
            "learner/int_rewems", 
            stats["int_rewems"], step)
        self.writer.add_scalar(
            "learner/n_updates", 
            stats["n_updates"], step)
        self.writer.add_scalar(
            "learner/explained_variance", 
            stats["explained_variance"], step)
        self.writer.add_scalar(
            "learner/entropy_coef", 
            stats["entropy_coef"], step)
        self.writer.add_scalar(
                f"learner/curr_lr",
                stats["curr_lr"], step
            )
        self.writer.add_scalar(
                f"learner/win_rate",
                self.statistics.cum_win_rate(), step
            )
        stats["curr_step"] = self.statistics.curr_step
        stats["curr_num_battle"] = self.statistics.cum_num_battle
        stats["curr_cum_num_win_battle"] = self.statistics.cum_num_win_battle
        with open(os.path.join(self.config.out_dir, "train_record.ndjson"), "a") as f:
            writer = ndjson.writer(f)
            writer.writerow(stats)

class DDALearner:
    
    def __init__(self, config: Config) -> None:
        
        self.is_cuda = torch.cuda.is_available()
        self.rnd_normalizer_outdir = os.path.join(config.out_dir, "rnd_normalizer.jb")
        self.reward_normalizer_outdir = os.path.join(
            config.out_dir, "reward_normalizer.jb")
        
        self.memory = LocalMemory()
        self.config: Config = config
        self.unique_index = format(os.getpid(), f"08d")
        self.obs_nomalizer = None
        # recording game info
        self.model = MainModel(
            *self.config.return_obses_dim()
        )
        self.win_predictor = LSTM_Predictor(
            *self.config.return_obses_dim()
        )
        self.rnd_model = RNDModel(
            *self.config.return_obses_dim()
        )
        self.win_predictor_optim = torch.optim.Adam(
            params=self.win_predictor.parameters(), 
            lr=self.config.win_predictor_lr, 
            weight_decay=1e-4)
        self.win_predictor_train_func = train_win_predictor
        self.statistics = GlobalStatistics(
            start_step=self.config.start_step,
            start_num_battle=self.config.start_num_battle,
            cum_num_win_battle=self.config.cum_num_win_battle
            )
        
        
        self.learner = self.create_learner()
        print("number of params", self.num_model_params())
        
        
        
        
        self.curr_step = 0
        
        self.logger = logging.getLogger(__name__)
        self.outdir = os.path.join(
            config.out_dir,
            f"dda_trained_models"
            )
        
        self.writer = SummaryWriter(
            os.path.join(config.out_dir,
                        "dda_tensorboard")
            )
        self.rnd_outdir = os.path.join(config.out_dir, "rnd_model.pt")
        self.win_predictor_outdir = os.path.join(config.out_dir, "win_predictor.pt")
        
        
   

    # initialization of model and agent
    def create_learner(self):
      
        optim = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        
        
        learner = DDA_PPO(
            gpu = 0 if self.is_cuda else None,
            obs_normalizer=self.obs_nomalizer,
            rnd_model=self.rnd_model,
            rnd_normalizer=RunningMeanStd(),
            #reward_normalizer=RunningMeanStd(),
            max_recurrent_sequence_len=self.config.max_recurrent_sequence_len,
            recurrent=True,
            model=self.model,
            optimizer=optim,
            value_func_coef=self.config.vf_coef,
            gamma=self.config.gamma,
            update_interval=self.config.update_interval,
            minibatch_size=self.config.minibatch_size,
            epochs=self.config.epochs,
            memory=self.memory,
            win_predictor=self.win_predictor,
            win_predictor_optim=self.win_predictor_optim,
            entropy_coef=self.config.entropy_coef
        )
        # if trained model == True, loading the model in advance
        learner.int_reward_fileter.rewems = self.config.int_rewems
        if self.config.trained_model:
            learner.load(self.config.trained_model)

        if self.config.rnd_normalizer_path:
            learner.load_rnd_normalizer(self.config.rnd_normalizer_path)
        
        if self.config.rnd_path:
            learner.load_rnd(self.config.rnd_path)
            
        if self.config.win_predictor_path:
            learner.load_win_predictor(self.config.win_predictor_path)
        
        if self.config.reward_normalizer_path and learner.reward_normalizer is not None:
            learner.load_reward_normalizer(self.config.reward_normalizer_path)
        
        
        if self.config.n_updates is not None:
            learner.n_updates = self.config.n_updates
        
        if self.config.dda_n_updates is not None:
            learner.dda_n_updates = self.config.dda_n_updates


        return learner
    
    def update_if_dataset_is_ready(self):
        
        dirname = None
        do_train = self.learner._update_if_dataset_is_ready()
        
           
        if do_train:
            self.writer_hook(self.learner)
            self.learner.save_rnd(self.rnd_outdir)
            if self.learner.reward_normalizer is not None:
                self.learner.save_reward_normalizer(self.reward_normalizer_outdir)
            self.learner.save_rnd_normalizer(self.rnd_normalizer_outdir)
            self.learner.save_win_predictor(self.win_predictor_outdir)
            dirname = self.save()
            
        return do_train, dirname
        
    def save(self):
        
        dirname = save_agent(
                    self.learner, 
                    self.statistics.curr_step, 
                    self.outdir, 
                    self.logger, suffix="_checkpoint"
                )
        return dirname
    
    def num_model_params(self):
        params = 0
        for p in self.model.parameters():
            if p.requires_grad:
                params += p.numel()
        return params
    
    def writer_hook(self, agent: DDA_PPO):      
        step = self.statistics.curr_step
        stats = dict(agent.get_statistics())
        dda_stats = dict(agent.get_dda_statistics())
        self.writer.add_scalar(
            'learner/total_loss', 
            stats['average_value_loss'] + stats['average_policy_loss'], 
            step
        )
        self.writer.add_scalar(
            'learner/policy_loss', 
            stats['average_policy_loss'], step)
        self.writer.add_scalar(
            'learner/approx_kl_div', 
            stats['average_approx_kl_div'], step)
        self.writer.add_scalar(
            'learner/value_loss', 
            stats['average_value_loss'], step)
        self.writer.add_scalar(
            'learner/int_value_loss', 
            stats['average_int_value_loss'], step)
        self.writer.add_scalar(
            'learner/rnd_loss', 
            stats['average_rnd_loss'], step)

        self.writer.add_scalar(
            'learner/rnd_mean', 
            stats['average_rnd_mean'], step)
        self.writer.add_scalar(
            'learner/rnd_var', 
            stats['average_rnd_var'], step)
        self.writer.add_scalar(
            "learner/int_explained_variance", 
            stats["int_explained_variance"], step)


        self.writer.add_scalar(
            "learner/int_rewems", 
            stats["int_rewems"], step)

        self.writer.add_scalar(
            "learner/n_updates", 
            stats["n_updates"], step)
        self.writer.add_scalar(
            "learner/explained_variance", 
            stats["explained_variance"], step)
        self.writer.add_scalar(
            "learner/average_dda_loss", 
            dda_stats["average_dda_loss"], step)
        
        self.writer.add_scalar(
            "learner/average_dda_accuracy", 
            dda_stats["average_dda_accuracy"], step)
        
        self.writer.add_scalar(
                f"learner/curr_lr",
                stats["curr_lr"], step
            )
        self.writer.add_scalar(
                f"learner/win_rate",
                self.statistics.cum_win_rate(), step
            )
        
        self.writer.add_scalar(
            "learner/entropy_coef", 
            stats["entropy_coef"], step)
        if "dda_curr_lr" in stats.keys():
            self.writer.add_scalar(
                    f"learner/dda_curr_lr",
                    stats["dda_curr_lr"], step
                )
        
        if "dda_n_updates" in stats.keys():
            self.writer.add_scalar(
                    f"learner/dda_n_updates",
                    stats["dda_n_updates"], step
                )
        
        stats["curr_step"] = self.statistics.curr_step
        stats["curr_num_battle"] = self.statistics.cum_num_battle
        stats["curr_cum_num_win_battle"] = self.statistics.cum_num_win_battle
        
        #stats.update(dda_stats)
        with open(os.path.join(self.config.out_dir, "train_record.ndjson"), "a") as f:
            writer = ndjson.writer(f)
            writer.writerow(stats)



class Population:
    def __init__(self, config) -> None:
        
        self.unique_index = format(os.getpid(), f"08d")
        self.id = 0
        self.env_dict = {}
        self.config = config
        
        self.oppo_list = []
        self.player_list = []
        self.model = MainModel(
            *self.config.return_obses_dim()
        )
        self.rnd_model = RNDModel(
            *self.config.return_obses_dim()
        )
        self.past_oppo_prob = 0.2
        self.oppo_agentbuffer = AgentBuffer(self.config.model_path)
        
        self.config: Config = config
        
        self.env_algorithm_kwarg_template = {
            'agent':None, 
            'steps':self.config.steps, 
        }
        self.learner = Learner(config)
        
        self.writer = self.learner.writer
        self.statistics = self.learner.statistics

        
    def update_populations(self, dirname):
        
        
        # giving the model to the all opponents and players
        for env in self.env_dict.values():
            env["player"].load(dirname)
            env["player"].load_rnd(self.learner.rnd_outdir)
            if random.random() < self.past_oppo_prob:
                # ranodm < 0.2
                env["opponent"].load(self.oppo_agentbuffer.random_sample())
            else:
                env["opponent"].load(dirname)
            
    def update_if_dataset_is_ready(self):
        do_train, dirname = self.learner.update_if_dataset_is_ready()
            
        if do_train:
            self.oppo_agentbuffer.add_models_path(dirname)
            #print(self.oppo_agentbuffer._keys)
            self.update_populations(dirname)
            self.writer_hook()
        return do_train
    
    def reset_battles(self):
        for player, oppo in zip(self.player_list, self.oppo_list):
            player.reset_battles()
            
            oppo.reset_battles()
            

    def _create_one_agent(self):
        if self.config.env_oracle_param is not None:
            oracle_param = self.config.env_oracle_param[self.id]
        else:
            oracle_param = 0
        env = base_env(
            "P"+self.unique_index+str(self.id),
            oracle_param=oracle_param,
            replay_path=self.config.replay_outdir
            )
        env.seed(self.id)
        env.statistics = self.statistics
        
        env._battle_count_queue = Queue(1000)
        env._start_new_battle = True    
        #optim = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        
        agent = PPO(
            
            obs_normalizer=self.learner.obs_nomalizer,
            rnd_model=self.rnd_model,
            rnd_normalizer=RunningMeanStd(),
            max_recurrent_sequence_len=self.config.max_recurrent_sequence_len,
            recurrent=True,
            model=self.model,
            optimizer=None,
            value_func_coef=self.config.vf_coef,
            gamma=self.config.gamma,
            update_interval=self.config.update_interval,
            minibatch_size=self.config.minibatch_size,
            epochs=self.config.epochs,
            memory=self.learner.memory
        )
        
        
        oppo_agent = PPO(
            
            obs_normalizer=self.learner.obs_nomalizer,
            max_recurrent_sequence_len=self.config.max_recurrent_sequence_len,
            recurrent=True,
            model=copy.deepcopy(self.model),
            optimizer=None,
            value_func_coef=self.config.vf_coef,
            gamma=self.config.gamma,
            update_interval=self.config.update_interval,
            minibatch_size=self.config.minibatch_size,
            epochs=self.config.epochs
        )
        oppo_agent.training = False
        oppo = OppoPlayer("O"+self.unique_index+str(self.id), oppo_agent)
        self.oppo_list.append(oppo)
        self.player_list.append(env)
        
        

        # oracle
        if self.config.use_oracle:
            env.oppo_info_buffer = OppoInfoBuffer()
            oppo.oppo_info_buffer = env.oppo_info_buffer
            
        # if trained model == True, loading the model in advance
        if self.config.trained_model:
            agent.load(self.config.trained_model)
            if random.random() < self.past_oppo_prob:
                # ranodm < 0.2
                oppo_agent.load(self.oppo_agentbuffer.random_sample())
            else:
                oppo_agent.load(self.oppo_agentbuffer.get_latest_model())

        if self.config.rnd_normalizer_path:
            agent.load_rnd_normalizer(self.config.rnd_normalizer_path)
        
        if self.config.rnd_path:
            agent.load_rnd(self.config.rnd_path)
        
        if self.config.env_steps is not None:
            agent.global_step = self.config.env_steps[self.id]
            
        


        player_env_algorithm_kwargs = self.env_algorithm_kwarg_template.copy()
        player_env_algorithm_kwargs["agent"] = agent
        player_env_algorithm_kwargs["outdir"] = f"src/pytorch_experiment_results/.experiment_output_{self.id}/",
        
        Thread(
            target=lambda: env_algorithm_wrapper(
                env, self.id, player_env_algorithm_kwargs
                )
        ).start()
        
        
        
        return (agent, oppo_agent)
    

    def create_n_agent(self, n):
        for _ in range(n):
            agent, oppo_agent = self._create_one_agent()
            new_env_info = {"player": agent, "opponent":oppo_agent}
            self.env_dict[self.id] = new_env_info
            self.id += 1
        return self.player_list, self.oppo_list
    def writer_hook(self):  
        
        for id in range(len(self.env_dict)):
            agent: PPO = self.env_dict[id]["player"]
            stats = dict(agent.get_statistics())
            self.writer.add_scalar(
                f'env_id:{id}/entropy', 
                stats['average_entropy'], agent.global_step)
            self.writer.add_scalar(
                f'env_id:{id}/average_int_reward', 
                stats['average_int_reward'], agent.global_step)
            self.writer.add_scalar(
                f'env_id:{id}/int_value', 
                stats['average_int_value'], agent.global_step)

            self.writer.add_scalar(
                f'env_id:{id}/average_length_episode', 
                stats['average_length_episode'], 
                agent.global_step)

            self.writer.add_scalar(
                f'env_id:{id}/oracle_param',
                self.player_list[id].oracle_param, agent.global_step)
            stats["curr_step"] = agent.global_step
            stats["oracle_param"] = self.player_list[id].oracle_param
            # for resume
            with open(os.path.join(self.config.out_dir, f"env{id}_record.ndjson"), "a") as f:
                writer = ndjson.writer(f)
                writer.writerow(stats)
    def get_agent(self, index)->PPO:
        return list(self.env_dict.values())[index]["player"]
        
   

class DDAPopulation:
    def __init__(self, config: Config) -> None:
        
        self.unique_index = format(os.getpid(), f"08d")
        self.id = 0
        self.env_dict = {}
        self.config = config
        self.oppo_list = []
        self.player_list = []
        # recording game info
        self.model = MainModel(
            *self.config.return_obses_dim()
        )
        self.win_predictor = LSTM_Predictor(
            *self.config.return_obses_dim()
        )
        self.rnd_model = RNDModel(
            *self.config.return_obses_dim()
        )
        
        self.config: Config = config
        
        self.env_algorithm_kwarg_template = {
            'agent':None, 
            'steps':self.config.steps, 
        }
        
        self.learner = DDALearner(config)
       
        
        self.statistics = self.learner.statistics
        
        self.oppo_info_buffer = OppoInfoBuffer()
        
        self.writer = self.learner.writer

        self.oppo_agentbuffer = AgentBuffer(self.config.oppo_model_path)
        
   
    def update_populations(self, dirname):
        
        for env in self.env_dict.values():
            
            # players: update the model to the latest
            env["player"].load(dirname)
            env["player"].load_rnd(self.learner.rnd_outdir)
            env["player"].load_win_predictor(self.learner.win_predictor_outdir)

            # opponents: sample differen agent for each env 
            
            env["opponent"].load(self.oppo_agentbuffer.random_sample())
        
            
        
    def update_if_dataset_is_ready(self):
        
       
        # if meeting the amount of data, train agent
        do_train, dirname = self.learner.update_if_dataset_is_ready()
        
        
        # processing after training 
        if do_train:
            self.update_populations(dirname)
            self.writer_hook()
        return do_train
            
    
    def reset_battles(self):
        for player, oppo in zip(self.player_list, self.oppo_list):
            player.reset_battles()
            
            oppo.reset_battles()
            
    def _create_one_agent(self):
        # env pkayer
        if self.config.env_oracle_param is not None:
            oracle_param = self.config.env_oracle_param[self.id]
        else:
            oracle_param = 0
        env = dda_env(
            "PDDA"+self.unique_index+str(self.id), 
            oracle_param=oracle_param,
            replay_path=self.config.replay_outdir
            )
        env.seed(self.id)
        env.statistics = self.statistics
        
        #env._battle_count_queue = Queue(1000)
        env._start_new_battle = True    
        #optim = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        env.oppo_info_buffer = self.oppo_info_buffer
        agent = DDA_PPO(
            
            obs_normalizer=self.learner.obs_nomalizer,
            rnd_model=self.rnd_model,
            rnd_normalizer=RunningMeanStd(),
            max_recurrent_sequence_len=self.config.max_recurrent_sequence_len,
            recurrent=True,
            model=self.model,
            optimizer=None,
            value_func_coef=self.config.vf_coef,
            gamma=self.config.gamma,
            update_interval=self.config.update_interval,
            minibatch_size=self.config.minibatch_size,
            epochs=self.config.epochs,
            memory=self.learner.memory,
            win_predictor=self.win_predictor
        )
        
        # oppo
        oppo_agent = PPO(
            
            obs_normalizer=self.learner.obs_nomalizer,
            max_recurrent_sequence_len=self.config.max_recurrent_sequence_len,
            recurrent=True,
            model=copy.deepcopy(self.model),
            optimizer=None,
            value_func_coef=self.config.vf_coef,
            gamma=self.config.gamma,
            update_interval=self.config.update_interval,
            minibatch_size=self.config.minibatch_size,
            epochs=self.config.epochs
        )
        oppo_agent.training = False
        oppo = OppoPlayer("ODDA"+self.unique_index+str(self.id), oppo_agent)
        
        # if trained model == True, loading the model in advance
        if self.config.trained_model:
            agent.load(self.config.trained_model)
            oppo_agent.load(self.oppo_agentbuffer.random_sample())
        
        # append to list
        self.oppo_list.append(oppo)
        self.player_list.append(env)
        # oracle
        if self.config.use_oracle:
            env.oppo_info_buffer = OppoInfoBuffer()
            oppo.oppo_info_buffer = env.oppo_info_buffer
        
        if self.config.rnd_normalizer_path:
            agent.load_rnd_normalizer(self.config.rnd_normalizer_path)
        
        if self.config.rnd_path:
            agent.load_rnd(self.config.rnd_path)

        if self.config.win_predictor_path:
            agent.load_win_predictor(self.config.win_predictor_path)
        
        if self.config.env_steps is not None:
            agent.global_step = self.config.env_steps[self.id]
        

      
        player_env_algorithm_kwargs = self.env_algorithm_kwarg_template.copy()
        player_env_algorithm_kwargs["agent"] = agent
        player_env_algorithm_kwargs["outdir"] = f"src/pytorch_experiment_results/.experiment_output_{self.id}/",
       
        Thread(
            target=lambda: env_algorithm_wrapper(
                env, self.id, player_env_algorithm_kwargs
                )
        ).start()
        
        
        return (agent, oppo_agent)
    
    def create_n_agent(self, n):
        for _ in range(n):
            agent, oppo_agent = self._create_one_agent()
            new_env_info = {"player": agent, "opponent":oppo_agent}
            self.env_dict[self.id] = new_env_info
            self.id += 1
        return self.player_list, self.oppo_list
    def writer_hook(self):  
        for id in range(len(self.env_dict)):
            agent: DDA_PPO = self.env_dict[id]["player"]
            stats = dict(agent.get_statistics())
            dda_stats = dict(agent.get_dda_statistics())

            self.writer.add_scalar(
                f'env_id:{id}/entropy', 
                stats['average_entropy'], agent.global_step
                )
            self.writer.add_scalar(
                f'env_id:{id}/average_int_reward',
                 stats['average_int_reward'], agent.global_step)
            self.writer.add_scalar(
                f'env_id:{id}/oracle_param',
                self.player_list[id].oracle_param, agent.global_step)

            self.writer.add_scalar(
                f'env_id:{id}/int_value', 
                stats['average_int_value'], agent.global_step)
            self.writer.add_scalar(
                f'env_id:{id}/average_length_episode', 
                stats['average_length_episode'], 
                agent.global_step)
            self.writer.add_scalar(
                f'env_id:{id}/average_reward', 
                dda_stats['average_reward'], agent.global_step)
            
            self.writer.add_scalar(
                f'env_id:{id}/average_final_reward', 
                dda_stats['average_final_reward'], agent.global_step)
                
            self.writer.add_scalar(
                f'env_id:{id}/average_dda_alpha', 
                dda_stats['average_dda_alpha'], agent.global_step)
            
            self.writer.add_scalar(
                f'env_id:{id}/average_dda_reward', 
                dda_stats['average_dda_reward'], agent.global_step)
            stats["curr_step"] = agent.global_step
            stats["oracle_param"] = self.player_list[id].oracle_param
            #stats.update(dda_stats)
            # for resume
            with open(os.path.join(self.config.out_dir, f"env{id}_record.ndjson"), "a") as f:
                writer = ndjson.writer(f)
                writer.writerow(stats)

            

  




    