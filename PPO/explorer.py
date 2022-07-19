
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init
from model import BasePreModel, torch_feature_categorizer
from typing import Tuple, List


def torch_feature_categorizer( 
                        obs: torch.Tensor,
                        total_obs_dim, 
                        battle_obs_dim,
                        player_obs_dim, 
                        oppo_obs_dim,
                        others_obs_dim,
                        total_poke_obs_dim,
                        moves_obs_dim
                        )->Tuple[List[torch.Tensor]]:

        masked_actions = obs[:, :14]
        obs = obs[:,  14:]
       
        assert obs.shape[-1] == total_obs_dim 
        
        battle_info = obs[:, : battle_obs_dim]
        player_info = obs[ :,  battle_obs_dim : battle_obs_dim+player_obs_dim]
        oppo_info = obs[:, battle_obs_dim+player_obs_dim : battle_obs_dim+player_obs_dim+oppo_obs_dim]

        # oppo info 
        oppo_pokes_info = torch.split(
            oppo_info[:,  :-others_obs_dim],
            (oppo_obs_dim-1)//6, 
            dim=-1
        )
        oppo_others_info = oppo_info[ :,  -others_obs_dim:]
        
        
        
        # move info for preprocess
        player_pokes_info = list(
            
            torch.split(
            player_info[:, :-others_obs_dim], 
            (player_obs_dim-1)//6,
            dim=-1
        )
        )
        
        # info about team conditon
        player_others_info = player_info[:, -others_obs_dim:]
        
       
                
        #player_preprocess_input_data = player_moves_pre + player_poke_pre_no_move + [player_others_info]
        player_input_data = player_pokes_info + [player_others_info] 
        oppo_input_data = list(oppo_pokes_info) + [oppo_others_info] 

        return (
            player_input_data, 
            oppo_input_data, 
            battle_info,
        )

class RNDPredictorModel(nn.Module):
    def __init__(self,
                total_obs_dim, 
                battle_obs_dim,
                player_obs_dim, 
                oppo_obs_dim,
                 
                 moves_obs_dim, 
                 total_poke_obs_dim,
                 others_obs_dim,
                ):
        super().__init__()
        self.total_obs_dim = total_obs_dim
        self.battle_obs_dim = battle_obs_dim
        self.player_obs_dim = player_obs_dim
        self.oppo_obs_dim = oppo_obs_dim
        self.others_obs_dim = others_obs_dim
        self.total_poke_obs_dim = total_poke_obs_dim
        self.moves_obs_dim = moves_obs_dim
        self.dmax_move_obs_dim = 23
        self.action_size = 14

        self.hidden_sizes = [156, 356, 256]

        self.player_pre_model = BasePreModel(
            self.hidden_sizes, 
            total_poke_obs_dim,
            others_obs_dim,
        )
        self.oppo_pre_model = BasePreModel(
            self.hidden_sizes, 
                  
            total_poke_obs_dim,
            others_obs_dim,
        )
        self.preprocess_output = self.hidden_sizes[-1]*2+battle_obs_dim + self.action_size
        self.output_size = 128

        self.output_dense = nn.Linear(
            self.preprocess_output,
            self.output_size
        )

    
    def forward(self, obs: torch.Tensor, act):
        # categorizing of features
        (
            player_input_data, 
            oppo_input_data, 
            battle_info
        ) = torch_feature_categorizer( 
                        obs,
                        self.total_obs_dim, 
                        self.battle_obs_dim,
                        self.player_obs_dim, 
                        self.oppo_obs_dim,
                        self.others_obs_dim,
                        self.total_poke_obs_dim,
                        self.moves_obs_dim
                        )
        # forwarding 
        player_pre_model_out = self.player_pre_model(player_input_data)
        oppo_pre_model_out = self.oppo_pre_model(oppo_input_data)
        
        pre_process_out = torch.cat(
            (
            player_pre_model_out, 
            oppo_pre_model_out,
            battle_info,
            act
            ), axis=-1
        )
        
        out = self.output_dense(pre_process_out)

        return out
        

class RNDModel(nn.Module):
    
    def __init__(self,
                total_obs_dim, 
                battle_obs_dim,
                player_obs_dim, 
                oppo_obs_dim,
                 
                 moves_obs_dim, 
                 total_poke_obs_dim,
                 others_obs_dim,):
        super().__init__()
        
        self.total_obs_dim = total_obs_dim
        self.battle_obs_dim = battle_obs_dim
        self.player_obs_dim = player_obs_dim
        self.oppo_obs_dim = oppo_obs_dim
        self.others_obs_dim = others_obs_dim
        self.total_poke_obs_dim = total_poke_obs_dim
        self.moves_obs_dim = moves_obs_dim
        
        
        self.predictor = RNDPredictorModel(
            total_obs_dim=self.total_obs_dim,
            battle_obs_dim=self.battle_obs_dim,
            player_obs_dim=self.player_obs_dim,
            oppo_obs_dim=self.oppo_obs_dim,
            moves_obs_dim=self.moves_obs_dim,
            total_poke_obs_dim=self.total_poke_obs_dim,
            others_obs_dim=self.others_obs_dim
        )

        self.target = RNDPredictorModel(
            total_obs_dim=self.total_obs_dim,
            battle_obs_dim=self.battle_obs_dim,
            player_obs_dim=self.player_obs_dim,
            oppo_obs_dim=self.oppo_obs_dim,
            moves_obs_dim=self.moves_obs_dim,
            total_poke_obs_dim=self.total_poke_obs_dim,
            others_obs_dim=self.others_obs_dim
        )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

        self.loss_func = nn.MSELoss(reduction="none")
        
        

    def forward(self, next_obs, act):
        target_feature = self.target(next_obs, act)
        predict_feature = self.predictor(next_obs, act)

        return predict_feature, target_feature

    def num_params(self):
        print(self.predictor)
        params = 0
        for p in self.predictor.parameters():
            if p.requires_grad:
                params += p.numel()
        return params
