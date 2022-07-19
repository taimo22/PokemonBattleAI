
from typing import Any, Callable, List, Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn.utils.rnn import (
    pad_packed_sequence,
    pad_sequence, 
    pack_sequence,
    pack_padded_sequence
    
    )
from torch._C import Value, device
from torch.nn.functional import relu

from pfrl.nn.recurrent import Recurrent
import numpy as np
import pfrl


def torch_feature_categorizer_2d( 
                        obs: torch.Tensor,
                        total_obs_dim, 
                        battle_obs_dim,
                        player_obs_dim, 
                        oppo_obs_dim,
                        others_obs_dim,
                        total_poke_obs_dim,
                        moves_obs_dim
                        )->Tuple[np.ndarray, List[np.ndarray]]:
        # action masking
        
        masked_actions = obs[:, :, :14]
        obs = obs[:, :,  14:]
       
        assert obs.shape[-1] == total_obs_dim 
        
        battle_info = obs[:, :, : battle_obs_dim]
        player_info = obs[:, :,  battle_obs_dim : battle_obs_dim+player_obs_dim]
        oppo_info = obs[:, :, battle_obs_dim+player_obs_dim : battle_obs_dim+player_obs_dim+oppo_obs_dim]
        
        '''
        oppo info
        
        '''
        # oppo info 
        oppo_pokes_info = torch.split(
            oppo_info[:, :,  :-others_obs_dim],
            (oppo_obs_dim-1)//6, 
            dim=-1
        )
        oppo_others_info = oppo_info[:, :,  -others_obs_dim:]
        
        
        
        # move info for preprocess
        
        
        '''
        player info
        
        '''
        player_pokes_info = list(
            
            torch.split(
            player_info[:, :, :-others_obs_dim], 
            (player_obs_dim-1)//6,
            dim=-1
        )
        )
        
        # info about team conditon
        player_others_info = player_info[:, :, -others_obs_dim:]
        
        # info about poekmons the player has

        player_act_poke_info = player_pokes_info[0]
    
        
        
        # move info for preprocess
       
        moves_info_post = []
    
        #move
        dmax_move_obs_dim = 23 * 4
        player_move_info = player_act_poke_info[:, :, -(moves_obs_dim+dmax_move_obs_dim):-dmax_move_obs_dim]
        player_dmaxmove_info = player_act_poke_info[:, :, -(dmax_move_obs_dim):]
        
        player_move_1, player_move_2, player_move_3, player_move_4 = torch.split(
            player_move_info, 
            player_move_info.shape[-1]//4, dim=-1)
        player_dmaxmove_1, player_dmaxmove_2, player_dmaxmove_3, player_dmaxmove_4 = torch.split(
            player_dmaxmove_info, 
            player_dmaxmove_info.shape[-1]//4, dim=-1)
        
        
        # info for post processing
        
        moves_info_post = [
                player_move_1, player_move_2, player_move_3, player_move_4,
                player_dmaxmove_1, player_dmaxmove_2, player_dmaxmove_3, player_dmaxmove_4
            ]
                
                
        #player_preprocess_input_data = player_moves_pre + player_poke_pre_no_move + [player_others_info]
        player_input_data = player_pokes_info + [player_others_info] 
        oppo_input_data = list(oppo_pokes_info) + [oppo_others_info] 
        
        
        
        '''
        [
            input_active_poke, input_reserved_poke1,
            input_reserved_poke2,input_reserved_poke3,
            input_reserved_poke4,input_reserved_poke5, 
            input_others
        ],
        '''
        return (
            player_input_data, 
            oppo_input_data, 
            battle_info, 
            moves_info_post, 
            masked_actions, 
            player_pokes_info
            )
    
    
def torch_feature_categorizer( 
                        obs: torch.Tensor,
                        total_obs_dim, 
                        battle_obs_dim,
                        player_obs_dim, 
                        oppo_obs_dim,
                        others_obs_dim,
                        total_poke_obs_dim,
                        moves_obs_dim
                        )->Tuple[np.ndarray, List[np.ndarray]]:
        # action masking
        
        masked_actions = obs[:, :14]
        obs = obs[:,  14:]
       
        assert obs.shape[-1] == total_obs_dim 
        
        battle_info = obs[:, : battle_obs_dim]
        player_info = obs[ :,  battle_obs_dim : battle_obs_dim+player_obs_dim]
        oppo_info = obs[:, battle_obs_dim+player_obs_dim : battle_obs_dim+player_obs_dim+oppo_obs_dim]
        
        '''
        oppo info
        
        '''
        # oppo info 
        oppo_pokes_info = torch.split(
            oppo_info[:,  :-others_obs_dim],
            (oppo_obs_dim-1)//6, 
            dim=-1
        )
        oppo_others_info = oppo_info[ :,  -others_obs_dim:]
        
        
        
        # move info for preprocess
        
        
        '''
        player info
        
        '''
        player_pokes_info = list(
            
            torch.split(
            player_info[:, :-others_obs_dim], 
            (player_obs_dim-1)//6,
            dim=-1
        )
        )
        
        # info about team conditon
        player_others_info = player_info[:, -others_obs_dim:]
        
        # info about poekmons the player has

        player_act_poke_info = player_pokes_info[0]
    
        
        
        # move info for preprocess
       
        moves_info_post = []
    
        #move
        dmax_move_obs_dim = 23 * 4
        player_move_info = player_act_poke_info[ :, -(moves_obs_dim+dmax_move_obs_dim):-dmax_move_obs_dim]
        player_dmaxmove_info = player_act_poke_info[ :, -(dmax_move_obs_dim):]
        
        player_move_1, player_move_2, player_move_3, player_move_4 = torch.split(
            player_move_info, 
            player_move_info.shape[-1]//4, dim=-1)
        player_dmaxmove_1, player_dmaxmove_2, player_dmaxmove_3, player_dmaxmove_4 = torch.split(
            player_dmaxmove_info, 
            player_dmaxmove_info.shape[-1]//4, dim=-1)
        
        
        # info for post processing
        
        moves_info_post = [
                player_move_1, player_move_2, player_move_3, player_move_4,
                player_dmaxmove_1, player_dmaxmove_2, player_dmaxmove_3, player_dmaxmove_4
            ]
                
                
        #player_preprocess_input_data = player_moves_pre + player_poke_pre_no_move + [player_others_info]
        player_input_data = player_pokes_info + [player_others_info] 
        oppo_input_data = list(oppo_pokes_info) + [oppo_others_info] 
        
        
        
        '''
        [
            input_active_poke, input_reserved_poke1,
            input_reserved_poke2,input_reserved_poke3,
            input_reserved_poke4,input_reserved_poke5, 
            input_others
        ],
        '''
        return (
            player_input_data, 
            oppo_input_data, 
            battle_info, 
            moves_info_post, 
            masked_actions, 
            player_pokes_info
            )

class BasePreModel(nn.Module):
    '''
    implement a simple neural network which the Agent will use to make decisions
    Since we are using PPO in this example, 
    the neural network need to have both a value head and policy head
    and forward() should return both action distribution(policy) and state value
    more details available in the PPO papar
    https://arxiv.org/abs/1707.06347
    '''
    def __init__(self, 
                 hidden_sizes, 
                  
                 total_poke_obs_dim,
                 others_obs_dim,
                ):
        super().__init__()
        
        self.others_obs_dim = others_obs_dim
        self.total_poke_obs_dim = total_poke_obs_dim
        
        self.hidden_sizes = hidden_sizes
        # first: encoding of each poke info 
        self.active_poke_dense = nn.Linear(
            self.total_poke_obs_dim, 
            hidden_sizes[0]
        )
        
        self.reserved_poke1_dense = nn.Linear(
            self.total_poke_obs_dim, 
            hidden_sizes[0]
        )
        self.reserved_poke2_dense = nn.Linear(
            self.total_poke_obs_dim, 
            hidden_sizes[0]
        )
        
        self.reserved_poke3_dense = nn.Linear(
            self.total_poke_obs_dim, 
            hidden_sizes[0]
        )
        
        self.reserved_poke4_dense = nn.Linear(
            self.total_poke_obs_dim, 
            hidden_sizes[0]
        )
        
        self.reserved_poke5_dense = nn.Linear(
            self.total_poke_obs_dim, 
            hidden_sizes[0]
        )
        
        # Second: concat the all encoded reserved poke info
        self.reserved_pokes_dense = nn.Linear(
            hidden_sizes[0]*5, 
            hidden_sizes[1]
        )
        
        self.all_dense =  nn.Linear(
            hidden_sizes[0]+hidden_sizes[1]+self.others_obs_dim, 
            hidden_sizes[2]
        )
        

    def forward(self, obs: List):
        '''
        input: obs(Tensor(dtype=float32))
        return : Tuple(policy(torch.distributions), value(Tensor(dtype=float32)))
        '''
        assert isinstance(obs, list)
        
        
        
        
        active_poke_dense_out = relu(self.active_poke_dense(obs[0]))
        
        reserved_poke1_dense_out = relu(self.reserved_poke1_dense(obs[1]))
        reserved_poke2_dense_out = relu(self.reserved_poke2_dense(obs[2]))
        reserved_poke3_dense_out = relu(self.reserved_poke3_dense(obs[3]))
        reserved_poke4_dense_out = relu(self.reserved_poke4_dense(obs[4]))
        reserved_poke5_dense_out = relu(self.reserved_poke5_dense(obs[5]))
        
        reserved_pokes_concat = torch.cat(
            [
            reserved_poke1_dense_out, 
            reserved_poke2_dense_out,
            reserved_poke3_dense_out,
            reserved_poke4_dense_out,
            reserved_poke5_dense_out
            ], dim=-1
            )
        
        reserved_pokes_dense_out = relu(
            self.reserved_pokes_dense(
                reserved_pokes_concat
                )
            )
        
        all_concat = torch.cat(
            [
             active_poke_dense_out, 
             reserved_pokes_dense_out,
             obs[6]
             
            ], dim=-1
            )
        all_dense_out = relu(self.all_dense(all_concat))
        
        
        return all_dense_out


class BasePostModel(nn.Module):
    '''
    implement a simple neural network which the Agent will use to make decisions
    Since we are using PPO in this example, 
    the neural network need to have both a value head and policy head
    and forward() should return both action distribution(policy) and state value
    more details available in the PPO papar
    https://arxiv.org/abs/1707.06347
    '''
    def __init__(self,
                 move_input_dim: int, 
                dmax_move_input_dim: int,
                reserved_poke_dim: int,
                val_input_dim: int,
                is_dda
                ):
        super().__init__()
        
        self.move_input_dim = move_input_dim
        self.dmax_move_input_dim = dmax_move_input_dim
        self.reserved_poke_dim = reserved_poke_dim
        self.val_input_dim = val_input_dim
        
        #move
        self.out_move1 = nn.Linear(
            self.move_input_dim,
            1
            )
        
        self.out_move2 = nn.Linear(
            self.move_input_dim,
            1
            )
        
        self.out_move3 = nn.Linear(
            self.move_input_dim,
            1
            )
        self.out_move4 = nn.Linear(
            self.move_input_dim,
            1
            )
        
        # dmax move
        self.out_dmaxmove1 = nn.Linear(
            self.dmax_move_input_dim,
            1
        )
        
        self.out_dmaxmove2 = nn.Linear(
            self.dmax_move_input_dim,
            1
        )
        self.out_dmaxmove3 = nn.Linear(
            self.dmax_move_input_dim,
            1
        )
        
        self.out_dmaxmove4 = nn.Linear(
            self.dmax_move_input_dim,
            1
        )
        
        # out for reserved pokes
        self.out_reserved_poke1 = nn.Linear(
            self.reserved_poke_dim,
            1
        )
        
        self.out_reserved_poke2 = nn.Linear(
            self.reserved_poke_dim,
            1
        )
        
        self.out_reserved_poke3 = nn.Linear(
            self.reserved_poke_dim,
            1
        )
        
        self.out_reserved_poke4 = nn.Linear(
            self.reserved_poke_dim,
            1
        )
        self.out_reserved_poke5 = nn.Linear(
            self.reserved_poke_dim,
            1
        )
        
        self.out_reserved_poke6 = nn.Linear(
            self.reserved_poke_dim,
            1
        )
        
       
        # model for value function
        
        self.value_dense = nn.Linear(
            val_input_dim,
            64
        )
        self.value_logit = nn.Linear(
            64,
            1
        )
        
        self.int_value_head = nn.Linear(64, 1)
        
        self.is_dda = is_dda
        if self.is_dda:
            self.dda_rnd_head =  nn.Linear(64, 1)                                                                              
        
        
        
    
    def forward(self, obs: List, action_masks):
        '''
        input: obs(Tensor(dtype=float32))
        return : Tuple(policy(torch.distributions), value(Tensor(dtype=float32)))
        '''
        
        
        move1_out = relu(self.out_move1(obs[0]))
        move2_out = relu(self.out_move2(obs[1]))
        move3_out = relu(self.out_move3(obs[2]))
        move4_out = relu(self.out_move4(obs[3]))
        
        dmax_move1_out = relu(self.out_dmaxmove1(obs[4]))
        dmax_move2_out = relu(self.out_dmaxmove1(obs[5]))
        dmax_move3_out = relu(self.out_dmaxmove1(obs[6]))
        dmax_move4_out = relu(self.out_dmaxmove1(obs[7]))
        
        reserved_poke1_out = relu(self.out_reserved_poke1(obs[8]))
        reserved_poke2_out = relu(self.out_reserved_poke2(obs[9]))
        reserved_poke3_out = relu(self.out_reserved_poke3(obs[10]))
        reserved_poke4_out = relu(self.out_reserved_poke4(obs[11]))
        reserved_poke5_out = relu(self.out_reserved_poke5(obs[12]))
        reserved_poke6_out = relu(self.out_reserved_poke6(obs[13]))
        
        logit = torch.cat(
                (
                    move1_out, move2_out, move3_out, move4_out, 
                    dmax_move1_out, dmax_move2_out, dmax_move3_out, dmax_move4_out,
                    reserved_poke1_out, reserved_poke2_out, reserved_poke3_out, reserved_poke4_out,
                    reserved_poke5_out, reserved_poke6_out
                ), dim=-1
                )
        
        policy_logit = logit * action_masks
        
        
        
        value_dense_out = relu(self.value_dense(obs[14]))
        value = self.value_logit(value_dense_out)
        int_value = self.int_value_head(value_dense_out)

        if self.is_dda:
            dda_value = self.dda_rnd_head(value_dense_out)
            return policy_logit, value, int_value, dda_value
        return policy_logit, value, int_value

class MainModel(Recurrent, nn.Module):
    def __init__(self,
                  
                total_obs_dim, 
                battle_obs_dim,
                player_obs_dim, 
                oppo_obs_dim,
                 
                 moves_obs_dim, 
                 total_poke_obs_dim,
                 others_obs_dim,
                 is_dda=False
                 ) -> None:
        super().__init__()
        
        self.hidden_sizes = [150, 300, 200]
        self.total_obs_dim = total_obs_dim
        self.battle_obs_dim = battle_obs_dim
        self.player_obs_dim = player_obs_dim
        self.oppo_obs_dim = oppo_obs_dim
        self.others_obs_dim = others_obs_dim
        self.total_poke_obs_dim = total_poke_obs_dim
        self.moves_obs_dim = moves_obs_dim
        self.dmax_move_obs_dim = 23
        self.action_size = 14
        
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
        lstm_input_size = self.hidden_sizes[-1]*2+self.battle_obs_dim+self.action_size
        self.lstm = nn.LSTM(
            num_layers=1, 
            input_size=lstm_input_size, 
            hidden_size=120
            )
        
        self.post_process_model = BasePostModel(
            self.moves_obs_dim//4 + 120, 
            self.dmax_move_obs_dim + 120,
            self.total_poke_obs_dim + 120,
            120,
            is_dda
        )
        self.is_dda = is_dda
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor, recurrent_state: torch.Tensor):
        
        #obs size: (seq, batch, feature)
        if len(obs.size()) == 3: # training
            seq_size, batch_size, feature_size = obs.size()
            obs = torch.reshape(obs, (-1, feature_size))
            prev_action = torch.reshape(prev_action, (-1, self.action_size))
        else: # collect data
            seq_size = 1 
            batch_size = 1
        
        # categorizing of features
        (
            player_input_data, 
            oppo_input_data, 
            battle_info, 
            moves_info_post, 
            masked_actions, 
            player_pokes_info
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
            (player_pre_model_out, 
             oppo_pre_model_out,
             battle_info,
             prev_action
             ), axis=-1
            )
        
        
        # LSTM
        if (recurrent_state is not None) and (pre_process_out.device.type != "cpu"):
            recurrent_state = list(recurrent_state)
            recurrent_state[0] = recurrent_state[0].to(self.device)
            recurrent_state[1] = recurrent_state[1].to(self.device)
            recurrent_state = tuple(recurrent_state)
            
        pre_process_out = torch.reshape(pre_process_out, (seq_size, batch_size, -1))
        lstm_out, (h, c) = self.lstm(pre_process_out)
        
        lstm_out = torch.reshape(lstm_out, (-1, lstm_out.shape[-1]))
        concat_features = [
            torch.cat(
            (info, lstm_out),
            axis=-1) 
            for info in moves_info_post + player_pokes_info]
        
        
        concat_features += [lstm_out]

        if self.is_dda:

            # post process (output value and policy)
            (
                policy, 
                value, 
                int_value, 
                dda_value
            ) = self.post_process_model(
                concat_features, 
                masked_actions
                )
            
            policy = torch.reshape(
                policy, (seq_size, batch_size, -1)
                )
            
            value = torch.reshape(
                value, (seq_size, batch_size, -1)
            )
            int_value = torch.reshape(
                int_value, (seq_size, batch_size, -1)
            )
            dda_value = torch.reshape(
                dda_value, (seq_size, batch_size, -1)
            )
        
            return ((policy, value, int_value, dda_value), (h, c))
        else:
            # post process (output value and policy)
            (
                policy, 
                value, 
                int_value
            ) = self.post_process_model(
                concat_features, 
                masked_actions
                )
            
            policy = torch.reshape(
                policy, (seq_size, batch_size, -1)
                )
            
            value = torch.reshape(
                value, (seq_size, batch_size, -1)
            )
            int_value = torch.reshape(
                int_value, (seq_size, batch_size, -1)
            )
            
            return ((policy, value, int_value), (h, c))
  