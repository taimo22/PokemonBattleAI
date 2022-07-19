import random
from types import new_class
from typing import Sequence, Optional, Any, List


import numpy as np

from typing import Any, Callable, List, Dict, Optional, Tuple, Union
import pickle
import pandas as pd
import os
import torch
from torch import nn
from torch import optim
from torch.nn.modules import loss

from model import BasePreModel
from torch.nn.utils.rnn import (
    pad_packed_sequence,
    pad_sequence, 
    pack_sequence,
    pack_padded_sequence
    
    )
from torch.nn.functional import relu, sigmoid

class OppoInfoBuffer:
    def __init__(self) -> None:
        self.buffer = []
        self.turns = []
        self.act_pokes = []
        
    def reset(self):
        self.buffer = []
        self.turns = []
        self.act_pokes = []

class PokeBattle:
    def __init__(self) -> None:
        self.is_win = None
        
        self.turn = []
        self.actions_history = []
        self.rewards = []
        self.recorded_states = []
        
        #self.discount = discount
        
    def length(self):
        return len(self.recorded_states)
        
    def make_obs(self, index):
        return self.recorded_states[index]
    
    def make_action(self, index):
        return torch.unsqueeze(self.actions_history[index], axis=0)
    
    def make_obs_as_seq(self, index, seq_len): 
        #assert index + seq_len <= self.length()
        seq_obs = np.concatenate(self.recorded_states[index: index+seq_len], axis=0)
        
        return torch.from_numpy(seq_obs)
    
    def make_obs_as_all_seq(self): 
        #assert index + seq_len <= self.length()
        seq_obs = np.concatenate(self.recorded_states[: ], axis=0)
        return torch.from_numpy(seq_obs)
    
    def append_records(self, turn, state, action, reward):
        
        #if turn not in self.turn:
                
        self.turn.append(turn)
        self.rewards.append(reward)
        onehot_action = torch.zeros(1, 14)
        onehot_action[:, action] = 1
        self.actions_history.append(onehot_action)
        self.recorded_states.append(torch.unsqueeze(state, 0))

class BattleBuffer:
    def __init__(self, buffer_window_size, buffer_batch_size) -> None:
        self.window_size = buffer_window_size
        self.batch_size = buffer_batch_size
        self.buffer = []
        
        self.battle_len = []
        
        self.win_buffer = []
        self.lose_buffer = []
        
        self.win_battle_len = []
        self.lose_battle_len = []
    
    @property
    def num_sample(self):
        return sum(self.battle_len)
    
    @property
    def win_data_ratio(self):
        return len(self.win_buffer)/len(self.buffer)
        
    def save_game(self, game: PokeBattle):
        # if the capacity of the buffer is full
        if len(self.buffer) > self.window_size:
            n = self.battle_len.index(min(self.battle_len))
            # n = 0
            poped_buffer = self.buffer.pop(0)
            if self.win_buffer[0] == poped_buffer:
                self.win_buffer.pop(0)
            else:
                self.lose_buffer.pop(0)
            
            poped_len = self.battle_len.pop(0)
            if self.win_battle_len[0] == poped_len:
                self.win_battle_len.pop(0)
            else:
                self.lose_battle_len.pop(0)
            
        assert game.is_win == 1 or game.is_win == 0
        if game.is_win == 1:
            self.win_buffer.append(game)
            self.win_battle_len.append(game.length())
        else:
            self.lose_buffer.append(game)
            self.lose_battle_len.append(game.length())
        self.buffer.append(game)
        self.battle_len.append(game.length())
    
    
    
    def make_balanced_batch(self, device):
        balanced_size = min(len(self.win_buffer), len(self.lose_buffer))
        np.random.shuffle(self.win_buffer)
        np.random.shuffle(self.lose_buffer)
        
        games = self.win_buffer[:balanced_size] + self.lose_buffer[:balanced_size]
        
        
        # list
        obs_list = [g.make_obs_as_all_seq()
                            for g in games]
        act_list = [ torch.cat(g.actions_history[:])
                            for g in games]
        iswin_list = [[g.is_win] for g in games]
        
        
        
        # converts to tensor
        obs_tensor = pad_sequence(
                                obs_list
                                ).to(device=device)
        act_tensor = pad_sequence(
            act_list
        ).to(device=device)
        
        
        iswin_tensor = torch.from_numpy(
            np.expand_dims(np.concatenate(iswin_list), axis=-1)).float().to(device=device)
        
        
        # split into valid and training data
        total_samples = obs_tensor.size()[1]
        n_val = int(0.2 * total_samples)
        n_train = total_samples - n_val
        random_i = torch.tensor(random.sample(range(total_samples), total_samples)).to(device=device)
        
        
        train_obs = torch.index_select(obs_tensor, 1, random_i[:n_train])
        train_act = torch.index_select(act_tensor, 1, random_i[:n_train])
        train_iswin = torch.index_select(iswin_tensor, 0, random_i[:n_train])
        
        val_obs = torch.index_select(obs_tensor, 1, random_i[n_train:n_train + n_val])
        val_act = torch.index_select(act_tensor, 1, random_i[n_train:n_train + n_val])
        val_iswin = torch.index_select(iswin_tensor, 0, random_i[n_train:n_train + n_val])
        
        return (
            (train_obs, train_act, train_iswin),
            (val_obs, val_act, val_iswin)
            )
            
   
    # Select a random game from the buffer
    def sample_game(self):
        return random.choice(self.buffer)
    
    # Select a random game from the buffer
    def sample_win_game(self):
        return random.choice(self.win_buffer)
    # Select a random game from the buffer
    def sample_lose_game(self):
        return random.choice(self.lose_buffer)
    
    def reset(self):
        self.buffer = []
        
        self.battle_len = []
        
        self.win_buffer = []
        self.lose_buffer = []
        
        self.win_battle_len = []
        self.lose_battle_len = []
   
    
    # Identify a suitable game position.
    def sample_sequence_position(self, game, prediction_steps: int) -> int:
        # Sample position from game either uniformly or according to some priority.
        if game.length() > prediction_steps-1:
            #print(game.length(), prediction_steps)
            return random.randint(0, game.length()-prediction_steps)
        else:
            return 0
    def save(self, path):
        pickle.dump(
                {   
                    "buffer": self.buffer,
                    "battle_len": self.battle_len,
                    "win_buffer": self.win_buffer,
                    "lose_buffer": self.lose_buffer,
                    "win_battle_len": self.win_battle_len,
                    "lose_battle_len": self.lose_battle_len
                },
                open(os.path.join(path, "replay_buffer.pkl"), "wb"),
            )
    
    def load(self, path):
        if path:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        buffer_infos = pickle.load(f)
                    self.buffer = buffer_infos["buffer"]
                    self.battle_len = buffer_infos["battle_len"]
                    self.win_buffer = buffer_infos["win_buffer"]
                    self.lose_buffer = buffer_infos["lose_buffer"]
                    self.win_battle_len = buffer_infos["win_battle_len"]
                    self.lose_battle_len = buffer_infos["lose_battle_len"]
                    
                else:
                    print(
                        f"Warning: Replay buffer path '{path}' doesn't exist.  Using empty buffer."
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

                
                
        #player_preprocess_input_data = player_moves_pre + player_poke_pre_no_move + [player_others_info]
        player_input_data = player_pokes_info + [player_others_info] 
        oppo_input_data = list(oppo_pokes_info) + [oppo_others_info] 
        
        
        
      
        return (
            player_input_data, 
            oppo_input_data, 
            battle_info, 
            )

# model for predicting win rate
class LSTM_Predictor(nn.Module):
    def __init__(self, 
                total_obs_dim, 
                battle_obs_dim,
                player_obs_dim, 
                oppo_obs_dim,
                
                moves_obs_dim, 
                total_poke_obs_dim,
                others_obs_dim,
            
                cell_size = 120,
                action_size = 14,
                
                ):
        super().__init__()
        
        self.total_obs_dim = total_obs_dim
        self.battle_obs_dim = battle_obs_dim
        self.player_obs_dim = player_obs_dim
        self.oppo_obs_dim = oppo_obs_dim
        self.others_obs_dim = others_obs_dim
        self.total_poke_obs_dim = total_poke_obs_dim
        self.moves_obs_dim = moves_obs_dim
        
        pre_input_move_dim = moves_obs_dim//4
        pre_input_poke_dim = total_poke_obs_dim
        
        
        
        self.built = True
        self.cell_size = cell_size
        
        self.training_step = 0
        
        
        self.agent_obs_size = action_size
        
        
        
        hidden_sizes = [248, 512, 128]
        
        
        self.player_pre_model = BasePreModel(
            hidden_sizes, 
                  
            total_poke_obs_dim,
            others_obs_dim,
        )
        self.oppo_pre_model = BasePreModel(
            hidden_sizes, 
                  
            total_poke_obs_dim,
            others_obs_dim,
        )
        lstm_input_size = hidden_sizes[-1] *2 + 19 + self.agent_obs_size
        
        # creating lstm model
        lstm_hidden_size = 120
        self.lstm = nn.LSTM(
            num_layers=1, 
            input_size=lstm_input_size, 
            hidden_size=lstm_hidden_size,
            batch_first=False
            )
        
        # dense for post processing
        
        self.dense1 = nn.Linear(
           lstm_hidden_size, 
           48
        )
        
        self.prob_dense = nn.Linear(
           48, 
           1
        )
       
        
        
       
    def forward(self, inputs, recurrent_state):
        x, actions = inputs
        
        # firstly, dividing x into each category 
        (
            player_preprocess_input_data, 
            oppo_preprocess_input_data, 
            battle_info, 
        ) = torch_feature_categorizer(
            obs=x,
            total_obs_dim=self.total_obs_dim, 
            battle_obs_dim=self.battle_obs_dim,
            player_obs_dim=self.player_obs_dim, 
            oppo_obs_dim=self.oppo_obs_dim,
            others_obs_dim=self.others_obs_dim,
            total_poke_obs_dim=self.total_poke_obs_dim,
            moves_obs_dim=self.moves_obs_dim
            )
        
        
        player_pre_model_out = self.player_pre_model(player_preprocess_input_data)
        oppo_pre_model_out = self.oppo_pre_model(oppo_preprocess_input_data)
        
        pre_process_out = torch.cat(
            (player_pre_model_out, 
             oppo_pre_model_out,
             battle_info,
             actions
             ), axis=-1
            )

        
        #out_time_dim = tf.reshape(out, (-1, max(seq_lens), self.prepro_out_size))
        lstm_output, (h, c) = self.lstm(
            pre_process_out, recurrent_state)
        
        
        dense_input = lstm_output[-1]
        
        dense1_out = relu(self.dense1(dense_input))
        
        prob_out = torch.sigmoid(self.prob_dense(dense1_out))
                
        return prob_out, (h, c)
        
     
def binary_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return torch.mean((y_true == torch.round(y_pred)).float())


def batch_generator(obs, act, iswin, batch_size,device):
    total_samples = obs.size()[1]
    random_i = random.sample(range(total_samples), total_samples)
    i = 0
    while len(random_i[i*batch_size : i*batch_size+batch_size]) != 0:
        
        
        
        batch_ind = torch.tensor(random_i[i*batch_size : i*batch_size+batch_size]).to(device=device)
        
        batch_obs = torch.index_select(obs, 1, batch_ind)
        batch_act = torch.index_select(act, 1, batch_ind)
        batch_iswin = torch.index_select(iswin, 0, batch_ind)
       
        
        i += 1
        
        
        yield (batch_obs, batch_act, batch_iswin)
        
        

def train_win_predictor(config,
                        model: LSTM_Predictor,
                        buffer: BattleBuffer,
                        writer
                        ):
  
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    
    log_dir = config.out_dir
    prediction_steps = config.prediction_steps
    epoch = config.win_predictor_epoch
    
    # exponential decay of learning rate
    learning_rate = config.win_predictor_lr_init * config.win_predictor_lr_decay_rate**(
                    model.training_step / config.win_predictor_lr_decay_steps
                    )
    optimizer = optim.Adam(params=model.parameters(), 
                           lr=learning_rate, weight_decay=1e-4)
    loss_func= nn.BCELoss()
    
    
    # extracting batches
    train_data, val_data = buffer.make_balanced_batch(device)
    train_obs, train_act, train_iswin = train_data
    val_obs, val_act, val_iswin = val_data
    
    
    prev_loss = np.inf
    
    statistics = {
        "train_loss_list": [],
        "train_accuracy_list": [],
        "ave_train_loss": None,
        "val_loss": None,
        "ave_train_accuracy": None,
        "val_accuracy": None
    }
    
    #for early stopping 
    patient = 3
    # training model
    for i in range(1, epoch+1):
        model.training_step += 1
        model = model.to(device)
        for batch in batch_generator(train_obs, train_act,
                                     train_iswin, config.buffer_batch_size, device):
            
            
            (
                train_obs, 
                train_act, 
                train_iswin
            ) = batch
            train_pred, _ = model(
                    (
                        train_obs,
                        train_act
                    ),
                    None
                )
            loss = loss_func(train_pred, train_iswin)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        
            
            #record metrics
            statistics["train_accuracy_list"].append(
                binary_accuracy(train_iswin, train_pred)
                )
            
            statistics["train_loss_list"].append(
                loss.item()
                ) 
           
        
        #val
        val_pred, _ = model(
                (
                    val_obs,
                    val_act
                ), 
                None
            )
            
        val_loss = loss_func( val_pred, val_iswin).item()
        val_accur = binary_accuracy(val_iswin, val_pred)
        
        statistics["val_accuracy"] = val_accur
        statistics["val_loss"] = val_loss
        
         
        ave_train_loss = sum(statistics["train_loss_list"])/len(statistics["train_loss_list"])
        
        ave_train_accur = sum(statistics["train_accuracy_list"])/len(statistics["train_accuracy_list"])
        
        statistics["ave_train_loss"] = ave_train_loss
        statistics["ave_train_accuracy"] = ave_train_accur
        
        
            
        # tensorboard
        step = model.training_step
        writer.add_scalar(
            'win_predictor/ave_train_loss', 
            statistics["ave_train_loss"], 
            step
        )
        writer.add_scalar(
            'win_predictor/val_loss', 
            statistics['val_loss'], step)
        writer.add_scalar(
            'win_predictor/ave_train_accuracy', 
            statistics["ave_train_accuracy"], step)
        writer.add_scalar(
            'win_predictor/val_accuracy', 
            statistics["val_accuracy"], step)
        
        statistics = {
        "train_loss_list": [],
        "train_accuracy_list": [],
        "ave_train_loss": None,
        "val_loss": None,
        "ave_train_accuracy": None,
        "val_accuracy": None
        }
        
        # eatly stopping 
        if val_loss >= prev_loss:
            patient -= 1
            if patient == 0:
                #print("early_stopping")
                break
        else:
            # saveing weight 
            torch.save(model.to('cpu').state_dict(), config.checkpoint_path)
        
        prev_loss = val_loss
        
        
    
    