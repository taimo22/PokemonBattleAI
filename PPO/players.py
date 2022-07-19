
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import LocalhostServerConfiguration
from poke_env.player.env_player import EnvPlayer, Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player.player import Player
from torch._C import device

from features import PokeDataBase
from model import MainModel

import os
import torch
import numpy as np

from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
import pfrl
from torch_win_rate import LSTM_Predictor, BattleBuffer, PokeBattle, OppoInfoBuffer


class base_env(Gen8EnvSinglePlayer):
    '''
    Use poke-env's EnvPlayer class to define a env
    thru which the Agent will interact with a Showdown Server.
    Must implement:
        embed_battle: defines what observation the agent will see
        computer_reward: defines what reward the agent will receive
    '''
    def __init__(self, 
                 unique_index, 
                 oracle_param,
                 replay_path
                 , *args, **kwargs
                 ): # the index is for preventing duplicated username
        Gen8EnvSinglePlayer.__init__(self, 
                                     player_configuration=PlayerConfiguration('Env'+unique_index, None), 
                                     
                                     )
    
        self.poke_database = PokeDataBase()
        self.statistics = None
        
        # for oracle guding 
        self.oppo_info_buffer: OppoInfoBuffer = None
        self.oracle_param = oracle_param
        self.start_oracle_param = self.oracle_param
        
        self.oracle_param_decay_step = 1000
        self.end_oracle_param = 0
        self.battle_num = 0

        
        self.replay_path = replay_path
        
    def oracle_param_update(self):
        if self.oracle_param > 0:
            diff = self.end_oracle_param - self.start_oracle_param
            self.oracle_param = self.start_oracle_param + diff * (
                self.battle_num / self.oracle_param_decay_step)
            self.oracle_param = max(self.oracle_param, 0)   

    def embed_battle(self, battle: Battle):
        
        (
            battle_info, 
            player_info, 
            oppo_info, 
            self.masked_actions
        ) = self.poke_database.make_feature(battle)
        
        
        # oracle guidging 
        if self.oracle_param > 0 and self.oppo_info_buffer.buffer:
            # reading opponent's info
            perfect_oppo_info = self.oppo_info_buffer.buffer[-1].copy()
            
            # identify what features are unobsevable
            # if in perfect obs not zero and in not perfect, zero,
            # this info is masked => unobserable 
            unobs_ind = set(np.where(perfect_oppo_info != 0)[0]) & set(np.where(oppo_info==0)[0])
            
            # oracle guiding
            dropout_matrix = np.random.binomial(size=len(unobs_ind), n=1, p=self.oracle_param)
            perfect_oppo_info[list(unobs_ind)] *= dropout_matrix
        
            oppo_info = perfect_oppo_info
            
        features = np.concatenate([
            self.masked_actions, 
            battle_info,
            player_info,
            oppo_info
            ])
        
    
        
        if battle.finished: 
            # updating orcle guiding param 
            if self.oppo_info_buffer:
                self.battle_num +=1
                self.oracle_param_update()
                self.oppo_info_buffer.reset()
                
            if self.statistics:
               self.statistics.cum_num_battle += 1
               self.statistics.cum_num_win_battle += int(battle.won)

          
            
            
        if self.statistics:
            self.statistics.curr_step += 1
        
        return torch.Tensor(features).to(torch.float32)
    
    
        
    def compute_reward(self, battle) -> float:
        '''
        ideally the agent should learn with only victory rewards
        however in reality, if the agent is not learning, 
        try turn up faint_value & hp_value
        '''
    
        return self.reward_computing_helper(
            battle, fainted_value=1.0, hp_value=0.7, victory_value=10
        )
        
    def simple_reward_compute(self, battle: Battle, victory_value):
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = 0
        current_value = 0

        # win or lose
        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value
        
        self._reward_buffer[battle] = current_value

        return current_value
    
    def reward_computing_helper(
        self,
        battle: Battle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
        take_diff: bool = False
    ) -> float:
        """A helper function to compute rewards.

        The reward is computed by computing the value of a game state, and by comparing
        it to the last state.

        State values are computed by weighting different factor. Fainted pokemons,
        their remaining HP, inflicted statuses and winning are taken into account.

        For instance, if the last time this function was called for battle A it had
        a state value of 8 and this call leads to a value of 9, the returned reward will
        be 9 - 8 = 1.

        Consider a single battle where each player has 6 pokemons. No opponent pokemon
        has fainted, but our team has one fainted pokemon. Three opposing pokemons are
        burned. We have one pokemon missing half of its HP, and our fainted pokemon has
        no HP left.

        The value of this state will be:

        - With fainted value: 1, status value: 0.5, hp value: 1:
            = - 1 (fainted) + 3 * 0.5 (status) - 1.5 (our hp) = -1
        - With fainted value: 3, status value: 0, hp value: 1:
            = - 3 + 3 * 0 - 1.5 = -4.5

        :param battle: The battle for which to compute rewards.
        :type battle: AbstractBattle
        :param fainted_value: The reward weight for fainted pokemons. Defaults to 0.
        :type fainted_value: float
        :param hp_value: The reward weight for hp per pokemon. Defaults to 0.
        :type hp_value: float
        :param number_of_pokemons: The number of pokemons per team. Defaults to 6.
        :type number_of_pokemons: int
        :param starting_value: The default reference value evaluation. Defaults to 0.
        :type starting_value: float
        :param status_value: The reward value per non-fainted status. Defaults to 0.
        :type status_value: float
        :param victory_value: The reward value for winning. Defaults to 1.
        :type victory_value: float
        :return: The reward.
        :rtype: float
        """
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0
        
        active_value = 1.0
        # Player
        for mon in battle.team.values():
            if mon.active:
                current_value += mon.current_hp_fraction * hp_value * active_value
            else:
                current_value += mon.current_hp_fraction * hp_value
                
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value
            

        current_value += (number_of_pokemons - len(battle.team)) * hp_value
        
        # Opponent
        for mon in battle.opponent_team.values():
            
            if mon.active:
                current_value -= mon.current_hp_fraction * hp_value * active_value
            else:
                current_value -= mon.current_hp_fraction * hp_value
            
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value
         
            
        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value
        
        #type
        type_value = 0.4
        oppo_actpoke = battle.opponent_active_pokemon
        player_actpoke = battle.active_pokemon
        #type
        if oppo_actpoke and player_actpoke:
            # type: when player attack opponent
            type_reward1 = player_actpoke.type_1.damage_multiplier(
                oppo_actpoke.type_1, oppo_actpoke.type_2
                ) 
            
            if player_actpoke.type_2:
                type_reward1 += player_actpoke.type_2.damage_multiplier(
                oppo_actpoke.type_1, oppo_actpoke.type_2
                ) 
          
            type_reward1 *= type_value
            current_value += type_reward1

            # type: when opponent attack player
            type_reward2 = oppo_actpoke.type_1.damage_multiplier(
                player_actpoke.type_1, player_actpoke.type_2
                ) 
            if oppo_actpoke.type_2:
                type_reward2 += oppo_actpoke.type_2.damage_multiplier(
                player_actpoke.type_1, player_actpoke.type_2
                )
            type_reward2 *= type_value
            current_value -= type_reward2
        '''
            # move type combination
            move_type_reward1 = 0
            for move in player_actpoke.moves.values():
                move_type_reward1 += move.type.damage_multiplier(
                 oppo_actpoke.type_1, oppo_actpoke.type_2
                ) 
            
                if player_actpoke.type_2:
                    move_type_reward1 += move.type.damage_multiplier(
                    oppo_actpoke.type_1, oppo_actpoke.type_2
                    ) 
            move_type_reward1 *= type_value
            current_value += move_type_reward1

            # move type combination
            move_type_reward2 = 0
            for move in oppo_actpoke.moves.values():
                move_type_reward2 += move.type.damage_multiplier(
                 player_actpoke.type_1, player_actpoke.type_2
                ) 
            
                if player_actpoke.type_2:
                    move_type_reward2 += move.type.damage_multiplier(
                    player_actpoke.type_1, player_actpoke.type_2
                    ) 
            move_type_reward2 += (4 - len(oppo_actpoke.moves.values()))
            move_type_reward2 *= type_value
            current_value -= move_type_reward2
            
            
        '''  
        
        
        
        
        # win or lose
        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value
        
        
        # take the difference between reward and the pastest reward
        if take_diff:
            to_return = current_value - self._reward_buffer[battle]
        else:
            to_return = current_value
        self._reward_buffer[battle] = current_value

        return to_return
    
    def _action_to_move(
        self, action:int, battle:Battle)->str:
        """Converts actions to move orders.

        The conversion is done as follows:

        0 <= action < 4:
            The actionth available move in battle.available_moves is executed.
        
        4 <= action < 8:
            The action - 4th available move in battle.available_moves is executed,
            while dynamaxing.
        8 <= action < 14
            The action - 8th available switch in battle.available_switches is executed.

        If the proposed action is illegal, a random legal move is performed.

        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        
        action = action.item()
        
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.active_pokemon.moves)
            and not battle.force_switch
        ):
            
            move = list(battle.active_pokemon.moves.values())[action]
            available_moves_names = [m.id for m in battle.available_moves]
            if move.id in available_moves_names: 
                return self.create_order(move)
            else:
                return self.choose_random_move(battle)
        elif (
            battle.can_dynamax
            and 0 <= action - 4 < len(battle.active_pokemon.moves)
            and not battle.force_switch
        ):
            move = list(battle.active_pokemon.moves.values())[action - 4]
            available_moves_names = [m.id for m in battle.available_moves]
            if move.id in available_moves_names: 
                return self.create_order(move, dynamax=True)
            else:
                return self.create_order(battle.available_moves[0], dynamax=True)
        
            
        elif 0 <= action - 8 < len(battle.team.values()):
            
            reserved_pokes = [poke for poke in battle.team.values() if poke.active] + [poke  for poke in battle.team.values() if not poke.active]
            change_to_poke = reserved_pokes[action - 8]
            
            if change_to_poke in battle.available_switches:
                return self.create_order(change_to_poke)
            else:
                return self.choose_random_move(battle)
        else:
            return self.choose_random_move(battle)

class dda_env(base_env):
    def __init__(self, 
                 unique_index, oracle_param, replay_path,
                 
                  *args, **kwargs):
        super().__init__(
            unique_index,
            oracle_param,
            replay_path,
            player_configuration=PlayerConfiguration('DDA_Env'+unique_index, None), 
            #log_level="INFO",
            server_configuration=LocalhostServerConfiguration)
        
        
        # for dda
        self.alpha = 0
        self.reward_for_fighting = 0
        self.reward_for_dda = 0
        self.reward = 0
        self.obs = None
        self.action = None
        self.prev_action = 0
        
        # for predicting win
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
        
    
    def embed_battle(self, battle: Battle):
        obs = super().embed_battle(battle)    

        return obs
    
    def choose_move(self, battle: Battle)->str:
        
        if battle not in self._observations or battle not in self._actions:
            self._init_battle(battle)
            
        obs = self.embed_battle(battle)
        self._observations[battle].put(obs)
        action = self._actions[battle].get()
        
        taken_action = self._action_to_move(action, battle)
    
        return taken_action
    
    def compute_reward(self, battle: Battle) -> float:
        
        reward = super().compute_reward(battle)
        
        return reward
    
    def calc_reward_for_dda_by_winprob(self, obs)->float:
        
        assert self.win_predictor is not None
        
        
        if self.poke_battle.length() > 0:
            self.prev_action_onehot = self.poke_battle.make_action(-1)
        
        inputs = (
            torch.from_numpy(obs).float().to(device=self.device), 
            torch.reshape(self.prev_action_onehot, (1, 1, 14)).to(device=self.device)
            )
        if self.prev_hstate is not None:
            self.prev_hstate[0] = self.prev_hstate[0].to(self.device)
            self.prev_hstate[1] = self.prev_hstate[1].to(self.device)
            
        pred_result, h_state = self.win_predictor(
            inputs,
            self.prev_hstate
            )
        pred_result = pred_result.detach().to(device("cpu")).numpy().item()
        self.prev_hstate = h_state
        
        
        balanced_win_prob = 0.5
        # The closer the predicted win rate is to 0.5, the higher the reward should be.
        reward_for_dda = 0.5 - (
            np.abs(pred_result - balanced_win_prob)
            ) 
        #self.predicted_results.append(pred_result)
        
        self.reward_for_dda = reward_for_dda
        return reward_for_dda
    

class OppoPlayer(Player):
    def __init__(self, unique_index, agent):
        Player.__init__(self, player_configuration=PlayerConfiguration('Oppo'+unique_index, None))

        
        self.poke_database = PokeDataBase()
        features_dim_dict = self.poke_database.get_features_dims()
        total_features_dim = features_dim_dict["total_features_dim"]
        
        battle_obs_dim = features_dim_dict["battle_obs_dim"]
        oppo_obs_dim = features_dim_dict["oppo_obs_dim"]
        player_obs_dim = features_dim_dict["player_obs_dim"]
        total_features_dim = features_dim_dict["total_features_dim"]
            
        poke_obs_dim = features_dim_dict["poke_obs_dim"]
        moves_obs_dim  = features_dim_dict["moves_obs_dim"]
        ability_obs_dim = features_dim_dict["ability_obs_dim"]
        item_obs_dim = features_dim_dict["item_obs_dim"]
        others_obs_dim = features_dim_dict["others_obs_dim"]
        total_poke_dim = features_dim_dict["total_poke_dim"]
        
        
        self.model = MainModel(
            total_obs_dim=total_features_dim,
            battle_obs_dim=battle_obs_dim,
            player_obs_dim=player_obs_dim,
            oppo_obs_dim=oppo_obs_dim,
            moves_obs_dim=moves_obs_dim,
            total_poke_obs_dim=total_poke_dim,
            others_obs_dim=others_obs_dim
        )
        
        self.agent: pfrl.agents.PPO = agent
        self.oppo_info_buffer: OppoInfoBuffer = None
   
        
    def embed_battle(self, battle: Battle):
        
        (
            battle_info, 
            player_info, 
            oppo_info, 
            self.masked_actions
        ) = self.poke_database.make_feature(battle)
        
        
        features = np.concatenate([
            self.masked_actions, 
            battle_info,
            player_info,
            oppo_info
            ])
        
        if self.oppo_info_buffer:
            self.oppo_info_buffer.buffer.append(player_info)
            self.oppo_info_buffer.turns.append(battle.turn)
        
        return torch.Tensor(features).to(torch.float32)
    
    def choose_move(self, battle):
        
        # getting the observation from battle object 
        obs = self.embed_battle(battle)
        
        # deciding an action based on the observation 
        action = self.agent.act(obs)

        return self._action_to_move(action, battle)
    
    def _action_to_move(
        self, action:int, battle:Battle)->str:
        """Converts actions to move orders.

        The conversion is done as follows:

        0 <= action < 4:
            The actionth available move in battle.available_moves is executed.
        
        4 <= action < 8:
            The action - 4th available move in battle.available_moves is executed,
            while dynamaxing.
        8 <= action < 14
            The action - 8th available switch in battle.available_switches is executed.

        If the proposed action is illegal, a random legal move is performed.

        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        
        action = action.item()
        
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.active_pokemon.moves)
            and not battle.force_switch
        ):
            
            move = list(battle.active_pokemon.moves.values())[action]
            available_moves_names = [m.id for m in battle.available_moves]
            if move.id in available_moves_names: 
                return self.create_order(move)
            else:
                return self.choose_random_move(battle)
        elif (
            battle.can_dynamax
            and 0 <= action - 4 < len(battle.active_pokemon.moves)
            and not battle.force_switch
        ):
            move = list(battle.active_pokemon.moves.values())[action - 4]
            available_moves_names = [m.id for m in battle.available_moves]
            if move.id in available_moves_names: 
                return self.create_order(move, dynamax=True)
            else:
                return self.create_order(battle.available_moves[0], dynamax=True)
        
            
        elif 0 <= action - 8 < len(battle.team.values()):
            
            reserved_pokes = [poke for poke in battle.team.values() if poke.active] + [poke  for poke in battle.team.values() if not poke.active]
            change_to_poke = reserved_pokes[action - 8]
            
            if change_to_poke in battle.available_switches:
                return self.create_order(change_to_poke)
            else:
                return self.choose_random_move(battle)
        else:
            return self.choose_random_move(battle)
