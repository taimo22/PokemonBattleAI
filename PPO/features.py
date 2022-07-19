
from collections import defaultdict
from dataclasses import field, fields
from logging import NOTSET
from os import stat
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.weather import Weather
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.data import MOVES, GEN_TO_MOVES
from poke_env.environment.field import Field
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.status import Status
from poke_env.environment.weather import Weather


import numpy as np
import re
import pandas as pd
import copy
import gym_notices
from poke_env.environment.effect import Effect
from numpy import ravel, concatenate
import torch

from typing import Any, Callable, List, Dict, Optional, Tuple, Union
import time
from cProfile import Profile
import os



class PokeDataBase:
    
    def __init__(self):
        
        # read the feature data
        
        base_path = "PPO\\master_data_process\\reduced_data"

        move_data_path = os.path.join(base_path,  "move_reduced_data.csv")
        poke_data_path = os.path.join(base_path,  "poke_reduced_data.csv")
        ability_data_path = os.path.join(base_path,  "ability_reduced_data.csv")
        item_data_path = os.path.join(base_path,  "item_reduced_data.csv")

        assert(os.path.exists(move_data_path))
        assert(os.path.exists(poke_data_path))
        assert(os.path.exists(ability_data_path))
        assert(os.path.exists(item_data_path))

        self.data_move_dict, self.move_features, self.move_names = self.df_to_dict(move_data_path)
        self.data_poke_dict, self.poke_features, self.poke_names = self.df_to_dict(poke_data_path)
        self.data_ability_dict, self.ability_features, self.ability_names = self.df_to_dict(ability_data_path)
        self.data_item_dict, self.item_features, self.item_names = self.df_to_dict(item_data_path)
     

        # dimension of feature data
        self.moves_obs_dim: int = (len(self.move_features)+1+1)*4
        
        self.poke_obs_dim: int = len(self.poke_features)+1+1+6+7+6+1+1
        
        self.ability_obs_dim: int = len(self.ability_features)
        self.item_obs_dim: int = len(self.item_features)
        self.dmaxmoves_obs_dim: int = 23*4
        self.others_obs_dim: int = 1
        self.total_poke_dim: int = self.poke_obs_dim+self.moves_obs_dim+self.item_obs_dim+self.ability_obs_dim+self.dmaxmoves_obs_dim
        
        self.regex = re.compile('[^a-zA-Z0-9]')
        
        
       
        self.battle_obs_dim: int = 19
        
       
        self.oppo_obs_dim: int = 6*(self.total_poke_dim)+self.others_obs_dim
        
        self.player_obs_dim: int = 6*(self.total_poke_dim)+self.others_obs_dim
        
        
        self.total_features_dim = self.battle_obs_dim+self.oppo_obs_dim+self.player_obs_dim
        
        self.effect_dict_master = self.make_effect_dict()
        
        # for time inspect
        self.time_inspect = False
        self.time_dict = defaultdict(list)
        self.time_list = []
        self.total_time = 0

    def make_effect_dict(self)->Dict[str, int]:
        effect_dict_master = {}
        for ef_i in range(1, len(Effect)+1):
            effect_dict_master[Effect(ef_i).name] = 0
            
        return effect_dict_master

    def df_to_dict(self, data_path: str)->Tuple[List[str], Dict[str, float]]:


        df_data: pd.DataFrame = pd.read_csv(data_path).set_index("name")
        features: List[str] = [col for col in df_data.columns if col != "name"]
        dict_data = df_data.to_dict(orient='index')
        for key in dict_data:
            column_data = dict_data[key]
            dict_data[key] = np.array(list(column_data.values()))

        return dict_data, features, df_data.index
        
    
    def get_features_dims(self)->Dict[str, int]:
        feature_num_dict = {
            "battle_obs_dim":self.battle_obs_dim,
            "oppo_obs_dim":self.oppo_obs_dim,
            "player_obs_dim":self.player_obs_dim,
            "total_features_dim": self.total_features_dim,
        
            "poke_obs_dim" : self.poke_obs_dim,
            "moves_obs_dim" : self.moves_obs_dim,
            "ability_obs_dim" : self.ability_obs_dim,
            "item_obs_dim" : self.item_obs_dim,
            "others_obs_dim": self.others_obs_dim,
            "total_poke_dim": self.total_poke_dim
        }
        return feature_num_dict
    
    def __repr__(self) -> str:
        feature_num_dict = {
            "battle_obs_dim":self.battle_obs_dim,
            "oppo_obs_dim":self.oppo_obs_dim,
            "player_obs_dim":self.player_obs_dim,
            "total_features_dim": self.total_features_dim,
            
            "poke_obs_dim" : self.poke_obs_dim,
            "moves_obs_dim" : self.moves_obs_dim,
            "ability_obs_dim" : self.ability_obs_dim,
            "item_obs_dim" : self.item_obs_dim,
            "others_obs_dim": self.others_obs_dim,
            "total_poke_dim": self.total_poke_dim
        }
        
        returned_str = ""
        for k, v in feature_num_dict.items():
            returned_str+=f"{k} : {v}, \n"
        
        return returned_str
    
    def make_feature(self, battle: Battle):
        if self.time_inspect:
            start = time.time()
        #extracting state info related to battle
        battle_info = self.extract_battle_info(battle)
        
        #extracting player team inf
        player_info, oppo_info = self.extract_players_info(battle)
       
        
        # updating masked actions 
        self.update_masked_action(battle)
        
        # for time inspection
        if self.time_inspect:
            self.time_list.append(time.time()-start)
            if battle.finished:
                
                sum_time = sum(self.time_list)
                self.total_time += sum_time
                print()
                print(sum_time/len(self.time_list))
                print(sum_time)
                print(self.total_time)
                self.time_list = []
                
        
        
        return (
            battle_info, 
            player_info, 
            oppo_info, 
            self.masked_actions
            )
    
    
    
    def update_masked_action(self, battle: Battle): # making no avaliable actions
        if battle.won == None:
            # move 
            
            if len(battle.available_moves) > 0: # if any moves are available
                
                if len(battle.available_moves) and battle.available_moves[0].id == "struggle":
                    all_moves = [1, 0, 0, 0]
                    all_dmax_moves = [1, 0, 0, 0]
                elif battle.active_pokemon.is_dynamaxed:
                    all_moves = [0]*4
                    all_dmax_moves = [1]*4
                else:
                    # initialization
                    move_poke_have = list(battle.active_pokemon.moves.keys())
                    all_moves_dict = {}
                    for i in range(4):
                        if i < len(move_poke_have):
                            all_moves_dict[move_poke_have[i]] = 0
                        else:
                            all_moves_dict[f"None_{i}"] = 0
                    
                    # mapping from the names of the moves to unique ids
                    move_to_index = {move_name : i
                                    for i, move_name in enumerate(all_moves_dict.keys())}
                    all_dmax_moves = np.array([0, 0, 0, 0])
                    
                    # 
                    for move in battle.available_moves:
                        if move.id not in all_moves_dict.keys():
                            all_moves_dict = {move_name: 1
                                for move_name in all_moves_dict.keys()}
                            break
                        all_moves_dict[move.id] = 1
                        all_dmax_moves[move_to_index[move.id]] = 1
                    all_moves = list(all_moves_dict.values())
            else: # no moves are avalilable
                all_dmax_moves = [0]*4
                all_moves = [0]*4
            
            # masking for dmaxbattle
            if (not battle.can_dynamax) and (not battle.active_pokemon.is_dynamaxed):
                all_dmax_moves = [0]*4
            
            
            # masking for changing pokemon
            all_available_switches_dict = {p.species : 0  for p in battle.team.values()}
            for p in battle.available_switches:
                # for debug
                if p.species not in all_available_switches_dict.keys():
                    import ipdb; ipdb.set_trace()
                all_available_switches_dict[p.species] = 1
                
            all_available_switches = list(all_available_switches_dict.values())
            # concat to make one list for masking
            
            if len(all_moves) != 4:
                import ipdb; ipdb.set_trace()
            assert len(all_dmax_moves) == 4
            assert len(all_available_switches) == 6
            
            self.masked_actions = np.concatenate([
                                                all_moves,
                                                all_dmax_moves,
                                                all_available_switches
                                                ])
        else: # if the battle ends, there is no need to do masking
            self.masked_actions = np.zeros(14)
        
    
    
    def oppo_loop_wrapper(self, 
                          i: int, 
                          oppo_team: List[Pokemon],
                          battle: Battle
                          ):
        
        if i < len(oppo_team):
            oppo_poke = oppo_team[i]
            
            
            # poke info(observable)
            oppo_poke_info = self.data_poke_dict[oppo_poke._species]

            
            # hp fraction = 1
            hp_frac = oppo_poke.current_hp_fraction
            
            
            # status = 6
            status_dict = {
                "BRN" : 0,
                "FRZ" : 0, 
                "PAR" : 0,
                "PSN" : 0,
                "SLP" : 0,
                "TOX" : 0,
            }
            if (oppo_poke.status != None) and (oppo_poke.status.name != "FNT"): 
                status_dict[oppo_poke.status.name] = 1
            status_count = oppo_poke.status_counter/5
            
            # effect = 163
            '''
            effect_dict = self.effect_dict_master.copy()
                
            for effe, count in oppo_poke.effects.items():
                effect_dict[effe.name] = 1
            
            '''
                
            # boost = 7
            boosts_dict = {
                'accuracy': 0, 
                'atk': 0, 'def': 0, 
                'evasion': 0, 'spa': 0, 'spd': 0, 'spe': 0}
            for k, v in oppo_poke.boosts.items():
                boosts_dict[k] = v
            
            sum_boost = sum(list(boosts_dict.values()))
            if sum_boost != 0:
                for key, value in boosts_dict.items():
                    boosts_dict[key] = value/sum_boost
        
            # stats = 6 
            # because the stats of the opponent's pokemon is not observable  
            
            stats =  {"hp": 0, 
                    "atk": 0, 
                    "def": 0, 
                    "spa": 0, 
                    "spd": 0, 
                    "spe": 0
                    }
            
            
            # faint = 1
            p_is_alive = [not(oppo_poke.fainted)]
            
            # active = 1
            p_is_active = [oppo_poke.active]
            
            
            oppo_poke_info = concatenate([
                                        oppo_poke_info, 
                                        [hp_frac],
                                        list(status_dict.values()),
                                        [status_count],
                                        #list(effect_dict.values()),
                                        list(boosts_dict.values()),
                                        list(stats.values()),
                                        p_is_alive,
                                        p_is_active
                                        ])
            
         
            
            
            # info about ability(unobservable)
            if oppo_poke.ability:
                oppo_ability_info = self.data_ability_dict[oppo_poke.ability]
            else:
                oppo_ability_info = np.zeros(self.ability_obs_dim,
                                          dtype='float32')
            
            # info about items (unobservable)
            if oppo_poke.item and oppo_poke.item in self.item_names:
                oppo_item_info = self.data_item_dict[oppo_poke.item]
            else:
                oppo_item_info = np.zeros(self.item_obs_dim,
                                          dtype='float32')
            
            # info about move(unobservable)     
            
            oppo_moves_info = np.zeros(self.moves_obs_dim,
                                          dtype='float32')
            num_one_move_feature = self.moves_obs_dim//4
            oppo_dmaxmoves_info = np.zeros(self.dmaxmoves_obs_dim,
                                          dtype='float32')
            num_one_dmaxmove_feature = self.dmaxmoves_obs_dim//4
            
            for i, move in enumerate(oppo_poke.moves.values()):
                
                move_info = copy.deepcopy(list(self.data_move_dict[move.id]))
                '''
                
                
                '''
                pp_frac = move.current_pp/move.max_pp
                
                
                #moves_dmg_multiplier
                move_dmg_multiplier = 0
                if move.type:
                    move_dmg_multiplier = move.type.damage_multiplier(
                        battle.active_pokemon.type_1,
                        battle.active_pokemon.type_2,
                    )
               
                move_info.extend([move_dmg_multiplier, pp_frac])
                
                
                oppo_moves_info[num_one_move_feature*i : (num_one_move_feature*i)+num_one_move_feature ] = move_info
                
                
                # info about dmax moves (unobservable)
                weather_features_dict = {
                "HAIL": 0,
                "RAINDANCE": 0,
                "SANDSTORM":0 ,
                "SUNNYDAY": 0
                }
                field_features_dict = {
                    "ELECTRIC_TERRAIN": 0,
                    "GRASSY_TERRAIN": 0,
                    "MISTY_TERRAIN": 0,
                    "PSYCHIC_TERRAIN": 0,
                }
                dmax_boosts_dict = {
                    "atk": 0,
                    "def": 0,
                    "spa": 0,
                    "spd": 0,
                    "spe": 0
                }
                dmax_self_boosts_dict = dmax_boosts_dict.copy()
                category_dict = {
                    "STATUS": 0,
                    "PHYSICAL": 0,
                    "SPECIAL": 0
                }
                
                base_power = move.dynamaxed.base_power/200
                
                is_protect_move = move.dynamaxed.is_protect_move
                category_dict[move.dynamaxed.category.name] = 1
                
                if move.dynamaxed.boosts is not None:
                    dmax_boosts_dict.update(move.dynamaxed.boosts)
                if move.dynamaxed.self_boost is not None:
                    dmax_self_boosts_dict.update(move.dynamaxed.self_boost)
                
                if move.dynamaxed.terrain is not None:
                    field_features_dict[move.dynamaxed.terrain.name] = 1
                if move.dynamaxed.weather is not None:
                    weather_features_dict[move.dynamaxed.weather.name] = 1
                dmax_move_info = []
                dmax_move_info.append(base_power)
                dmax_move_info.append(is_protect_move)
                dmax_move_info.extend(list(category_dict.values()))
                dmax_move_info.extend(list(dmax_boosts_dict.values()))
                dmax_move_info.extend(list(dmax_self_boosts_dict.values()))
                dmax_move_info.extend(list(field_features_dict.values()))
                dmax_move_info.extend(list(weather_features_dict.values()))
                
              
                
                oppo_dmaxmoves_info[num_one_dmaxmove_feature*i : 
                    (num_one_dmaxmove_feature*i)+num_one_dmaxmove_feature ] = dmax_move_info
           
            
                
            
            # concat the all info 
            oppo_total_poke_info = concatenate(
                [
                oppo_poke_info, 
                oppo_ability_info,
                oppo_item_info,
                oppo_moves_info,
                oppo_dmaxmoves_info
                ])
            
            
        else:
            oppo_total_poke_info = np.zeros(self.total_poke_dim,dtype='float32')
            
            
        return oppo_total_poke_info
    
    def player_loop_wrapper(self, 
                          p: Pokemon,
                          battle: Battle
                          ):
       
        if p._species:
            
            poke_info = self.data_poke_dict[p._species]
            
            
            # hp fraction = 1
            hp_frac = p.current_hp_fraction
            
            # status = 6
            status_dict = {
                "BRN" : 0,
                "FRZ" : 0, 
                "PAR" : 0,
                "PSN" : 0,
                "SLP" : 0,
                "TOX" : 0,
            }
            if (p.status != None) and (p.status.name != "FNT"): 
                status_dict[p.status.name] = 1
            status_count = p.status_counter/5
            
            
            # effect = 163
            '''
            effect_dict = self.effect_dict_master.copy()
            for effe, count in p.effects.items():
                effect_dict[effe.name] = 1
            '''
            
                
            
            # boost = 7
            boosts_dict = {'accuracy': 0, 
                            'atk': 0, 
                            'def': 0, 
                            'evasion': 0, 'spa': 0, 'spd': 0, 'spe': 0}
            boosts_dict.update(p.boosts.copy())
            for k, v in p.boosts.items():
                boosts_dict[k] = v
            sum_boost = sum(list(boosts_dict.values()))
            if sum_boost != 0:
                for key, value in boosts_dict.items():
                    boosts_dict[key] = value/sum_boost
            


            
            
            # stats = 6
            stats = p.stats.copy()
            sum_of_stats = sum(p.stats.values())+p.max_hp
            stats["hp"] = p.max_hp
            for k, v in stats.items():
                stats[k] = v/sum_of_stats
              
            # faint = 1
            p_is_alive = [not(p.fainted)]
            
            # active = 1
            p_is_active = [p.active]
            
            
            poke_info = concatenate([poke_info, 
                                        [hp_frac],
                                        list(status_dict.values()),
                                        [status_count],
                                        #list(effect_dict.values()),
                                        list(boosts_dict.values()),
                                        list(stats.values()),
                                        p_is_alive,
                                        p_is_active
                                        ]
                                    )  
           
        
        #ability
        if p.ability:
            
            ability_info = self.data_ability_dict[
                p.ability]
            
        
        # item 
        if p.item and p.item in self.item_names:
            
            item_info = self.data_item_dict[p.item]
            
        else:
            item_info = np.zeros(
                self.item_obs_dim,
                dtype='float32'
                )
        
        #move and dmax_movs   
        moves_info = np.zeros(self.moves_obs_dim,
                                          dtype='float32')
        num_one_move_feature = self.moves_obs_dim//4
        dmax_moves_info = np.zeros(self.dmaxmoves_obs_dim,dtype='float32')
        num_one_dmaxmove_feature = self.dmaxmoves_obs_dim//4
        for i, move in enumerate(p.moves.values()):
            
            move_info = copy.deepcopy(list(self.data_move_dict[move.id]))
            '''
            
            '''
            pp_frac = move.current_pp/move.max_pp
            #moves_dmg_multiplier
            move_dmg_multiplier = 0
            if move.type:
                move_dmg_multiplier = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )
                
            move_info.extend([move_dmg_multiplier, pp_frac])
            moves_info[num_one_move_feature*i : (num_one_move_feature*i)+num_one_move_feature ] = move_info
            
            # dmaxmove
            weather_features_dict = {
                "HAIL": 0,
                "RAINDANCE": 0,
                "SANDSTORM":0 ,
                "SUNNYDAY": 0
            }
            field_features_dict = {
                "ELECTRIC_TERRAIN": 0,
                "GRASSY_TERRAIN": 0,
                "MISTY_TERRAIN": 0,
                "PSYCHIC_TERRAIN": 0,
            }
            dmax_boosts_dict = {
                "atk": 0,
                "def": 0,
                "spa": 0,
                "spd": 0,
                "spe": 0
            }
            dmax_self_boosts_dict = dmax_boosts_dict.copy()
            category_dict = {
                "STATUS": 0,
                "PHYSICAL": 0,
                "SPECIAL": 0
            }
            
            
            base_power = move.dynamaxed.base_power/200
            
            is_protect_move = move.dynamaxed.is_protect_move
            category_dict[move.dynamaxed.category.name] = 1
            
            if move.dynamaxed.boosts is not None:
                dmax_boosts_dict.update(move.dynamaxed.boosts)
            if move.dynamaxed.self_boost is not None:
                dmax_self_boosts_dict.update(move.dynamaxed.self_boost)
            
            if move.dynamaxed.terrain is not None:
                field_features_dict[move.dynamaxed.terrain.name] = 1
            if move.dynamaxed.weather is not None:
                weather_features_dict[move.dynamaxed.weather.name] = 1
            
            dmax_move_info = []
            dmax_move_info.append(base_power)
            dmax_move_info.append(is_protect_move)
            dmax_move_info.extend(list(category_dict.values()))
            dmax_move_info.extend(list(dmax_boosts_dict.values()))
            dmax_move_info.extend(list(dmax_self_boosts_dict.values()))
            dmax_move_info.extend(list(field_features_dict.values()))
            dmax_move_info.extend(list(weather_features_dict.values()))
        
            dmax_moves_info[num_one_dmaxmove_feature*i : (num_one_dmaxmove_feature*i)+num_one_dmaxmove_feature ] = dmax_move_info
        
        # concat 
        player_total_poke_info = concatenate([
            poke_info, 
            ability_info, 
            item_info, 
            moves_info, 
            dmax_moves_info
            ], axis=0)
        
        
        
        return player_total_poke_info
    
    def extract_players_info(self, battle: Battle)->np.ndarray:
        
        
        oppo_pokes_info_list = []
        
        oppo_team = [
                poke
                for poke in battle.opponent_team.values()
                if poke.active] + [poke 
                                for poke in battle.opponent_team.values()
                                if not poke.active
                                ]
        
        player_pokes_info_list = []
        player_team = [
                poke
                for poke in battle.team.values()
                if poke.active] + [poke 
                                for poke in battle.team.values() 
                                if not poke.active
                                ]
       
        
        for i in [0, 1, 2, 3, 4, 5]:
            # oppo
            
            oppo_pokes_info = self.oppo_loop_wrapper(
                i, oppo_team, battle
            )
            
            oppo_pokes_info_list.append(oppo_pokes_info)
            
            # player
            
            player_pokes_info = self.player_loop_wrapper(
                player_team[i], battle
            )
            player_pokes_info_list.append(player_pokes_info)
            
            
        # count of the opponent's remaining pokemons
        remaining_mon_opponent = (
            (6 - len([mon for mon in oppo_team if mon.fainted])) / 6
        )
        oppo_pokes_info_list.append([remaining_mon_opponent])
        oppo_info = concatenate(
            oppo_pokes_info_list
        )
        
        # We count how many pokemons have not fainted in my team
        remaining_mon_team = (
            (6 - len([mon for mon in player_team if mon.fainted])) / 6
        )
        player_pokes_info_list.append([remaining_mon_team])
        player_info = concatenate(
            player_pokes_info_list
        )
        
        
        return player_info, oppo_info
        
   
    
    
    def extract_battle_info(self, battle: Battle):
        
        '''
        Field:
            ELECTRIC_TERRAIN
            GRASSY_TERRAIN
            GRAVITY
            HEAL_BLOCK
            MAGIC_ROOM
            MISTY_TERRAIN
            MUD_SPORT
            MUD_SPOT
            PSYCHIC_TERRAIN
            TRICK_ROOM
            WATER_SPORT
            WONDER_ROOM
        '''
        
        field_features_dict = {
            "ELECTRIC_TERRAIN": 0,
            "GRASSY_TERRAIN": 0,
            "GRAVITY": 0,
            "HEAL_BLOCK": 0,
            "MAGIC_ROOM": 0,
            "MISTY_TERRAIN": 0,
            "MUD_SPORT": 0,
            "MUD_SPOT": 0,
            "PSYCHIC_TERRAIN": 0,
            "TRICK_ROOM": 0,
            "WATER_SPORT": 0,
            "WONDER_ROOM": 0,
        }
        
        if len(battle.fields):
            for field, turn in battle.fields.items():
                field_features_dict[field.name] = 1
        
        field_features = list(field_features_dict.values())
        
        
        '''
        Weather:
            DESOLATELAND 
            DELTASTREAM 
            HAIL 
            PRIMORDIALSEA 
            RAINDANCE 
            SANDSTORM
            SUNNYDAY
        '''
        
        weather_features_dict = {
            "DESOLATELAND": 0,
            "DELTASTREAM": 0,
            "HAIL": 0,
            "PRIMORDIALSEA": 0,
            "RAINDANCE": 0,
            "SANDSTORM":0 ,
            "SUNNYDAY": 0
        }
        if len(battle.weather):
            for weather, turn in battle.weather.items():
                weather_features_dict[weather.name] = 1
        
        weather_features = list(weather_features_dict.values())
        
        # trapped?
        features = np.concatenate(
                [ 
                   field_features, weather_features
                ]
            )
        
        return features 
    
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
        
        
        
        return (
            player_input_data, 
            oppo_input_data, 
            battle_info, 
            moves_info_post, 
            masked_actions, 
            player_pokes_info
            )