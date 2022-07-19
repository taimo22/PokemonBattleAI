import orjson
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import re
from typing import Any, Dict, Union

# TODO: make the feature which explains weakness of types
if __name__ == "__main__":    
    master_data_poke = pd.read_json(
        r'src\utils_dire\master_data_process\master_data\pokedex.json')
    master_data_poke = master_data_poke.T.reset_index()
    
    
    
    master_data_poke_name = master_data_poke["name"].copy()
    master_data_poke = master_data_poke.drop(["index", "num", 
                                              "abilities", 'gender', 
                                              'gen', 'evoLevel',
                                              'evoItem', 'prevo',
                                              'genderRatio', 'evoType',
                                              'formeOrder', 'canHatch',
                                              'requiredAbility', 'requiredItems',
                                              'requiredItem', 'evoCondition',
                                              'evoType','battleOnly',
                                              'requiredMove', 'evoMove',
                                              'maxHP', "otherFormes",
                                              'baseSpecies', 'forme', 'changesFrom'
                                              ,'baseForme', 'canGigantamax', 'cannotDynamax',
                                               "color", "eggGroups"
                                              ], axis=1)
    
    
    print(master_data_poke.columns)
    
    #type_chart_path = r"C:\Users\taimo\Desktop\SeniorThesis\utils_dire\master_data_process\master_data\typeChart.json"
    '''
    # 1.types
    
   
    
    
    def compute_type_chart(chart_path: str) -> Dict[str, Dict[str, float]]:
        """Returns the pokemon type_chart.

        Returns a dictionnary representing the Pokemon type chart, loaded from a json file
        in `data`. This dictionnary is computed in this file as `TYPE_CHART`.

        :return: The pokemon type chart
        :rtype: Dict[str, Dict[str, float]]
        """
        with open(chart_path) as chart:
            json_chart = orjson.loads(chart.read())
        
        types = [str(entry["name"]) for entry in json_chart]

        type_chart = {type_1: {type_2: 1.0 for type_2 in types} for type_1 in types}

        for entry in json_chart:
            type_ = entry["name"]
            for immunity in entry["immunes"]:
                type_chart[immunity][type_] = 0.0
            for weakness in entry["weaknesses"]:
                type_chart[weakness][type_] = 0.5
            for strength in entry["strengths"]:
                type_chart[strength][type_] = 2.0
        return type_chart
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    type_chart = compute_type_chart(type_chart_path)
    vs_type_feature_list = ['vs_normal', 'vs_fire', 'vs_water',
                    'vs_electric','vs_grass', 
                    'vs_ice', 'vs_fighting', 
                    'vs_poison', 'vs_ground', 'vs_flying'
                    ,'vs_psychic', 'vs_bug', 'vs_rock', 
                    'vs_ghost', 'vs_dragon', 'vs_dark', 
                    'vs_steel', 'vs_fairy']
    
    df_type_multi = pd.DataFrame(1, index=np.arange(len(master_data_poke)),columns=vs_type_feature_list, dtype=np.float32)
    master_data_poke = pd.concat([master_data_poke, df_type_multi], axis=1)
    
    for i, type_list in enumerate(master_data_poke['types']):
        
        for own_type in type_list:
            for k, v in type_chart[own_type].items():
                master_data_poke.loc[i, f"vs_{k.lower()}"] *= v
    
    
    types_unique = ['types_Ice', 'types_Flying', 'types_Ghost',
                    'types_Dragon', 'types_Poison', 'types_Fire', 
                    'types_Electric', 'types_Steel', 'types_Fighting', 
                    'types_Bug', 'types_Dark', 'types_Ground', 'types_Fairy',
                    'types_Rock', 'types_Grass', 'types_Water', 
                    'types_Psychic', 'types_Normal']
    df_types = None
    for i, l in enumerate(master_data_poke['types']):
        series_types = pd.DataFrame(np.zeros((1, len(types_unique))),
                                    columns=types_unique, dtype=np.int32)
        
        for type in l:
            series_types["types_" + type] = 1
          
        if df_types is None:
            df_types = series_types
        else:
            df_types = pd.concat([df_types, series_types])
        
    
    master_data_poke = pd.concat([master_data_poke, df_types.reset_index()], axis=1)
    master_data_poke = master_data_poke.drop(["index", "types"], axis=1)
    
    master_data_poke[types_unique] = master_data_poke[types_unique].astype(np.int32)
    
    #2. baseStats
    
    baseStats_uniques = ["baseStats_hp", "baseStats_atk", 
                      "baseStats_def", "baseStats_spa", 
                      "baseStats_spd", "baseStats_spe"]
    master_data_poke[baseStats_uniques] = 0
    
    for i, dic in enumerate(master_data_poke['baseStats']):
        
        for k, v in dic.items():
            master_data_poke.loc[i, "baseStats_"+k] = v
            
    master_data_poke = master_data_poke.drop(["baseStats"], axis=1)
    
    master_data_poke[baseStats_uniques] = master_data_poke[baseStats_uniques].astype(np.float32)
    
    #3. evos
    
    master_data_poke.loc[(~(master_data_poke["evos"].isnull())) , "evos"] = 1
    master_data_poke.loc[(master_data_poke["evos"].isnull()) , "evos"] = 0
    
    
    #4. eggGroups(no necesary)
    
    eggGroups_uniques = ['eggGroups_Ditto', 'eggGroups_Monster', 
                         'eggGroups_Human-Like', 'eggGroups_Water 3', 
                         'eggGroups_Fairy', 'eggGroups_Bug', 'eggGroups_Grass',
                         'eggGroups_Flying', 'eggGroups_Mineral',
                         'eggGroups_Field', 'eggGroups_Amorphous',
                         'eggGroups_Water 2', 'eggGroups_Water 1',
                         'eggGroups_Dragon', 'eggGroups_Undiscovered']
    master_data_poke[eggGroups_uniques] = 0
    for i, l in enumerate(master_data_poke['eggGroups']):
        
        for eggGr in l:
            master_data_poke.loc[i, 'eggGroups_'+eggGr] = 1
    
    master_data_poke = master_data_poke.drop(['eggGroups'], axis=1)        
    
    master_data_poke[eggGroups_uniques] = master_data_poke[eggGroups_uniques].astype(np.int32)
    
    '''
    '''
    
    
    #5. tags
    
    master_data_poke.loc[(~(master_data_poke["tags"].isnull())) , "tags"] = 1
    master_data_poke.loc[(master_data_poke["tags"].isnull()) , "tags"] = 0
    
    #6. cannnotDynamax
    #master_data_poke.loc[(master_data_poke["cannotDynamax"].isnull()) , "cannotDynamax"] = False
    
    
    #7. canGigantamax
    
    #master_data_poke.loc[(~(master_data_poke["canGigantamax"].isnull())) , "canGigantamax"] = 1
    #master_data_poke.loc[(master_data_poke["canGigantamax"].isnull()) , "canGigantamax"] = 0
    
    #7. color 
    
    #master_data_poke = pd.get_dummies(master_data_poke ,columns=["color"])
    
    
    #7. adjusting type 
    master_data_poke[['heightm', 'weightkg']] = master_data_poke[['heightm', 'weightkg']].astype(np.float32)
    
    master_data_poke[["evos","tags"]] = master_data_poke[["evos", "tags"]].astype(np.int32)
    
    #9. normalization
 
    num_features = [
        'heightm', 'weightkg', 'baseStats_hp', 'baseStats_atk',
       'baseStats_def', 'baseStats_spa', 'baseStats_spd', 'baseStats_spe',
    ]+vs_type_feature_list
    
    
    scaler = MinMaxScaler()
    scaler.fit(master_data_poke[num_features])
    master_data_poke[num_features] = scaler.transform(master_data_poke[num_features])
    
    
    
    
    #11. make csv file
    regex = re.compile('[^a-zA-Z0-9]')
    for i, l in enumerate(master_data_poke['cosmeticFormes']):
        name = master_data_poke.loc[i, "name"]
        
        if isinstance(l, list):
            for j, d in enumerate(l):
                
                new_series = master_data_poke.iloc[i].copy()
                #new_series = new_series.drop(["0"], axis=1)
                
                #print(new_series)
                new_series = new_series.replace(name, d)
                
                master_data_poke = master_data_poke.append(new_series)
      
    
    master_data_poke = master_data_poke.drop(['cosmeticFormes'], axis=1)
    
    master_data_poke["name"] = master_data_poke["name"].apply(lambda x: regex.sub('', x).lower())
    master_data_poke.to_csv('utils_dire\master_data_process\data\poke_data_with_name.csv', index = False)
    
    print(master_data_poke.columns)
    print(len(master_data_poke.columns))
    
    for col in master_data_poke.columns:
        print(col)
        print(master_data_poke[col].unique())
    # test
    '''
    
    