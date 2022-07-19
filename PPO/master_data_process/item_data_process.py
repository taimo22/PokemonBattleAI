import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import re


if __name__ == "__main__":    
    master_data_item = pd.read_json(
        r'src\utils_dire\master_data_process\master_data\items.json')
    master_data_item = master_data_item.T.reset_index()
    
    master_data_item = master_data_item.drop([
        "index", "num", "alias", "gen", "sprite", 
        "desc", "shortDesc", "canBeHeld", "fling",
        "naturalGift", "itemUser", "megaEvolves",
        "zMoveFrom", "zMove", "megaStone", "zMoveType",
        "onNegateImmunity", "onHitPriority", "spritenum",
        "forcedForme"
        
        ], axis=1
                                                   )
    print(master_data_item.columns)
    
    '''
    
    #1. boosts
    
    boosts_unique = ['boosts_atk', 'boosts_spe', 'boosts_def', 'boosts_spa', 'boosts_spd']
    master_data_item[boosts_unique] = 0
    for i, dic in enumerate(master_data_item["boosts"]):
        for k, v in dic.items():
            master_data_item.loc[i, "boosts_"+k] = v
            
    master_data_item = master_data_item.drop(["boosts"], axis=1)
    master_data_item[boosts_unique]=master_data_item[boosts_unique].astype(np.int32)
    
    
    #2. fling
    
    fling_unique = [
        'fling_volatileStatus', 
        'fling_status', 'fling_basePower', 
    ]
    for i, dic in enumerate(master_data_item["fling"]):
        
        for k, v in dic.items():
            print(v)
            master_data_item.loc[i, "fling_"+k] = v
            
    master_data_item = master_data_item.drop(["fling"], axis=1)
    master_data_item[fling_unique]=master_data_item[fling_unique].fillna(value=-100).astype(np.int32)
    '''
    
    '''
    #3. condition
    
    condition_unique = [
        'condition_duration', 'condition_onSourceModifyAccuracyPriority', 
        'condition_onTryMovePriority'
    ]
    for i, dic in enumerate(master_data_item["condition"]):
        
        for k, v in dic.items():
            master_data_item.loc[i, "condition_"+k] = v
            
    master_data_item = master_data_item.drop(["condition"], axis=1)
    #4. itemUser
    
    
    #6. one hot
    master_data_item.loc[(~master_data_item["onPlate"].isnull()), "onPlate"] = 1
    master_data_item.loc[(master_data_item["onPlate"].isnull()), "onPlate"] = 0
    master_data_item["onPlate"] = master_data_item["onPlate"].astype(np.uint8)
    
    master_data_item.loc[(~master_data_item["onDrive"].isnull()), "onDrive"] = 1
    master_data_item.loc[(master_data_item["onDrive"].isnull()), "onDrive"] = 0
    master_data_item["onDrive"] = master_data_item["onDrive"].astype(np.uint8)
    
    
    master_data_item.loc[(~master_data_item["onMemory"].isnull()), "onMemory"] = 1
    master_data_item.loc[(master_data_item["onMemory"].isnull()), "onMemory"] = 0
    
    
    cate_features = [
        "onMemory", "isNonstandard", "onTakeItem", 
        "onEat", "onFractionalPriority", "condition_duration",
        "condition_onSourceModifyAccuracyPriority",
        "condition_onTryMovePriority", 
        "onBasePowerPriority", "isBerry", "onModifySpDPriority",
        "isPokeball", "onTryHealPriority", "onResidualOrder",
        "onResidualSubOrder", "onModifyAccuracyPriority", "isGem",
        "ignoreKlutz", "onModifyAtkPriority", "isChoice",                                       
        "onModifySpAPriority", "onFractionalPriorityPriority",       
        "onAttractPriority", "onAfterMoveSecondaryPriority",                    
        "onModifyDefPriority",  "onModifyMovePriority",                    
        "onBoostPriority", "onDamagePriority",                                
        "onDamagingHitOrder", "onTrapPokemonPriority",                           
        "onAfterMoveSecondarySelfPriority", "onSourceModifyAccuracyPriority"           
    ]
    master_data_item = pd.get_dummies(master_data_item,columns=cate_features)
    
    #print(master_data_item.shape)
    print(master_data_item.columns)
    print(len(master_data_item.columns))
    #print(master_data_item.dtypes)
    
    regex = re.compile('[^a-zA-Z0-9]')
    master_data_item["name"] = master_data_item["name"].apply(lambda x: regex.sub('', x).lower())
    #master_data_item = master_data_item.drop(['Unnamed: 0'], axis=1)
    
    master_data_item.to_csv('utils_dire\master_data_process\data\item_data_with_name.csv', index = False)
    print(master_data_item.shape)
    
    for i, col in enumerate(master_data_item.columns):
        
        print(col)
        print(master_data_item[col].unique())
    '''