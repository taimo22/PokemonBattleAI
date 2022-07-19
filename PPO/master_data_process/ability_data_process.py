import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
import re


if __name__ == "__main__":    
    master_data_ability = pd.read_json(
        r"src\utils_dire\master_data_process\master_data\abilities.json")
    master_data_ability = master_data_ability.T.reset_index()
    
    
    master_data_ability = master_data_ability.drop([
        "index", "num", "desc", "shortDesc", "slug", "onFoeTryEatItem", "alias", 'rating'
        ], axis=1)
    
    print(master_data_ability.columns)
    
    '''
    
    
    # 1. condition
    condition_unique = ['condition_onModifyAtkPriority', 
                        'condition_duration', 
                        'condition_noCopy', 
                        'condition_onModifySpAPriority'
    ]
    for i, dic in enumerate(master_data_ability["condition"]):
        for k, v in dic.items():
            master_data_ability.loc[i, "condition_"+k] = v
            
    
    master_data_ability = master_data_ability.drop(["condition"], axis=1)
    
    
    
    
    # one hot encoding 
    master_data_ability = pd.get_dummies(master_data_ability ,columns=
        condition_unique+["onSourceModifyDamagePriority", 
                          "isNonstandard", "onFractionalPriority", 
                          "onCriticalHit"]
        )
     
     
    # adjustment of types
    master_data_ability[["isUnbreakable", "suppressWeather"]] = master_data_ability[["isUnbreakable", "suppressWeather"]].astype(np.uint8)
    
    
    float_features = [
        "onBeforeMovePriority",
    ]
    master_data_ability[float_features] = master_data_ability[float_features].astype(np.float32)
    
    int_features = [
        "onModifyTypePriority", "onBasePowerPriority", "onDamagingHitOrder",
        "onResidualOrder","onResidualSubOrder", "onAllyBasePowerPriority", 
        "onModifyAtkPriority", "onModifySpAPriority", "onSourceModifyAccuracyPriority", 
        "onModifyMovePriority", "onAnyBasePowerPriority", "onDamagePriority", 
        "onFoeBasePowerPriority", "onAllyModifyAtkPriority", "onAllyModifySpDPriority",
        "onModifyDefPriority", "onSourceBasePowerPriority", "onModifyWeightPriority",
        "onAnyFaintPriority",  "onDragOutPriority","onSourceModifyAtkPriority", "onSourceModifySpAPriority",     
        "onFractionalPriorityPriority", "onTryEatItemPriority", "onModifyAccuracyPriority",                
        "onTryHitPriority", "onAnyInvulnerabilityPriority",  
        ]
    
    master_data_ability[int_features] = master_data_ability[int_features].astype(np.int32)
    
    # normalization
 
    normilized_features = [
        col for col in master_data_ability.columns if col != "name"
    ]
    
    
    scaler = MinMaxScaler()
    scaler.fit(master_data_ability[normilized_features])
    master_data_ability[normilized_features] = scaler.transform(master_data_ability[normilized_features])
    
    
    # upload as csv
    regex = re.compile('[^a-zA-Z0-9]')
    
    #First parameter is the replacement, second parameter is your input string
    print(master_data_ability.columns)
    print(len(master_data_ability.columns))
    print(master_data_ability.shape)
    
    master_data_ability["name"] = master_data_ability["name"].apply(lambda x: regex.sub('', x).lower())
    #master_data_ability.to_csv('utils_dire\\master_data_process\\data\\ability_data_with_name.csv', index=False)
    '''

'''
42
(268, 42)
'''