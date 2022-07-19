import collections
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import re
from sklearn.preprocessing import MinMaxScaler
    

if __name__ == "__main__":    
    master_data_move = pd.read_json(
        r'src\utils_dire\master_data_process\master_data\moves.json')
    master_data_move = master_data_move.T.reset_index()
    
    master_data_move_name = master_data_move["name"].copy()
    
    master_data_move = master_data_move.drop(["index", 
                                              "name", 
                                              "isZ", "onDamagePriority",
                                              "contestType", 
                                              'num', "isNonstandard",
                                              "zMove",'onHitSide', 'realMove',
                                              'multihit','isMax', "noSketch",
                                              'basePowerCallback', 'onHit', 'onTryHit', 
                                              'multiaccuracy','sideCondition',
                                              'noMetronome', "secondaries", "ignoreImmunity",
                                              "maxMove"
                                              ], axis=1)
    
    print(master_data_move.columns)
    
    '''
    master_data_move = pd.get_dummies(master_data_move,columns=["category","target", "type"])
    
    
    master_data_move = master_data_move.fillna(value=np.nan)
    
    # change dypes to feed data into model
    master_data_move[master_data_move.select_dtypes(include=np.int64).columns] = master_data_move[master_data_move.select_dtypes(include=np.int64).columns].astype(np.int32)
    master_data_move[master_data_move.select_dtypes(include=np.float64).columns] = master_data_move[master_data_move.select_dtypes(include=np.float64).columns].astype(np.float32)
    
    
    
    
    #=========================================process data whose dtype is object=======================
    # 1. flags
    flags_unique = {
        'flags_gravity', 'flags_contact', 
        'flags_mystery', 'flags_punch', 
        'flags_authentic', 'flags_dance', 
        'flags_nonsky', 'flags_protect', 
        'flags_powder', 'flags_heal', 
        'flags_distance', 'flags_snatch', 
        'flags_recharge', 'flags_mirror', 
        'flags_defrost', 'flags_reflectable', 
        'flags_bullet', 'flags_charge', 
        'flags_pulse', 'flags_bite', 
        'flags_sound'
        } 
    
    df_flags = None
    for dic in master_data_move["flags"]:
        series_flags = pd.DataFrame(np.zeros((1, 21)), columns=flags_unique, dtype=np.int32)
        for k, v in dic.items():
            series_flags["flags_"+k] = v

        if df_flags is None:
            df_flags = series_flags
        else:
            df_flags = pd.concat([df_flags, series_flags])

    master_data_move = pd.concat([master_data_move, df_flags.reset_index()], axis=1)
    
    master_data_move = master_data_move.drop(["index", "flags"], axis=1)
    
    
    
    #2. secondary
    secondary_unique = {
        'secondary_self_boosts_atk', 'secondary_chance', 
        'secondary_dustproof', 'secondary_onHit', 
        'secondary_self_boosts_spd', 'secondary_volatileStatus', 
        'secondary_self_boosts_spa', 'secondary_self_boosts_spe',
        'secondary_self_boosts_def', 'secondary_self_boosts_evasion', 
        'secondary_status'}
    
    
    df_secondary = None
    for dic in master_data_move['secondary']:
        series_secondary = pd.DataFrame(np.zeros((1, len(secondary_unique))),
                                    columns=secondary_unique, dtype=np.int32)
        if dic is np.nan:
            if df_secondary is None:
                df_secondary = series_secondary
            else:
                df_secondary = pd.concat([df_secondary, series_secondary])
            continue
        for k, v in dic.items():
            if isinstance(v, dict):
                for k_2, v_2 in v.items():
                    if isinstance(v_2, dict):
                        for k_3, v_3 in v_2.items():
                            key = 'secondary_'+k+"_"+k_2+"_"+k_3
                            series_secondary[key] = v_3
            elif k not in secondary_unique:
                series_secondary['secondary_'+k] = v
                
        if df_secondary is None:
            df_secondary = series_secondary
        else:
            df_secondary = pd.concat([df_secondary, series_secondary])

    master_data_move = pd.concat([master_data_move, df_secondary.reset_index()], axis=1)
    
    master_data_move = master_data_move.drop(["index", "secondary"], axis=1)
    
    for n in ["secondary_onHit", "secondary_status", "secondary_volatileStatus"]:
        master_data_move.loc[(master_data_move[n] == 0), n] = np.nan
    
    
    # 3 drain
   
    drain_unique =set()
    for i, d in enumerate(master_data_move['drain']):
        if d is np.nan:
            master_data_move.loc[i, "drain"] = 0
        else:
            master_data_move.loc[i, "drain"] = d[0]/d[1]
           
    master_data_move['drain'] = master_data_move['drain'].astype(np.float32)
    
    
    #4.boosts
    
    boosts_unique = {'boosts_spd', 'boosts_evasion', 
                     'boosts_accuracy', 'boosts_spa', 
                     'boosts_atk', 'boosts_def', 'boosts_spe'}
    df_boosts = None
    for i, dic in enumerate(master_data_move["boosts"]):
        series_boosts = pd.DataFrame(np.zeros((1, len(boosts_unique))),
                                    columns=boosts_unique, dtype=np.int32)
        if isinstance(dic, dict):
            for k, v in dic.items():
                series_boosts["boosts_" + k] = v
        
        if df_boosts is None:
            df_boosts = series_boosts
        else:
            df_boosts = pd.concat([df_boosts, series_boosts])
    
    master_data_move = pd.concat([master_data_move, df_boosts.reset_index()], axis=1)
    master_data_move = master_data_move.drop(["index", "boosts"], axis=1)
    
    
    #5. condition
    
    condition_unique = {
        'condition_onAnyModifyDamage', 'condition_onTryAddVolatile', 
        'condition_onTypePriority', 'condition_onFoeRedirectTarget', 
        'condition_duration', 'condition_onBeforeMovePriority', 
        'condition_onSourceInvulnerabilityPriority', 'condition_onDamagePriority',
        'condition_onFieldResidualSubOrder', 'condition_onModifyType', 
        'condition_onStart', 'condition_onAccuracy', 'condition_onBeforeMove', 
        'condition_onFieldEnd', 'condition_onNegateImmunity', 
        'condition_onTryHeal', 'condition_onResidualSubOrder', 
        'condition_onAnyPrepareHitPriority', 'condition_onFoeTrapPokemonPriority', 
        'condition_onTryPrimaryHitPriority', 'condition_onSourceBasePower', 
        'condition_onHit', 'condition_onFieldRestart', 
        'condition_onImmunity', 'condition_onSourceInvulnerability', 
        'condition_onFoeBeforeMove', 'condition_onTryMove', 
        'condition_onOverrideAction', 'condition_onAnyDragOut', 
        'condition_onType', 'condition_onResidualOrder', 
        'condition_onFoeRedirectTargetPriority', 'condition_onTryHit', 
        'condition_onDamage', 'condition_onLockMove', 
        'condition_onAnyInvulnerability', 'condition_onCopy', 
        'condition_onSourceAccuracy', 'condition_onBeforeSwitchOut', 
        'condition_onFaint', 'condition_onEnd', 'condition_onDragOut', 
        'condition_onTryMovePriority', 'condition_onFieldResidualOrder', 
        'condition_onAllyTryHitSide', 'condition_onSideResidualOrder', 
        'condition_onModifyAccuracy', 'condition_onFoeBeforeMovePriority', 
        'condition_onRedirectTargetPriority', 'condition_durationCallback', 
        'condition_onEffectiveness', 'condition_onDamagingHit', 
        'condition_onModifyCritRatio', 'condition_onSideStart', 
        'condition_onSideResidualSubOrder', 'condition_onUpdate', 
        'condition_onSideRestart', 'condition_onRestart', 
        'condition_onSwap', 'condition_onAccuracyPriority', 
        'condition_onRedirectTarget', 'condition_noCopy', 
        'condition_onAnyBasePower', 'condition_onAnyPrepareHit', 
        'condition_onFieldStart', 'condition_onAnySetStatus', 
        'condition_onModifyTypePriority', 'condition_onModifyMove', 
        'condition_onDisableMove', 'condition_onMoveAborted', 
        'condition_onTrapPokemon', 'condition_onFoeTrapPokemon', 
        'condition_onBoost', 'condition_onModifySpe', 'condition_onSideEnd', 
        'condition_onSetStatus', 'condition_onEffectivenessPriority',
        'condition_onInvulnerability', 'condition_onTryPrimaryHit', 
        'condition_onSwitchIn', 'condition_onModifyBoost', 
        'condition_onCriticalHit', 'condition_onFoeDisableMove', 'condition_onBasePower',
        'condition_onBasePowerPriority', 'condition_onResidual', 'condition_onSourceModifyDamage', 
        'condition_onTryHitPriority'
        }
    
    df_condition = None
    
    
    for i, d in enumerate(master_data_move["condition"]):
        series_condition = pd.DataFrame(np.zeros((1, len(condition_unique))),
                                    columns=condition_unique)
        if isinstance(d, dict):
            for k, v in d.items():
                #print(k, v)
                series_condition["condition_" + k] = v
        
        if df_condition is None:
            df_condition = series_condition
        else:
            df_condition= pd.concat([df_condition, series_condition])
        
    
    master_data_move = pd.concat([master_data_move, df_condition.reset_index()], axis=1)
    master_data_move = master_data_move.drop(["index", "condition"], axis=1)
    
    
    for col in condition_unique:
        if master_data_move[col].dtype == object:
            master_data_move.loc[(master_data_move[col]==0.0), col] = np.nan
           
    
    
    
    
    # 6.others
  
    
    
    
    # recoil and "heal"
    for i, (r, h) in enumerate(zip(master_data_move["recoil"], master_data_move["heal"])):
        
        
        if isinstance(r, list):
            
            master_data_move.loc[i, "recoil"] = r[0]/r[1]
        else:
            master_data_move.loc[i, "recoil"] = 0
        
        
        
        if isinstance(h, list):
            master_data_move.loc[i, "heal"] = h[0]/h[1]
        else:
            master_data_move.loc[i, "heal"] = 0
    master_data_move = master_data_move.astype({"heal": np.float32, "recoil": np.float32})
    
    
    # "self", "selfBoost"
    self_unique = {
        'self_boosts_spe', 'self_onHit', 'self_chance', 'self_sideCondition', 
        'self_boosts_def', 'self_boosts_atk', 'self_boosts_spa',
        'self_volatileStatus', 'self_boosts_spd', 'self_pseudoWeather'
        }
    
    maxMove_unique = {'maxMove_basePower'}
    selfBoost_unique = {'selfBoost_boosts_spe', 
                        'selfBoost_boosts_spd', 
                        'selfBoost_boosts_atk', 
                        'selfBoost_boosts_spa', 
                        'selfBoost_boosts_def'
                        }
    
    
    
    names = ["self", "selfBoost"]
    def sort_data(name, uniques):
        df = None
        for i, d in enumerate(master_data_move[name]):
            series = pd.DataFrame(np.zeros((1, len(uniques))),
                                    columns=uniques)
            if isinstance(d , dict):
                for k, v in d.items():
                    if isinstance(v , dict):
                        for k_2, v_2 in v.items():
                            
                            key = name+"_"+k+"_"+k_2
                            series[key] = v_2
                            
                            
                    else:
                        key = name+"_"+k
                        series[key] = v
                        
            if df is None:
                df = series
            else:
                df = pd.concat([df, series])
                
        return df
    
    
    for n, u in zip(names, [self_unique, maxMove_unique, selfBoost_unique]):
        df = sort_data(n, u)
        master_data_move = pd.concat([
            master_data_move, df.reset_index()
            ], axis=1)
        
        master_data_move = master_data_move.drop(["index", n], axis=1)
    
    print(master_data_move["self_pseudoWeather"].unique())
    for n in ["self_pseudoWeather", "self_boosts_spd",
              "self_onHit", "self_sideCondition", "self_volatileStatus"]:
        master_data_move.loc[(master_data_move[n] == 0.0), n] = np.nan
    
    #
    # 7, final step: deal with categorical variables
    
    features_name_object = [
       'volatileStatus','onTry', 'onModifyType',
       'stallingMove', 'onPrepareHit', 'selfSwitch', 'beforeTurnCallback',
       'onAfterMove', 'onModifyMove','beforeMoveCallback',
       'useSourceDefensiveAsOffensive',  'onTryMove', 
       'onBasePower', 'onTryImmunity', 'ignoreDefensive',
       'ignoreEvasion', 'forceSwitch', 'onAfterSubDamage',
       'damageCallback', 'onHitField', 'onAfterHit', 'nonGhostTarget',
       'status', 'isFutureMove', 'smartTarget', 'damage', 'terrain',
       'selfdestruct', 'pseudoWeather', 'onDamage', 'breaksProtect',
       'onAfterMoveSecondarySelf',  'ohko', 'onEffectiveness',
       'useTargetOffensive', 'willCrit', 'ignoreAbility',
       'onModifyPriority', 'weather', 'slotCondition',
       'hasCrashDamage', 'onMoveFail', 'pressureTarget', 'onUseMoveMessage',
       'onModifyTarget', 'mindBlownRecoil', 'defensiveCategory',
       'thawsTarget', 'noPPBoosts', 'sleepUsable', 'tracksTarget',
       'stealsBoosts', 'struggleRecoil'
    ]
    
    master_data_move.loc[(master_data_move["accuracy"]==True),"accuracy"] = 120
    master_data_move = master_data_move.astype({"accuracy": np.int32})
    
    for name in features_name_object:
        if name in master_data_move.columns:
            master_data_move = master_data_move.astype({name: np.str})
        else:
            print(name)
        
        
    categorical_features = [
        "secondary_onHit", "secondary_status", "secondary_volatileStatus",
        'volatileStatus','onTry', 'onModifyType',
       'stallingMove', 'onPrepareHit', 'selfSwitch', 'beforeTurnCallback',
       'onAfterMove', 'onModifyMove','beforeMoveCallback',
       'useSourceDefensiveAsOffensive',  'onTryMove', 
       'onBasePower', 'onTryImmunity', 'ignoreDefensive',
       'ignoreEvasion', 'forceSwitch', 'onAfterSubDamage',
       'damageCallback', 'onHitField', 'onAfterHit', 'nonGhostTarget',
       'status', 'isFutureMove', 'smartTarget', 'damage', 'terrain',
       'selfdestruct', 'pseudoWeather', 'onDamage', 'breaksProtect',
       'onAfterMoveSecondarySelf',  'ohko', 'onEffectiveness',
       'useTargetOffensive', 'willCrit', 'ignoreAbility',
       'onModifyPriority', 'weather', 'slotCondition',
       'hasCrashDamage', 'onMoveFail', 'pressureTarget', 'onUseMoveMessage',
       'onModifyTarget', 'mindBlownRecoil', 'defensiveCategory',
       'thawsTarget', 'noPPBoosts', 'sleepUsable', 'tracksTarget',
       'stealsBoosts', 'struggleRecoil', 'condition_onBeforeSwitchOut',
       'condition_onAccuracy', 'condition_onModifySpe', 'condition_onDragOut', 
       'condition_onAnyInvulnerability', 'condition_onInvulnerability',
       'condition_onStart', 'condition_onCopy', 'condition_onRedirectTarget', 
       'condition_onFoeBeforeMove', 'condition_onTryPrimaryHit', 
       'condition_onModifyCritRatio', 'condition_onResidual', 
       'condition_onSideRestart', 'condition_onAnyModifyDamage', 
       'condition_onSideStart', 'condition_onSideEnd', 'condition_onBeforeMove', 
       'condition_onFieldRestart', 'condition_onSourceBasePower', 
       'condition_onModifyMove', 'condition_onSourceInvulnerability', 
       'condition_onSwap', 'condition_onTryMove', 'condition_onTryHit', 
       'condition_onUpdate', 'condition_onTryHeal', 'condition_onFoeRedirectTarget',
       'condition_onSourceAccuracy', 'condition_onBasePower', 
       'condition_onAnyPrepareHit', 'condition_onType', 'condition_onMoveAborted', 
       'condition_onDamagingHit', 'condition_onFaint', 'condition_onModifyAccuracy', 
       'condition_onBoost', 'condition_onAllyTryHitSide', 'condition_onSourceModifyDamage',
       'condition_onModifyBoost', 'condition_onEnd',
       'condition_onLockMove', 'condition_onAnyDragOut', 'condition_onTrapPokemon', 
       'condition_onEffectiveness', 'condition_onSetStatus', 'condition_onFieldStart', 'condition_onSwitchIn',
       'condition_onDisableMove', 'condition_onNegateImmunity', 'condition_onAnyBasePower', 'condition_onFieldEnd', 'condition_onDamage',
       'condition_onModifyType', 'condition_onFoeTrapPokemon',
       'condition_onAnySetStatus', 'condition_onFoeDisableMove', 'condition_onHit',
       'condition_onOverrideAction', 'condition_durationCallback', 'condition_onImmunity', 
       'condition_onRestart', 'condition_onTryAddVolatile', 'self_sideCondition', 
       'self_boosts_spd', 'self_onHit','self_volatileStatus', 'self_pseudoWeather'
    ]
    
    for ca in categorical_features:
        if ca not in master_data_move.columns:
            print(ca)
    
    master_data_move = pd.get_dummies(master_data_move,columns=categorical_features)
 
 
 
    
    # normalization
    normilized_features = master_data_move.columns
    scaler = MinMaxScaler()
    scaler.fit(master_data_move[normilized_features])
    master_data_move[normilized_features] = scaler.transform(master_data_move[normilized_features])
    
    
    
    # making csv
    master_data_move = pd.concat([master_data_move, master_data_move_name], axis=1)
    regex = re.compile('[^a-zA-Z0-9]')
    
    master_data_move["name"] = master_data_move["name"].apply(lambda x: regex.sub('', x).lower())
    master_data_move = master_data_move.fillna(value=-1)
    
    master_data_move.to_csv('utils_dire\master_data_process\data\move_data_with_name.csv', index = False)
    
    print(master_data_move[master_data_move.columns].isnull().all(axis=0))
    print(len(master_data_move.columns))
    print(master_data_move.dtypes)
    
    for col in master_data_move.columns:
        if col != "name":
            print(col) 
            print(master_data_move[col].unique())   
    '''
    
    
    