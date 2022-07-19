
from functools import reduce
import pandas as pd
from pandas.core.frame import DataFrame
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import tensorboardX as tbx
import math
import enum

import copy


class AE():
    def __init__(self, latent_dim, size_encoder_input, hidden_size):
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        self.size_encoder_input = size_encoder_input
        self.hidden_size = hidden_size

        input_layer = tf.keras.layers.Input(shape = (size_encoder_input,))
        en_layer1 = tf.keras.layers.Dense(self.hidden_size[0], activation = 'relu')(input_layer)
        en_layer2 = tf.keras.layers.Dense(self.hidden_size[1], activation = 'relu')(en_layer1)
        en_layer3 = tf.keras.layers.Dense(self.hidden_size[2], activation = 'relu')(en_layer2)
        latent = tf.keras.layers.Dense(self.latent_dim, activation = 'linear')(en_layer3)
        self.encoder = tf.keras.Model(inputs = input_layer, outputs = latent, name = 'encoder')
        
        input_layer_decoder = tf.keras.layers.Input(shape = (self.latent_dim, ))
        de_layer1 = tf.keras.layers.Dense(self.hidden_size[2], activation = 'relu')(input_layer_decoder)
        de_layer2 = tf.keras.layers.Dense(self.hidden_size[1], activation = 'relu')(de_layer1)
        de_layer3 = tf.keras.layers.Dense(self.hidden_size[0], activation = 'relu')(de_layer2)
        constructed = tf.keras.layers.Dense(self.size_encoder_input, activation = 'sigmoid')(de_layer3)
        decoder = tf.keras.Model(inputs = input_layer_decoder, outputs = constructed, name= 'decoder')
        
        
        self.autoencoder = tf.keras.Model(inputs = self.encoder.input, outputs = decoder(self.encoder.output))
        
    def run(self, epoch, data, selected):
        lr = 5e-5
        
        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
                                 , loss=tf.keras.losses.MeanSquaredError())
        
        checkpoint_path = f"utils_dire\master_data_process\{selected}_AE_checkpoints\model.ckpt"
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, 
                                                    save_weights_only=True,
                                                    verbose=2
                                                    )
        self.autoencoder.fit(data, 
            data,  
            epochs=epoch,
            #callbacks=[cp_callback], 
            
            ) 



if __name__ == "__main__":   
    basepath = "utils_dire\master_data_process\data"
    '''
    poke: 47
    move: 384
    ability: 42
    item: 73
    
    '''
    options = {
      "move": 20, "poke": 20, "ability": 20,"item":20
    }
    hidden_size = {
        "move": [248, 128, 64], 
        "poke": [40, 32, 24], 
        "ability": [40, 32, 24],
        "item": [64, 32, 24]
    }
    
    
    selected = list(options.keys())[0]
    dim_after = options[selected]
    hidden_size = hidden_size[selected]
    
    
    master_data_pd = pd.read_csv(basepath+f"\{selected}_data_with_name.csv")
    used_featuers = [col for col in master_data_pd.columns if col != "name" and col != 'Unnamed: 0' and col != 'cosmeticFormes']
    
    #print(used_featuers)
    master_data = master_data_pd[used_featuers].values
    
    epoch = 1000
    num_features = master_data.shape[-1]
    
    autoencoder = AE(dim_after, num_features, hidden_size)
    
    
    autoencoder.run(epoch, master_data, selected)
    
    # upload reduced data as csv file
    reduced_data_np = np.zeros((master_data.shape[0], dim_after))
    for i in range(master_data.shape[0]):
        data = np.reshape(master_data[i, :], (1, -1))
        encoded = autoencoder.encoder(data)
        reduced_data_np[i,:] = encoded
        
    mmscaler = MinMaxScaler() 
    mmscaler.fit(reduced_data_np)  
    reduced_data_np = mmscaler.transform(reduced_data_np) 
    
    
    reduced_data_pd = pd.DataFrame(reduced_data_np,
                                   columns=[f"{dim}" for dim in range(1, dim_after+1)])
    reduced_data_pd_to_tsv = copy.deepcopy(reduced_data_pd) 
    reduced_data_pd["name"] = master_data_pd["name"]
    reduced_data_pd_to_tsv_meta = master_data_pd["name"]
    
    # tsv data 
    reduced_data_pd_to_tsv_meta.to_csv(f'C:\\Users\\taimo\\Desktop\\SeniorThesis\\utils_dire\\master_data_process\\reduced_data\\meta_{selected}_reduced_data.tsv'
                           ,index=False, header=False,sep='\t')
    reduced_data_pd_to_tsv.to_csv(f'C:\\Users\\taimo\\Desktop\\SeniorThesis\\utils_dire\\master_data_process\\reduced_data\\{selected}_reduced_data.tsv'
                           ,index=False, header=False, sep='\t')
    print("complete to tsv")
    reduced_data_pd.to_csv(
        f'C:\\Users\\taimo\\Desktop\\SeniorThesis\\utils_dire\\master_data_process\\reduced_data\\{selected}_reduced_data.csv', index=False)