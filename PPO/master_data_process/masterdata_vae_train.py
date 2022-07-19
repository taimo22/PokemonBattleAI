
import pandas as pd
from scipy.stats.stats import mode
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, optimizers

from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import tensorboardX as tbx
import math
import enum
import copy

class VAE:

    def __init__(self, original_dim, hidden_sizes, latent_dim):
        """コンストラクタ

        :param original_dim: 銘柄数の次元
        :param intermediate_dim: 中間層の次元
        :param latent_dim: 潜在空間の次元
        """
        self.original_dim = original_dim
        
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.callback = None
        self.hidden_size = hidden_sizes

    def setup_model(self):
        """モデルのセットアップ

        :return:
        """
        # encoder
        inputs = Input(shape=(self.original_dim, ), name='encoder_input')
        x = Dense(self.hidden_size[0], activation='relu')(inputs)
        x = Dense(self.hidden_size[1], activation='relu')(x)
        x = Dense(self.hidden_size[2], activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        z = Lambda(self.__sampling, output_shape=(self.latent_dim, ), name='z')([z_mean, z_log_var])
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # decoder
        latent_inputs = Input(shape=(self.latent_dim, ), name='z_sampling')
        x = Dense(self.hidden_size[2], activation='relu')(latent_inputs)
        x = Dense(self.hidden_size[1], activation='relu')(x)
        x = Dense(self.hidden_size[0], activation='relu')(x)
        x = Dense(self.original_dim, activation='linear')(x)
        self.decoder = Model(latent_inputs, x, name='decoder')

        # vae = encoder + decoder
        measure_dist_angle = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, measure_dist_angle, name='variational_autoencoder')

        # loss function
        reconstruction_loss = mse(inputs, measure_dist_angle)
        #reconstruction_loss *= self.original_dim/10
        #reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=measure_dist_angle, labels=inputs)
        #reconstruction_loss = -tf.reduce_sum(reconstruction_loss, axis=1)
        
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        
        self.vae.compile(optimizer=optimizers.Adam(learning_rate=0.003),)

        # callbacks
        self.callback = callbacks.EarlyStopping(
            monitor='loss', patience=10, verbose=1, mode='auto')
    
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)    
    
    def fit(self, x_train, epochs, batch_size):
        """学習の実行

        :param x_train: 訓練データ
        :param epochs: エポック回数
        :param batch_size: バッチサイズ
        :return:
        """
        self.vae.fit(x_train, epochs=epochs, batch_size=batch_size, callbacks=[self.callback])

    def save_weights(self, path):
        """学習モデルのセーブ

        :param path: セーブするファイルパス
        :return:
        """
        self.vae.save_weights(path)

    def load_weights(self, path):
        """学習モデルのロード

        :param path: ロードするファイルパス
        :return:
        """
        self.vae.load_weights(path)

    def decode(self, x, y, batch_size):
        return self.decoder.predict([[x, y]], batch_size=batch_size)

    def encode(self, x, batch_size):
        return self.encoder.predict(x, batch_size=batch_size)

    @staticmethod
    def __sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))

        return z_mean + K.exp(0.5 * z_log_var) * epsilon



if __name__ == "__main__":   
    basepath = r"src\utils_dire\master_data_process\data"
    '''
    poke: 47
    move: 383
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
    
    
    selected = list(options.keys())[1]
    dim_after = options[selected]
    hidden_size = hidden_size[selected]
    
    master_data_pd = pd.read_csv(basepath + f"\{selected}_data_with_name.csv")
    used_featuers = [col for col in master_data_pd.columns if col != "name" and col != 'Unnamed: 0' and col != 'cosmeticFormes']
    
    #print(used_featuers)
    master_data = master_data_pd[used_featuers].values
    mu = np.expand_dims(np.mean(master_data, axis=1), axis=1)
    std =  np.expand_dims(np.std(master_data, axis=1), axis=1)
    master_data = master_data - mu
    master_data = master_data / std
    
    # increasing data
    vae = VAE(
        original_dim=len(used_featuers), 
        hidden_sizes=hidden_size, 
        latent_dim=dim_after
        )
    vae.setup_model()
    vae.encoder.summary()
    vae.decoder.summary()
    
    
    vae.fit(master_data, epochs=300, batch_size=10)
    
    z_mean, z_log_var, z = vae.encode(master_data, batch_size=100)
    reduced_data_np = z
    print(reduced_data_np.shape)
    # upload reduced data as csv file
    reduced_data_pd = pd.DataFrame(reduced_data_np,
                                   columns=[f"{dim}" for dim in range(1, dim_after+1)])
    reduced_data_pd_to_tsv = copy.deepcopy(reduced_data_pd) 
    reduced_data_pd["name"] = master_data_pd["name"]
    reduced_data_pd_to_tsv_meta = master_data_pd["name"]
    
    # tsv data 
    reduced_data_pd_to_tsv_meta.to_csv(
      f'C:\\Users\\taimo\\Desktop\\SeniorThesis\\src\\utils_dire\\master_data_process\\vae_reduced_data\\meta_{selected}_reduced_data.tsv'
                           ,index=False, header=False,sep='\t')
    reduced_data_pd_to_tsv.to_csv(
      f'C:\\Users\\taimo\\Desktop\\SeniorThesis\\src\\utils_dire\\master_data_process\\vae_reduced_data\\{selected}_reduced_data.tsv'
                           ,index=False, header=False, sep='\t')
    print("complete to tsv")
    reduced_data_pd.to_csv(
        f'C:\\Users\\taimo\\Desktop\\SeniorThesis\\src\\utils_dire\\master_data_process\\vae_reduced_data\\{selected}_reduced_data.csv', index=False)