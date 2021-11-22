import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D as Conv2D
from tensorflow.keras.layers import Conv1D as Conv1D
from tensorflow.keras.layers import MaxPooling2D as Maxpool
from tensorflow.keras.layers import UpSampling2D as Upsample
from tensorflow.keras.layers import Reshape as Reshape
from tensorflow.keras.layers import Activation as Activation
from tensorflow.keras.activations import exponential as Exp
from tensorflow.keras.layers import Flatten as Flatten
from tensorflow.keras.layers import Dense as Dense
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import BatchNormalization as BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D as ZP

def Fully_connected_matrix(HEIGHT_in,WIDTH_in,NUM_frames):
    
  initializer = tf.random_uniform_initializer(minval=0., maxval=1.)
  # initializer = tf.random_normal_initializer(0.,0.02)
  T=tf.Variable(initial_value=initializer(shape=[NUM_frames,HEIGHT_in,WIDTH_in,1], dtype=tf.float32), trainable=True,name='real')
  
  return T


def forward_model_vae(HEIGHT_in,WIDTH_in,HEIGHT_out,WIDTH_out,sigma=0.1,mean_firing_rate=0.05):
  inputs= tf.keras.layers.Input(shape=[1,HEIGHT_in,WIDTH_in])

  

  blocks_1 = [
            Conv2D(filters=4, kernel_size=21,padding='valid',data_format='channels_first', use_bias=False,kernel_regularizer=l2(1e-5)),
            Reshape(( 4,(HEIGHT_in-21+1)*(WIDTH_in-21+1))) ]

  blocks_2=[
            
            Reshape(( 4,(HEIGHT_in-21+1),(WIDTH_in-21+1))),
            # tf.keras.layers.GaussianNoise(sigma),
            Activation('relu'),
            Conv2D(filters=4, kernel_size=15,padding='valid',use_bias=True, data_format='channels_first', kernel_regularizer=l2(1e-5)),
            # tf.keras.layers.GaussianNoise(sigma),
            Activation('relu'),
            Flatten(),
            Dense(HEIGHT_out*WIDTH_out,kernel_initializer='normal',use_bias=True,bias_initializer=tf.keras.initializers.Constant(value=(mean_firing_rate)),kernel_regularizer=l2(1e-5),activity_regularizer=None),
            # Activation('softplus'),
  ]
  
  x = inputs
  # Going through the model
  for block in blocks_1:
    x = block(x)
  
  x = tf.transpose(x,perm=[2,1,0])
 

  x = Conv1D(filters=4, kernel_size=40,padding='same',data_format='channels_first', groups=4, use_bias=True,kernel_regularizer=l2(1e-5))(x)
  

  x = tf.transpose(x,perm=[2,1,0])
  

  for block in blocks_2:
    x = block(x)
  
  last=Exp(x)
  return tf.keras.Model(inputs=inputs, outputs=last)

def encoder_vae(HEIGHT_in,WIDTH_in,HEIGHT_out,WIDTH_out,latent_dims,sigma=0.1):
  inputs= tf.keras.layers.Input(shape=[1,HEIGHT_in,WIDTH_in])

  initializer_mean = tf.keras.initializers.RandomNormal(mean=0., stddev=0.001)
  # initializer_log_var = tf.keras.initializers.RandomNormal(mean=-2.5, stddev=0.001)
  blocks_1 = [               
              Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=16, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Flatten(),
              Dense(2*latent_dims,kernel_initializer=initializer_mean,use_bias=False,kernel_regularizer=None,activity_regularizer=None),
              
            ]


  x = inputs
  # Going through the model
  for block in blocks_1:
    x = block(x)
  last=x
  return tf.keras.Model(inputs=inputs, outputs=last)


def decoder_vae(HEIGHT_in,WIDTH_in,HEIGHT_out,WIDTH_out,latent_dims,sigma=0.1,mean_firing_rate=0.05):
  inputs= tf.keras.layers.Input(shape=[latent_dims])
  
  blocks_1=[
      
            Dense(12*12*16,kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None,name='fuck1'),
            Reshape((16,12,12)),
            Activation('relu'),
            Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None,name='fuck2'),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),            
            ZP(((1,0),(1,0)),data_format='channels_first'),          
            Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None,name='fuck3'),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),
            Conv2D(filters=1, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None,name='fuck4'),
            Activation('sigmoid'),
  ]

  
  x = inputs
  # Going through the model
  for block in blocks_1:
    x = block(x)

  
  last=x
  return tf.keras.Model(inputs=inputs, outputs=last)



def backward_model(HEIGHT_in,WIDTH_in,HEIGHT_out,WIDTH_out):
  inputs= tf.keras.layers.Input(shape=[1,HEIGHT_in,WIDTH_in])
  

  blocks_1 = [               
              Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=16, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Flatten(),
              Dense(HEIGHT_out*WIDTH_out,kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None),
              Reshape((1,HEIGHT_out,WIDTH_out)),
              
            ]

  blocks_2=[
      
            Flatten(), 
            Dense(12*12*16,kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None),
            Reshape((16,12,12)),
            Activation('relu'),
            Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),            
            ZP(((1,0),(1,0)),data_format='channels_first'),          
            Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),
            Conv2D(filters=1, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),

  ]
  
  x = inputs

  # Going through the model
  for block in blocks_1:
    x = block(x)

  for block in blocks_2:
     x = block(x)
    
  last=x
  return tf.keras.Model(inputs=inputs, outputs=last)



def Auto_encoder(HEIGHT_in,WIDTH_in,HEIGHT_red,WIDTH_red):
  inputs= tf.keras.layers.Input(shape=[1,HEIGHT_in,WIDTH_in])
  

  blocks_1 = [               
              Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=16, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Flatten(),
              Dense(HEIGHT_red*WIDTH_red,kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None),
              Reshape((1,HEIGHT_red,WIDTH_red)),
              
            ]

  blocks_2=[
      
            Flatten(), 
            Dense(12*12*16,kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None),
            Reshape((16,12,12)),
            Activation('relu'),
            Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),            
            ZP(((1,0),(1,0)),data_format='channels_first'),          
            Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),
            Conv2D(filters=1, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),

  ]
  
  x = inputs

  # Going through the model
  for block in blocks_1:
    x = block(x)

  

  red_img=x

  for block in blocks_2:
    x = block(x)
  
  
  last=x
  return tf.keras.Model(inputs=inputs, outputs=[last,red_img])

def Attention (HEIGHT_in,WIDTH_in,HEIGHT_out,WIDTH_out,slots):
  inputs= tf.keras.layers.Input(shape=[2,HEIGHT_in,WIDTH_in])
  

  blocks_1 = [               
              Conv2D(filters=8, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=8, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=8, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Flatten(),
              Dense(HEIGHT_out*WIDTH_out*8,kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None),
              Activation('relu'),
              # Reshape((1,HEIGHT_out,WIDTH_out)),
              
            ]

  blocks_2=[
      
            # Flatten(), 
            Dense(12*12*8,kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None),
            Reshape((8,12,12)),
            Activation('relu'),
            Conv2D(filters=8, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),            
            ZP(((1,0),(1,0)),data_format='channels_first'),          
            Conv2D(filters=8, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),
            Conv2D(filters=1, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
  ]
  
  
  stimuli, log_Sprev = tf.split(inputs,2,axis=1)
  x =inputs
  # Going through the model
  for block in blocks_1:
        x = block(x)
        # print(x.shape)
    
  for block in blocks_2:
         x = block(x)
         # print(x.shape)
         
  alpha=Activation('sigmoid')(x)
  # print(x.shape)
      # log_alpha=tf.math.log_sigmoid()
  log_alpha=tf.math.log(alpha +1e-4)
  log_alpha_not=tf.math.log(1-alpha+1e-4)
  m_next = log_alpha + log_Sprev
  log_Sprev = log_alpha_not + log_Sprev 
  
 
    
  Masks=m_next 
  for jj in range(slots-1):
      x = tf.concat([stimuli,log_Sprev],axis=1)
      # Going through the model
      for block in blocks_1:
        x = block(x)
    
      for block in blocks_2:
         x = block(x)
         
      alpha=Activation('sigmoid')(x)
      # log_alpha=tf.math.log_sigmoid()
      log_alpha=tf.math.log(alpha + 1e-4)
      log_alpha_not=tf.math.log(1-alpha+ 1e-4)
      
      if jj == (slots-1-1):
          m_next =  log_Sprev
      else:
          m_next = log_alpha + log_Sprev
      Masks=tf.concat([Masks,m_next], axis=1)
      log_Sprev = log_alpha_not + log_Sprev
       
    
  last=Masks
  return tf.keras.Model(inputs=inputs, outputs=[last,alpha])

def Actor_vae(HEIGHT_in,WIDTH_in,HEIGHT_out,WIDTH_out):
  inputs= tf.keras.layers.Input(shape=[1,HEIGHT_in,WIDTH_in])
  

  blocks_1 = [               
              Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Maxpool((2,2),data_format='channels_first'),
              Conv2D(filters=16, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
              Activation('relu'),
              Flatten(),
              Dense(HEIGHT_out*WIDTH_out,kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None),
              Reshape((1,HEIGHT_out,WIDTH_out)),
              
            ]

  blocks_2=[
      
            Flatten(), 
            Dense(12*12*16,kernel_initializer='normal',use_bias=True,kernel_regularizer=None,activity_regularizer=None),
            Reshape((16,12,12)),
            Activation('relu'),
            Conv2D(filters=32, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),            
            ZP(((1,0),(1,0)),data_format='channels_first'),          
            Conv2D(filters=64, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('relu'),
            Upsample((2,2),data_format='channels_first'),
            Conv2D(filters=1, kernel_size=3,padding='same',data_format='channels_first', use_bias=True,kernel_regularizer=None),
            Activation('sigmoid'),
  ]
  
  x = inputs

  # Going through the model
  for block in blocks_1:
    x = block(x)

  for block in blocks_2:
     x = block(x)
    
  last=x
  return tf.keras.Model(inputs=inputs, outputs=last)



def bn_layer(x, nchan, size, l2_reg, sigma=0.05):

    """An individual batchnorm layer"""
    print(np.shape(x))
    n = int(x.shape[-1]) - size + 1
    

    y = Conv2D(nchan, size, data_format="channels_first", kernel_regularizer=l2(l2_reg))(x)

    y =Reshape((nchan, n, n))(BatchNormalization(axis=-1)(Flatten()(y)))

    return Activation('relu')(GaussianNoise(sigma)(y))

def bn_cnn(HEIGHT_in,WIDTH_in,HEIGHT_out,WIDTH_out, l2_reg=0.01):

    """Batchnorm CNN model"""
    n_out=HEIGHT_out*WIDTH_out
    inputs = tf.keras.layers.Input(shape=[HEIGHT_in,WIDTH_in,1])
    inputs2=tf.transpose(inputs,perm=[0,3,1,2])
    y = bn_layer(inputs2, 8, 10, l2_reg)

    y = bn_layer(y, 8, 7, l2_reg)

    y = Dense(n_out, use_bias=False)(Flatten()(y))

    last= Activation('softplus')(BatchNormalization(axis=-1)(y))
    last=tf.reshape(last,[-1,91,1])
    return tf.keras.Model(inputs=inputs, outputs=last, name='BN-CNN')




  


