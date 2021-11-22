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
from tensorflow.keras.initializers import TruncatedNormal as TN
from tensorflow.keras.initializers import Constant as CNST

def Fully_connected_matrix(HEIGHT_in,WIDTH_in,NUM_frames):
    
  initializer = tf.random_uniform_initializer(minval=0., maxval=1.)
  # initializer = tf.random_normal_initializer(0.,0.02)
  T=tf.Variable(initial_value=initializer(shape=[NUM_frames,HEIGHT_in,WIDTH_in,1], dtype=tf.float32), trainable=True,name='real')
  
  return T





def forward_model_vae(HEIGHT_in,WIDTH_in,HEIGHT_out,WIDTH_out,sigma=0.1,mean_firing_rate=0.05):
  inputs= tf.keras.layers.Input(shape=[1,HEIGHT_out,WIDTH_out])


  blocks_1 = [Flatten(),

            Dense(HEIGHT_out*WIDTH_out,kernel_initializer=TN(stddev=0.05),bias_initializer=CNST(value=0)),
            Activation('sigmoid'),
            Reshape(( 1,HEIGHT_out,WIDTH_out)),
            
            ]

  x = inputs
  print(x.shape)
  # Going through the model
  for block in blocks_1:
    x = block(x)
    print(x.shape)

  
  last=x
  return tf.keras.Model(inputs=inputs, outputs=last)

def encoder_vae(HEIGHT_in,WIDTH_in,HEIGHT_out,WIDTH_out,latent_dims,sigma=0.1):
  inputs= tf.keras.layers.Input(shape=[1,HEIGHT_in,WIDTH_in])

  # initializer_mean = tf.keras.initializers.RandomNormal(mean=0., stddev=0.001)
  # initializer_log_var = tf.keras.initializers.RandomNormal(mean=-2.5, stddev=0.001)
  blocks_1 = [               
              Flatten(),
              Dense(2*latent_dims,kernel_initializer=TN(stddev=0.05),bias_initializer=CNST(value=0)),
              
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
      
            Dense(HEIGHT_out*WIDTH_out,kernel_initializer=TN(stddev=0.05),bias_initializer=CNST(value=0)),
            Reshape((1,HEIGHT_out,WIDTH_out)),
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
  


  blocks_1 = [Flatten(),
              Dense(HEIGHT_out*WIDTH_out,kernel_initializer=TN(stddev=0.05),bias_initializer=CNST(value=0)),
              Activation('sigmoid'),
              Reshape((1,HEIGHT_out,WIDTH_out)),

            ]

  x = inputs
  print(x.shape)

  # Going through the model
  for block in blocks_1:
    x = block(x)
    print(x.shape)

    
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
