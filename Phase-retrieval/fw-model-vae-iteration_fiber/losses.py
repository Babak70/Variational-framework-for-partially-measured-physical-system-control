import tensorflow as tf
import tensorflow.keras.backend as K
from metrics import correlation_coefficient

from vae_utils import log_normal_pdf

def generator_loss(obs_rate,est_rate,loss_type):
    
  if loss_type =="mse":

      
      # loss=tf.math.reduce_mean(tf.keras.losses.MSE(disc_gen_generated_output, target_gen))
      # loss=K.mean(K.square(est_rate - obs_rate), axis=[1,2,3], keepdims=True)
      loss=tf.math.reduce_mean(tf.math.square(est_rate - obs_rate), axis=[1,2,3], keepdims=True)
     
  elif loss_type =="poisson":
      
      obs_rate=tf.transpose(obs_rate)
      est_rate=tf.transpose(est_rate)
      p=tf.keras.losses.Poisson(tf.keras.losses.Reduction.NONE)
      
      loss=p(obs_rate,est_rate)

  elif loss_type =="corr":
      
      loss = correlation_coefficient(obs_rate,est_rate)

  else:
      print('loss not found!')
      
      
  return loss



def vae_loss(obs_rate,est_rate,z,mean,logvar,beta,loss_type):
    
  if loss_type =="mse":

      # loss=tf.math.reduce_mean(tf.keras.losses.MSE(disc_gen_generated_output, target_gen))
      positive_log_likelihood = -K.mean(K.square(est_rate - obs_rate), axis=[1,2,3], keepdims=False)

     
  elif loss_type =="poisson":
      
      # obs_rate=tf.transpose(obs_rate)
      # est_rate=tf.transpose(est_rate)
      p=tf.keras.losses.Poisson(tf.keras.losses.Reduction.NONE)
      
      positive_log_likelihood=-p(obs_rate,est_rate)*1.#because the poisson takes mean and not sum over multiple units (9 here)

  else:
      print('loss not found!')
      

  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  
  loss_kl=-(logpz - logqz_x)
  # loss_kl=0
  
  loss= -tf.reduce_mean(positive_log_likelihood*beta - loss_kl) # loss_kl is normalized by the number of latent_dim (used tf.mean instead of tf.sum)
  
  return loss,tf.reduce_mean(-positive_log_likelihood),tf.reduce_mean(loss_kl)



def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z,apply_sigmoid=True)
  # x_logit = model.decode(z)
  # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  # logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  mse=K.mean(K.square(x_logit - x), axis=[1,2,3], keepdims=True)
  mse=mse*1000
  logpx_z = -tf.reduce_sum(mse, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)