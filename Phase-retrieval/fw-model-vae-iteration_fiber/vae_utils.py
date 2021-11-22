import tensorflow as tf
import numpy as np



# 

def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    # eps = tf.zeros_like(eps,tf.float32)
    # return tf.cast(eps *tf.math.log(1 + tf.exp(logvar * .5)),dtype=tf.float32) + mean # modified a bit for stability 
    return eps *tf.exp(logvar * .5) + mean # modified a bit for stability 

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)