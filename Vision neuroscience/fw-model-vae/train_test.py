import tensorflow as tf
import tensorflow.keras.backend as K
import datetime
import matplotlib.pyplot as plt
from metrics import fraction_of_explained_variance,correlation_coefficient

import numpy as np
from losses import vae_loss,generator_loss

from vae_utils import reparameterize
from my_network import Attention, Actor_vae

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))



@tf.function
def train_forward(stimuli, spikes, epoch,train_what,encoder,decoder,generator,encoder_optimizer,decoder_optimizer,generator_optimizer,beta):

   if train_what=='train':
         
      with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape,tf.GradientTape() as fwd_tape:
            mean_log_var= encoder(stimuli, training=True)
            mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
            z=reparameterize(mean,logvar)
            
            tsnf_stimuli=decoder(z, training=True)
           
            est_rate=  generator(tsnf_stimuli, training=True)
            est_rate=est_rate +1e-4
            obs_rate=spikes
        
            gen_total_loss,loss_rec,loss_kl,loss_rec_norm,loss_kl_norm = vae_loss(obs_rate,est_rate,z,mean,logvar,beta,"poisson")
            # if tf.math.is_nan(gen_total_loss).numpy()==True:
            #     print('fuck')
                


    
      fev=fraction_of_explained_variance(obs_rate, est_rate)
      fev=K.mean(fev)
      corr=K.mean(correlation_coefficient(obs_rate,est_rate))
      
      
      encoder_gradients = enc_tape.gradient(gen_total_loss,
                                              encoder.trainable_variables)
      encoder_optimizer.apply_gradients(zip(encoder_gradients,
                                              encoder.trainable_variables))
      
      
      decoder_gradients = dec_tape.gradient(gen_total_loss,
                                              decoder.trainable_variables)

      decoder_optimizer.apply_gradients(zip(decoder_gradients,
                                              decoder.trainable_variables))
      
      
      fwd_gradients = fwd_tape.gradient(gen_total_loss,
                                              generator.trainable_variables)

      generator_optimizer.apply_gradients(zip(fwd_gradients,
                                              generator.trainable_variables))
      # if epoch%3==0:
      #     plt.plot(mean[:,0].numpy())
      with tf.name_scope('fwd/train'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-total-train', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-train', loss_rec, step=epoch)
            tf.summary.scalar('loss-rec-norm-train', loss_rec_norm, step=epoch)
            tf.summary.scalar('loss-KL-train', loss_kl, step=epoch)
            tf.summary.scalar('loss-KL-norm-train', loss_kl_norm, step=epoch)
            
            # tf.summary.scalar('fraction',fev,step=epoch)
            tf.summary.scalar('corr',corr,step=epoch)
            tf.summary.image('stimuli_train', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('tnsf-stimuli-train', tf.transpose(tsnf_stimuli, perm=[0,2,3,1]),step=epoch)

        
   elif train_what=='val':

      mean_log_var= encoder(stimuli, training=False)
      mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
      z=reparameterize(mean,logvar)
           
      tsnf_stimuli=decoder(z, training=False)
           
      est_rate=  generator(tsnf_stimuli, training=False)
      est_rate=est_rate +1e-4
      obs_rate=spikes
        
      gen_total_loss,loss_rec,loss_kl ,loss_rec_norm,loss_kl_norm= vae_loss(obs_rate,est_rate,z,mean,logvar,beta,"poisson")

      fev=fraction_of_explained_variance(obs_rate, est_rate)
      fev=K.mean(fev)
      corr=K.mean(correlation_coefficient(obs_rate,est_rate))
      
      with tf.name_scope('fwd/val'):
          with summary_writer.as_default():
            tf.summary.scalar('loss-total-val', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-val', loss_rec, step=epoch)
            tf.summary.scalar('loss-rec-norm-val', loss_rec_norm, step=epoch)
            tf.summary.scalar('loss-KL-val', loss_kl, step=epoch)
            tf.summary.scalar('loss-KL-norm-val', loss_kl_norm, step=epoch)
            # tf.summary.scalar('fraction_val',fev,step=epoch)
            tf.summary.scalar('corr_val',corr,step=epoch)
            tf.summary.image('stimuli_val', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('tnsf-stimuli-val', tf.transpose(tsnf_stimuli, perm=[0,2,3,1]),step=epoch)
   elif train_what=='test':

      mean_log_var= encoder(stimuli, training=False)
      mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
      z=reparameterize(mean,logvar)
           
      tsnf_stimuli=decoder(z, training=False)
           
      est_rate=  generator(tsnf_stimuli, training=False)
      est_rate=est_rate +1e-4
      obs_rate=spikes
        
      gen_total_loss,loss_rec,loss_kl ,loss_rec_norm,loss_kl_norm= vae_loss(obs_rate,est_rate,z,mean,logvar,beta,"poisson")

      fev=fraction_of_explained_variance(obs_rate, est_rate)
      fev=K.mean(fev)
      corr=K.mean(correlation_coefficient(obs_rate,est_rate))
      # if epoch%3==0:
      #     plt.plot(mean[:,0].numpy())
      #     plt.show()
      with tf.name_scope('fwd/test'):
          with summary_writer.as_default():
            tf.summary.scalar('loss-total-test', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-test', loss_rec, step=epoch)
            tf.summary.scalar('loss-rec-norm-test', loss_rec_norm, step=epoch)
            tf.summary.scalar('loss-KL-test', loss_kl, step=epoch)
            tf.summary.scalar('loss-KL-norm-test', loss_kl_norm, step=epoch)
            # tf.summary.scalar('fraction_test',fev,step=epoch)
            tf.summary.scalar('corr_test',corr,step=epoch)
            tf.summary.image('stimuli_test', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('tnsf-stimuli-test', tf.transpose(tsnf_stimuli, perm=[0,2,3,1]),step=epoch)
            
            
   return est_rate,obs_rate,gen_total_loss,corr


# @tf.function
def test_forward(stimuli, spikes, epoch,train_what,encoder,decoder,generator,beta):
    

      mean_log_var= encoder(stimuli, training=False)
      mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
      # z=reparameterize(mean,logvar)
      z=tf.cast(tf.convert_to_tensor(np.random.normal(size=(mean.shape[0],mean.shape[1]))),dtype=tf.float32)

      tsnf_stimuli=decoder(z, training=False)
           
      est_rate =  generator(tsnf_stimuli, training=False)
      est_rate=est_rate +1e-4
      obs_rate=spikes
        
      gen_total_loss,loss_rec,loss_kl,loss_rec_norm,loss_kl_norm = vae_loss(obs_rate,est_rate,z,mean,logvar,beta,"poisson")

      fev=fraction_of_explained_variance(obs_rate, est_rate)
      fev=K.mean(fev)
      corr=K.mean(correlation_coefficient(obs_rate,est_rate))
      
      if epoch % 3 ==0:
          plt.figure()
          plt.plot(est_rate[:,0].numpy())
          plt.title('{}'.format(epoch))
          plt.show()
      # if epoch%3==0:
      #     plt.plot(mean[:,0].numpy())
      #     plt.show()
      with tf.name_scope('fwd/generation'):
          with summary_writer.as_default():
            tf.summary.scalar('loss-total-gen', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-gen', loss_rec, step=epoch)
            tf.summary.scalar('loss-KL-gen', loss_kl, step=epoch)
            # tf.summary.scalar('fraction_test',fev,step=epoch)
            tf.summary.scalar('corr_gen',corr,step=epoch)
            tf.summary.image('stimuli_gen', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
       

      return est_rate,obs_rate,gen_total_loss,corr

# @tf.function
def train_backward(stimuli,spikes, epoch,train_what,actor,actor_optimizer,generator):

   if train_what=='train':
         
      with tf.GradientTape() as gen_tape:
            tsnf_stimuli= actor(stimuli, training=True)
            # tsnf_stimuli=tf.transpose(tsnf_stimuli,perm=[0,2,3,1])
            # tsnf_stimuli=tf.image.resize(tsnf_stimuli,(stimuli.shape[2],stimuli.shape[3]),method='bicubic')
            # tsnf_stimuli=tf.transpose(tsnf_stimuli,perm=[0,3,1,2])
            est_rate= generator(tsnf_stimuli, training=False)
            est_rate=est_rate +1e-4
            obs_rate=spikes
            
            gen_total_loss = generator_loss(obs_rate,est_rate,"poisson")
            gen_total_loss=tf.math.reduce_mean(gen_total_loss)
    
      fev=fraction_of_explained_variance(obs_rate, est_rate)
      fev=K.mean(fev)
      corr=K.mean(correlation_coefficient(obs_rate,est_rate))
      
      actor_gradients = gen_tape.gradient(gen_total_loss,
                                              actor.trainable_variables)
      actor_optimizer.apply_gradients(zip(actor_gradients,
                                              actor.trainable_variables))
      with tf.name_scope('actor/train'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-train-tnsf', gen_total_loss, step=epoch)
            # tf.summary.scalar('fraction-tnsf',fev,step=epoch)
            tf.summary.scalar('corr-tnsf',corr,step=epoch)
            # tf.summary.scalar('lr',generator_optimizer.lr,step=epoch)
            tf.summary.image('stimuli_train-tnsf', tf.transpose(tsnf_stimuli,[0,2,3,1]),step=epoch)
            tf.summary.image('stimuli_train', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
        
   elif train_what=='val':
      
      tsnf_stimuli= actor(stimuli, training=False)
      # tsnf_stimuli=tf.transpose(tsnf_stimuli,perm=[0,2,3,1])
      # tsnf_stimuli=stimuli
      # tsnf_stimuli=tf.image.resize(tsnf_stimuli,(stimuli.shape[2],stimuli.shape[3]),method='nearest')
      # tsnf_stimuli=tf.transpose(tsnf_stimuli,perm=[0,3,1,2])
      est_rate= generator(tsnf_stimuli, training=False)
      est_rate=est_rate +1e-4
      obs_rate=spikes
      gen_total_loss = generator_loss(obs_rate,est_rate,"poisson")
      gen_total_loss=tf.math.reduce_mean(gen_total_loss)
    
      fev=fraction_of_explained_variance(obs_rate, est_rate)
      fev=K.mean(fev)
      corr=K.mean(correlation_coefficient(obs_rate,est_rate))
      with tf.name_scope('actor/val'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-val-tnsf', gen_total_loss, step=epoch)
            # tf.summary.scalar('fraction_val-tnsf',fev,step=epoch)
            tf.summary.scalar('corr_val-tnsf',corr,step=epoch)
            # tf.summary.scalar('lr_val',generator_optimizer.lr,step=epoch)
            tf.summary.image('stimuli_val-tnsf', tf.transpose(tsnf_stimuli,[0,2,3,1]),step=epoch)
            tf.summary.image('stimuli_val', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
            
   elif train_what=='test':
      
      tsnf_stimuli= actor(stimuli, training=False)
      # tsnf_stimuli=tf.transpose(tsnf_stimuli,perm=[0,2,3,1])
      # tsnf_stimuli=stimuli
      # tsnf_stimuli=tf.image.resize(tsnf_stimuli,(stimuli.shape[2],stimuli.shape[3]),method='nearest')
      # tsnf_stimuli=tf.transpose(tsnf_stimuli,perm=[0,3,1,2])
      est_rate= generator(tsnf_stimuli, training=False)
      est_rate=est_rate +1e-4
      obs_rate=spikes
      gen_total_loss = generator_loss(obs_rate,est_rate,"poisson")
      gen_total_loss=tf.math.reduce_mean(gen_total_loss)
    
      fev=fraction_of_explained_variance(obs_rate, est_rate)
      fev=K.mean(fev)
      corr=K.mean(correlation_coefficient(obs_rate,est_rate))
      with tf.name_scope('actor/test'):
        with summary_writer.as_default():
            tf.summary.scalar('loss-test-tnsf', gen_total_loss, step=epoch)
            # tf.summary.scalar('fraction_test-tnsf',fev,step=epoch)
            tf.summary.scalar('corr_test-tnsf',corr,step=epoch)
            # tf.summary.scalar('lr_val',generator_optimizer.lr,step=epoch)
            tf.summary.image('stimuli_test-tnsf', tf.transpose(tsnf_stimuli,[0,2,3,1]),step=epoch)
            tf.summary.image('stimuli_test', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
    
   return est_rate,obs_rate,gen_total_loss,corr

@tf.function
def train_actor_vae(stimuli,spikes, epoch,train_what,actor_vae,actor_vae_optimizer,\
                                                                                  
                                                                                                  encoder,decoder,generator,beta):
   if train_what=='train':
       
            gen_total_loss=0
            obs_rate=spikes
            
            with tf.GradientTape() as gen_tape:
                  tsnf_stimuli= actor_vae(stimuli, training=True)
                  mean_log_var= encoder(tsnf_stimuli, training=False)
                  mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
                  z=reparameterize(mean,logvar)
      

                  tsnf_stimuli_2=decoder(z, training=False)
                  
                  est_rate= generator(tsnf_stimuli_2, training=False)
                  est_rate=est_rate +1e-4
                  

                        
                  fev_temp=fraction_of_explained_variance(obs_rate, est_rate)
                  fev=K.mean(fev_temp)
                  corr=K.mean(correlation_coefficient(obs_rate,est_rate))

                        
                  gen_total_loss_temp = generator_loss(obs_rate,est_rate,"poisson")
                  gen_total_loss= tf.math.reduce_mean(gen_total_loss_temp)
            

        
        
            actor_vae_gradients = gen_tape.gradient(gen_total_loss,
                                                      actor_vae.trainable_variables)
            actor_vae_optimizer.apply_gradients(zip(actor_vae_gradients,
                                                      actor_vae.trainable_variables))
            
            with tf.name_scope('VAE/train'):                       
                with summary_writer.as_default():
                  tf.summary.scalar('corr-train-tnsf',corr,step=epoch)
                  tf.summary.image('stimuli_train-tnsf', tf.transpose(tsnf_stimuli,[0,2,3,1]),step=epoch)
                  tf.summary.image('stimuli_train', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
                  tf.summary.scalar('loss-train-tnsf', gen_total_loss, step=epoch)

    

        
   elif train_what=='val':
  
            gen_total_loss=0
            obs_rate=spikes

            tsnf_stimuli= actor_vae(stimuli, training=False)
            mean_log_var= encoder(tsnf_stimuli, training=False)
            mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
            z=reparameterize(mean,logvar)
      

            tsnf_stimuli_2=decoder(z, training=False)
                  
            est_rate= generator(tsnf_stimuli_2, training=False)
            est_rate=est_rate +1e-4
                  

                        
            fev_temp=fraction_of_explained_variance(obs_rate, est_rate)
            fev=K.mean(fev_temp)
            corr=K.mean(correlation_coefficient(obs_rate,est_rate))

                        
            gen_total_loss_temp = generator_loss(obs_rate,est_rate,"poisson")
            gen_total_loss= tf.math.reduce_mean(gen_total_loss_temp)

            
            with tf.name_scope('VAE/val'):                       
                with summary_writer.as_default():
                  tf.summary.scalar('corr-val-tnsf',corr,step=epoch)
                  tf.summary.image('stimuli_val-tnsf', tf.transpose(tsnf_stimuli,[0,2,3,1]),step=epoch)
                  tf.summary.image('stimuli_val', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
                  tf.summary.scalar('loss-val-tnsf', gen_total_loss, step=epoch)

   elif train_what=='test':
     
      
            gen_total_loss=0
            obs_rate=spikes

            tsnf_stimuli= actor_vae(stimuli, training=False)
            mean_log_var= encoder(tsnf_stimuli, training=False)
            mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
            z=reparameterize(mean,logvar)
      

            tsnf_stimuli_2=decoder(z, training=False)
                  
            est_rate= generator(tsnf_stimuli_2, training=False)
            est_rate=est_rate +1e-4
                  

                        
            fev_temp=fraction_of_explained_variance(obs_rate, est_rate)
            fev=K.mean(fev_temp)
            corr=K.mean(correlation_coefficient(obs_rate,est_rate))

                        
            gen_total_loss_temp = generator_loss(obs_rate,est_rate,"poisson")
            gen_total_loss= tf.math.reduce_mean(gen_total_loss_temp)

            
            with tf.name_scope('VAE/test'):                       
                with summary_writer.as_default():
                  tf.summary.scalar('corr-test-tnsf',corr,step=epoch)
                  tf.summary.image('stimuli_test-tnsf', tf.transpose(tsnf_stimuli,[0,2,3,1]),step=epoch)
                  tf.summary.image('stimuli_test', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
                  tf.summary.scalar('loss-test-tnsf', gen_total_loss, step=epoch)

          
   return est_rate,obs_rate,gen_total_loss,corr



def actor_vae_response(stimuli,spikes,encoder,decoder,generator,beta,checkpoint_dir,
                             f_stimuli,f_spikes,f_latent,
                             HEIGHT_stimuli,WIDTH_stimuli,WIDTH_transformed,HEIGHT_transformed,noise):
    
    actor_vae= Actor_vae(HEIGHT_stimuli,WIDTH_stimuli,WIDTH_transformed,HEIGHT_transformed)


    checkpoint=tf.train.Checkpoint(actor_vae=actor_vae)
    # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('latest checlpoint: {}'.format(latest))
    status=checkpoint.restore(latest)
    # status.assert_consumed()  # Optional sanity checks.
    ACV_test=checkpoint.actor_vae

    
    gen_total_loss=0
    obs_rate=spikes

    tsnf_stimuli = ACV_test(stimuli, training=False)
    mean_log_var= encoder(tsnf_stimuli, training=False)
    mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)

    z=reparameterize(mean,logvar,noise)
      

    tsnf_stimuli_2=decoder(z, training=False)
                  
    est_rate= generator(tsnf_stimuli_2, training=False)
    est_rate=est_rate +1e-4
                  

                        
    fev_temp=fraction_of_explained_variance(obs_rate, est_rate)
    fev=K.mean(fev_temp)
    corr=K.mean(correlation_coefficient(obs_rate,est_rate))

                        
    gen_total_loss_temp = generator_loss(obs_rate,est_rate,"poisson")
    gen_total_loss= tf.math.reduce_mean(gen_total_loss_temp)
    
    if noise!=None:
      z.numpy().astype(np.float32).tofile(f_latent)
      
    tsnf_stimuli.numpy().astype(np.float32).tofile(f_stimuli)
    est_rate.numpy().astype(np.float32).tofile(f_spikes)

    return gen_total_loss,corr,fev



def actor_vae_response_sampling(stimuli,spikes,encoder,decoder,generator,beta,checkpoint_dir,   
                                    HEIGHT_stimuli,WIDTH_stimuli,WIDTH_transformed,HEIGHT_transformed,noise):
    
    actor_vae= Actor_vae(HEIGHT_stimuli,WIDTH_stimuli,WIDTH_transformed,HEIGHT_transformed)


    checkpoint=tf.train.Checkpoint(actor_vae=actor_vae)
    # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('latest checlpoint: {}'.format(latest))
    status=checkpoint.restore(latest)
    # status.assert_consumed()  # Optional sanity checks.
    ACV_test=checkpoint.actor_vae

    
    gen_total_loss=0
    obs_rate=spikes

    tsnf_stimuli = ACV_test(stimuli, training=False)
    mean_log_var= encoder(tsnf_stimuli, training=False)
    mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)

    z=reparameterize(mean,logvar,noise)
      

    tsnf_stimuli_2=decoder(z, training=False)
                  
    est_rate= generator(tsnf_stimuli_2, training=False)
    est_rate=est_rate +1e-4
                  

                        
    fev_temp=fraction_of_explained_variance(obs_rate, est_rate)
    fev=K.mean(fev_temp)
    corr=K.mean(correlation_coefficient(obs_rate,est_rate))

                        
    gen_total_loss_temp = generator_loss(obs_rate,est_rate,"poisson")
    gen_total_loss= tf.math.reduce_mean(gen_total_loss_temp)


    return gen_total_loss,corr,fev,z,mean,logvar

# @tf.function
def response(stimuli,spikes, tsnf_stimuli,encoder):

   est_rate= encoder(tsnf_stimuli, training=False)
  
   est_rate=est_rate +1e-4
   obs_rate=spikes
            
   gen_total_loss = generator_loss(obs_rate,est_rate,"poisson")
   gen_total_loss=tf.math.reduce_mean(gen_total_loss)
    
   fev=fraction_of_explained_variance(obs_rate, est_rate)
   fev=K.mean(fev)
   corr=K.mean(correlation_coefficient(obs_rate,est_rate))
      
    
   return gen_total_loss,corr,fev

# @tf.function
def response_vae(stimuli,spikes,encoder,decoder,generator,beta):

      mean_log_var= encoder(stimuli, training=False)
      mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
      z=reparameterize(mean,logvar)
      
      # for i in range(z.shape[1].numpy()):
      #     plt.figure()
      #     plt.plot(z[:,i].numpy())
      #     plt.title('z_{}'.format(i))
      #     plt.show()
          
      tsnf_stimuli=decoder(z, training=False)
           
      est_rate =  generator(tsnf_stimuli, training=False)
      est_rate=est_rate +1e-4
      obs_rate=spikes
        
      gen_total_loss,loss_rec,loss_kl,loss_rec_norm,loss_kl_norm = vae_loss(obs_rate,est_rate,z,mean,logvar,beta,"poisson")

      fev=fraction_of_explained_variance(obs_rate, est_rate)
      fev=K.mean(fev)
      corr=K.mean(correlation_coefficient(obs_rate,est_rate))
      
      # for i in range(est_rate.shape[1].numpy()):
      #     plt.figure()
      #     plt.plot(est_rate[:,i].numpy())
      #     plt.title('est_rate{}'.format(i))
      #     plt.show()


 
      return gen_total_loss,corr,fev,z,mean,logvar