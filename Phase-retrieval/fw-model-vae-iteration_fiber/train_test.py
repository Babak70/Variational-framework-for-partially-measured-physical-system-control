import tensorflow as tf
import tensorflow.keras.backend as K
import datetime
import matplotlib.pyplot as plt
from metrics import fraction_of_explained_variance,correlation_coefficient
from experiment import send_out,run_exp

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
           
            est_rate =  generator(tsnf_stimuli, training=True)
            obs_rate = spikes
        
            gen_total_loss,loss_rec,loss_kl = vae_loss(obs_rate,est_rate,z,mean,logvar,beta,"mse")
            # if tf.math.is_nan(gen_total_loss).numpy()==True:
            #     print('fuck')
                
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
            tf.summary.scalar('loss-KL-train', loss_kl, step=epoch)
            
            # tf.summary.scalar('fraction',fev,step=epoch)
            tf.summary.scalar('corr',corr,step=epoch)
            tf.summary.image('input model_train', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('output model_train', tf.transpose(est_rate, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('output GT_train', tf.transpose(obs_rate, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('tnsf-input model-train', tf.transpose(tsnf_stimuli, perm=[0,2,3,1]),step=epoch)

        
   elif train_what=='val':

      mean_log_var= encoder(stimuli, training=False)
      mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
      z=reparameterize(mean,logvar)
           
      tsnf_stimuli=decoder(z, training=False)
           
      est_rate=  generator(tsnf_stimuli, training=False)
      obs_rate=spikes
        
      gen_total_loss,loss_rec,loss_kl = vae_loss(obs_rate,est_rate,z,mean,logvar,beta,"mse")

      fev=fraction_of_explained_variance(obs_rate, est_rate)
      fev=K.mean(fev)
      corr=K.mean(correlation_coefficient(obs_rate,est_rate))
      
      with tf.name_scope('fwd/val'):
          with summary_writer.as_default():
            tf.summary.scalar('loss-total-val', gen_total_loss, step=epoch)
            tf.summary.scalar('loss-rec-val', loss_rec, step=epoch)
            tf.summary.scalar('loss-KL-val', loss_kl, step=epoch)
            # tf.summary.scalar('fraction_val',fev,step=epoch)
            tf.summary.scalar('corr_val',corr,step=epoch)
            tf.summary.image('input model_val', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('output model_val', tf.transpose(est_rate, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('output GT_val', tf.transpose(obs_rate, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('tnsf-input model-val', tf.transpose(tsnf_stimuli, perm=[0,2,3,1]),step=epoch)
   elif train_what=='test':

      mean_log_var= encoder(stimuli, training=False)
      mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
      z=reparameterize(mean,logvar)
           
      tsnf_stimuli=decoder(z, training=False)
           
      est_rate=  generator(tsnf_stimuli, training=False)
      obs_rate=spikes
        
      gen_total_loss,loss_rec,loss_kl = vae_loss(obs_rate,est_rate,z,mean,logvar,beta,"mse")

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
            tf.summary.scalar('loss-KL-test', loss_kl, step=epoch)
            # tf.summary.scalar('fraction_test',fev,step=epoch)
            tf.summary.scalar('corr_test',corr,step=epoch)
            tf.summary.image('input model_test', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('output model_test', tf.transpose(est_rate, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('output GT_test', tf.transpose(obs_rate, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('tnsf-input model-test', tf.transpose(tsnf_stimuli, perm=[0,2,3,1]),step=epoch)
            
            
   return est_rate,obs_rate,gen_total_loss,corr


# @tf.function
def test_forward(stimuli, spikes, epoch,train_what,encoder,decoder,generator,beta):
    

      mean_log_var= encoder(stimuli, training=False)
      mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
      # z=reparameterize(mean,logvar)
      z=tf.cast(tf.convert_to_tensor(np.random.normal(size=(mean.shape[0],mean.shape[1]))),dtype=tf.float32)

      tsnf_stimuli=decoder(z, training=False)
           
      est_rate =  generator(tsnf_stimuli, training=False)
      obs_rate=spikes
        
      gen_total_loss,loss_rec,loss_kl = vae_loss(obs_rate,est_rate,z,mean,logvar,beta,"mse")

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



@tf.function
def train_actor_vae(stimuli,spikes,targets, epoch,train_what,actor_vae,actor_vae_optimizer,\
                                                                                    encoder,decoder,generator,beta):

       
   if train_what=='train':
       
            gen_total_loss=0
            obs_rate=targets
            
            with tf.GradientTape() as gen_tape:
                  tsnf_stimuli= actor_vae(targets, training=True)
                  mean_log_var= encoder(tsnf_stimuli, training=False)
                  mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
                  z=reparameterize(mean,logvar)
      

                  tsnf_stimuli_2=decoder(z, training=False)
                  
                  est_rate= generator(tsnf_stimuli_2, training=False)


                  corr=K.mean(correlation_coefficient(obs_rate,est_rate))

                        
                  # gen_total_loss_temp = generator_loss(obs_rate,est_rate,"corr")
                  # gen_total_loss= -tf.math.log((tf.math.reduce_mean(gen_total_loss_temp)+1.)/2.)
                  
                  
                  gen_total_loss_temp = generator_loss(obs_rate,est_rate,"mse")
                  gen_total_loss= (tf.math.reduce_mean(gen_total_loss_temp))

            actor_vae_gradients = gen_tape.gradient(gen_total_loss,
                                                      actor_vae.trainable_variables)
            actor_vae_optimizer.apply_gradients(zip(actor_vae_gradients,
                                                      actor_vae.trainable_variables))
            
            with tf.name_scope('VAE/train'):                       
                with summary_writer.as_default():
                  tf.summary.scalar('corr-train-tnsf',corr,step=epoch)
                  tf.summary.image('tnsf-output actor-train', tf.transpose(tsnf_stimuli,[0,2,3,1]),step=epoch)
                  tf.summary.image('targets_train', tf.transpose(obs_rate, perm=[0,2,3,1]),step=epoch)
                  tf.summary.image('output model_train', tf.transpose(est_rate, perm=[0,2,3,1]),step=epoch)
                  tf.summary.scalar('loss-train-tnsf', gen_total_loss, step=epoch)

    

        
   elif train_what=='val':
  
            gen_total_loss=0
            obs_rate=targets

            tsnf_stimuli= actor_vae(targets, training=False)
            mean_log_var= encoder(tsnf_stimuli, training=False)
            mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
            z=reparameterize(mean,logvar)
      

            tsnf_stimuli_2=decoder(z, training=False)
                  
            est_rate= generator(tsnf_stimuli_2, training=False)
                  

            corr=K.mean(correlation_coefficient(obs_rate,est_rate))

            # gen_total_loss_temp = generator_loss(obs_rate,est_rate,"corr")
            # gen_total_loss= -tf.math.log((tf.math.reduce_mean(gen_total_loss_temp)+1.)/2.)
                  
                  
            gen_total_loss_temp = generator_loss(obs_rate,est_rate,"mse")
            gen_total_loss= (tf.math.reduce_mean(gen_total_loss_temp))

            
            with tf.name_scope('VAE/val'):                       
                with summary_writer.as_default():
                  tf.summary.scalar('corr-val-tnsf',corr,step=epoch)
                  tf.summary.image('tnsf-output actor-val', tf.transpose(tsnf_stimuli,[0,2,3,1]),step=epoch)
                  tf.summary.image('targets_val', tf.transpose(obs_rate, perm=[0,2,3,1]),step=epoch)
                  tf.summary.image('output model_val', tf.transpose(est_rate, perm=[0,2,3,1]),step=epoch)
                  tf.summary.scalar('loss-val-tnsf', gen_total_loss, step=epoch)

   elif train_what=='test':
     
      
            gen_total_loss=0
            obs_rate=targets

            tsnf_stimuli= actor_vae(targets, training=False)
            mean_log_var= encoder(tsnf_stimuli, training=False)
            mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
            z=reparameterize(mean,logvar)
      

            tsnf_stimuli_2=decoder(z, training=False)
                  
            est_rate= generator(tsnf_stimuli_2, training=False)

            corr=K.mean(correlation_coefficient(obs_rate,est_rate))

                        
            # gen_total_loss_temp = generator_loss(obs_rate,est_rate,"corr")
            # gen_total_loss= -tf.math.log((tf.math.reduce_mean(gen_total_loss_temp)+1.)/2.)
                  
                  
            gen_total_loss_temp = generator_loss(obs_rate,est_rate,"mse")
            gen_total_loss= (tf.math.reduce_mean(gen_total_loss_temp))

            
            with tf.name_scope('VAE/test'):                       
                with summary_writer.as_default():
                  tf.summary.scalar('corr-test-tnsf',corr,step=epoch)
                  tf.summary.image('tnsf-output actor-test', tf.transpose(tsnf_stimuli,[0,2,3,1]),step=epoch)
                  tf.summary.image('targets_test', tf.transpose(obs_rate, perm=[0,2,3,1]),step=epoch)
                  tf.summary.image('output model_test', tf.transpose(est_rate, perm=[0,2,3,1]),step=epoch)
                  tf.summary.scalar('loss-test-tnsf', gen_total_loss, step=epoch)

          
   return est_rate,obs_rate,gen_total_loss,corr



def actor_vae_response(stimuli,spikes,targets,
                          beta,checkpoint_dir,
                                 f_stimuli,fwd_dir,file_to_use,
                                     HEIGHT_stimuli,WIDTH_stimuli,WIDTH_transformed,HEIGHT_transformed):
    
    actor_vae= Actor_vae(HEIGHT_stimuli,WIDTH_stimuli,WIDTH_transformed,HEIGHT_transformed)


    checkpoint=tf.train.Checkpoint(actor_vae=actor_vae)
    # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('latest checlpoint: {}'.format(latest))
    status=checkpoint.restore(latest)
    # status.assert_consumed()  # Optional sanity checks.
    ACV_test=checkpoint.actor_vae


    tsnf_stimuli= ACV_test(targets, training=False)
   
    tf.transpose(tsnf_stimuli,[1,0,3,2]).numpy().astype(np.float32).tofile(f_stimuli)# it is important to make the shape back in x and y for the next iteration
    

    send_out(tsnf_stimuli,'stimuli',fwd_dir)
    send_out(targets,'targets',fwd_dir)
    mse_out,corr_out=run_exp(fwd_dir,file_to_use)

    # est_rate.numpy().astype(np.float32).tofile(f_spikes)


    return mse_out,corr_out
# @tf.function
def response(stimuli,spikes, tsnf_stimuli,encoder):

   est_rate= encoder(tsnf_stimuli, training=False)
  
   est_rate=est_rate +1e-4
   obs_rate=spikes
            
   gen_total_loss = generator_loss(obs_rate,est_rate,"mse")
   gen_total_loss=tf.math.reduce_mean(gen_total_loss)
    
   fev=fraction_of_explained_variance(obs_rate, est_rate)
   fev=K.mean(fev)
   corr=K.mean(correlation_coefficient(obs_rate,est_rate))
      
    
   return gen_total_loss,corr

# @tf.function
def response_vae(stimuli,spikes,encoder,decoder,generator,beta):

      mean_log_var= encoder(stimuli, training=False)
      mean,logvar=tf.split(mean_log_var, num_or_size_splits=2, axis=1)
      z=reparameterize(mean,logvar)
      
      for i in range(z.shape[1].numpy()):
          plt.figure()
          plt.plot(z[:,i].numpy())
          plt.title('z_{}'.format(i))
          plt.show()
      tsnf_stimuli=decoder(z, training=False)
           
      est_rate =  generator(tsnf_stimuli, training=False)
      obs_rate=spikes
        
      gen_total_loss,loss_rec,loss_kl = vae_loss(obs_rate,est_rate,z,mean,logvar,beta,"mse")

      corr=K.mean(correlation_coefficient(obs_rate,est_rate))
      for i in range(est_rate.shape[1].numpy()):
          plt.figure()
          plt.plot(est_rate[:,i].numpy())
          plt.title('est_rate{}'.format(i))
          plt.show()


 
      return gen_total_loss,corr