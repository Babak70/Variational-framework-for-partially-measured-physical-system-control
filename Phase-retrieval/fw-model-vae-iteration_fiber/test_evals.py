from matplotlib import pyplot as plt
import tensorflow as tf
from dim_reduction_utils import pca_transform,auto_encoder_transform,actor_transform
from train_test import test_forward,response,response_vae,actor_vae_response
import numpy as np
import os

def test_pca(test_ds,checkpoint,checkpoint_dir,n_component):
    
    # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('latest checlpoint: {}'.format(latest))
    status=checkpoint.restore(latest)
    # status.assert_consumed()  # Optional sanity checks.
    encoder_test=checkpoint.encoder


    running_loss = 0
    running_corr = 0
    running_fev = 0
    running_size = 0
    corr_all=[]


    for n, (stimuli, spikes) in test_ds.enumerate():
                  
               # if n.numpy()!=0:
               #     continue
               
               stimuli=tf.transpose(stimuli,perm=[1,0,2,3])
               tnsf_stimuli=pca_transform(tf.reshape(tf.squeeze(stimuli),[-1,stimuli.shape[2]*stimuli.shape[3]]).numpy(), n_component)
               tnsf_stimuli=tf.convert_to_tensor(tnsf_stimuli)
               tnsf_stimuli=tf.reshape(tnsf_stimuli,[-1,stimuli.shape[2],stimuli.shape[3]])
               tnsf_stimuli=tf.expand_dims(tnsf_stimuli, axis=1)
               spikes=tf.squeeze(spikes,axis=[0,3])
               loss_t,corr_t,fev_t= response(stimuli, spikes,tnsf_stimuli,encoder_test)
               corr_all.append(corr_t.numpy())
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr += corr_t.numpy()
               running_fev += fev_t.numpy()
               running_size += 1
    
    running_loss /= running_size
    running_corr /= running_size
    running_fev /= running_size

    return running_loss,running_corr,running_fev,corr_all


def test_auto_encoder(test_ds,checkpoint,checkpoint_dir,checkpoint_dir_AE,
                      HEIGHT_stimuli=50,WIDTH_stimuli=50,WIDTH_transformed=50,HEIGHT_transformed=50):
    
    # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('latest checlpoint: {}'.format(latest))
    status=checkpoint.restore(latest)
    # status.assert_consumed()  # Optional sanity checks.
    encoder_test=checkpoint.encoder


    running_loss = 0
    running_corr = 0
    running_fev = 0
    running_size = 0

    for n, (stimuli, spikes) in test_ds.enumerate():
               if n.numpy()!=0:
                   continue
               
               stimuli=tf.transpose(stimuli,perm=[1,0,2,3])
               tnsf_stimuli=auto_encoder_transform(stimuli,checkpoint_dir_AE,
                                                   HEIGHT_stimuli,WIDTH_stimuli,
                                                   WIDTH_transformed,HEIGHT_transformed)
               # tnsf_stimuli=stimuli
               spikes=tf.squeeze(spikes,axis=[0,3])
               loss_t,corr_t,fev_t= response(stimuli, spikes,tnsf_stimuli,encoder_test)

               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr += corr_t.numpy()
               running_fev += fev_t.numpy()
               running_size += 1
    
    running_loss /= running_size
    running_corr /= running_size
    running_fev /= running_size


    return running_loss,running_corr,running_fev



def test_actor(test_ds,checkpoint,checkpoint_dir,checkpoint_dir_actor,
                      HEIGHT_stimuli=50,WIDTH_stimuli=50,WIDTH_transformed=50,HEIGHT_transformed=50):
    
    # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('latest checlpoint: {}'.format(latest))
    status=checkpoint.restore(latest)
    # status.assert_consumed()  # Optional sanity checks.
    encoder_test=checkpoint.encoder


    running_loss = 0
    running_corr = 0
    running_fev = 0
    running_size = 0
    corr_all=[]

    for n, (stimuli, spikes) in test_ds.enumerate():
               # if n.numpy()!=13:
               #       continue
               
               stimuli=tf.transpose(stimuli,perm=[1,0,2,3])
               tnsf_stimuli=actor_transform(stimuli,checkpoint_dir_actor,
                                                     HEIGHT_stimuli,WIDTH_stimuli,
                                                     WIDTH_transformed,HEIGHT_transformed)

                      
               # tnsf_stimuli=stimuli
               spikes=tf.squeeze(spikes,axis=[0,3])
               loss_t,corr_t,fev_t= response(stimuli, spikes,tnsf_stimuli,encoder_test)
               corr_all.append(corr_t.numpy())
               # print(n.numpy())
               # print(corr_t.numpy())

               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr += corr_t.numpy()
               running_fev += fev_t.numpy()
               running_size += 1
               
               
               if n.numpy()==0:
                     for ll in range(1000):
                       fig = plt.figure(figsize=(1, 2))
                       plt.subplot(1, 2, 1)
                       plt.imshow(stimuli[ll, 0,:, :].numpy(), cmap='gray')
                       plt.axis('off')
                       plt.subplot(1, 2, 2)
                       plt.imshow(tnsf_stimuli[ll, 0,:, :].numpy(), cmap='gray')
                       plt.axis('off')
                       plt.savefig('images/actor/images{}.png'.format(ll+1))
    
    running_loss /= running_size
    running_corr /= running_size
    running_fev /= running_size


    return running_loss,running_corr,running_fev,corr_all

def test_no_transform(test_ds,checkpoint,checkpoint_dir):
                     
    
    
    # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('latest checlpoint: {}'.format(latest))
    status=checkpoint.restore(latest)
    # status.assert_consumed()  # Optional sanity checks.
    encoder_test=checkpoint.encoder


    running_loss = 0
    running_corr = 0
    running_fev = 0
    running_size = 0
    corr_all=[]

    for n, (stimuli, spikes) in test_ds.enumerate():
               
               stimuli=tf.transpose(stimuli,perm=[1,0,3,2])

               # tnsf_stimuli=stimuli
               spikes=tf.squeeze(spikes,axis=[0,3])
               loss_t,corr_t,fev_t= response(stimuli, spikes,stimuli,encoder_test)
               corr_all.append(corr_t.numpy())

               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr += corr_t.numpy()
               running_fev += fev_t.numpy()
               running_size += 1
    
    running_loss /= running_size
    running_corr /= running_size
    running_fev /= running_size


    return running_loss,running_corr,running_fev,corr_all

def test_no_transform_vae(test_ds,checkpoint,checkpoint_dir,beta):
                     
    
    
    # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('latest checlpoint: {}'.format(latest))
    status=checkpoint.restore(latest)
    # status.assert_consumed()  # Optional sanity checks.
    encoder_test=checkpoint.encoder
    decoder_test=checkpoint.decoder
    generator_test=checkpoint.generator


    running_loss = 0
    running_corr = 0
    running_fev = 0
    running_size = 0
    corr_all=[]

    for n, (stimuli, spikes) in test_ds.enumerate():
               
               stimuli=tf.transpose(stimuli,perm=[1,0,3,2])

               # tnsf_stimuli=stimuli
               spikes=tf.squeeze(spikes,axis=[0,3])
               loss_t,corr_t,fev_t= response_vae(stimuli, spikes,encoder_test,decoder_test,generator_test,beta)
               corr_all.append(corr_t.numpy())

               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr += corr_t.numpy()
               running_fev += fev_t.numpy()
               running_size += 1
    
    running_loss /= running_size
    running_corr /= running_size
    running_fev /= running_size


    return running_loss,running_corr,running_fev,corr_all




def test_actor_vae(test_ds,fwd_dir,file_to_use,checkpoint_dir_actor_vae,
                      beta,HEIGHT_stimuli=100,WIDTH_stimuli=100,WIDTH_transformed=51,HEIGHT_transformed=51,write_dir=None):



    running_loss = 0
    running_corr = 0

    running_size = 0
    corr_all=[]

    # with open(write_dir[0], "wb") as f_stimuli, open(write_dir[1],"wb") as f_spikes:
        
        
    try:
        os.remove(write_dir[1]) 
    except OSError:
        pass

    
    with open(write_dir[0], "wb") as f_stimuli:
      for n, (stimuli, spikes, targets) in test_ds.enumerate():
               # if n.numpy()!=13:
               #       continue
               stimuli=tf.transpose(stimuli,perm=[1,0,3,2])
               spikes=tf.transpose(spikes,perm=[1,0,3,2])
               targets=tf.transpose(targets,perm=[1,0,3,2])
               loss_t,corr_t=actor_vae_response(stimuli,spikes, targets,
                                                    beta,checkpoint_dir_actor_vae,
                                                         f_stimuli,fwd_dir,file_to_use,
                                                             HEIGHT_stimuli,WIDTH_stimuli,
                                                                 WIDTH_transformed,HEIGHT_transformed)


               assert np.isfinite(loss_t)
               running_loss += loss_t
               running_corr += corr_t

               running_size += 1
    
    running_loss /= running_size
    running_corr /= running_size



    return running_loss,running_corr,corr_all

def test_analyse(test_ds,frame_rate,checkpoint,checkpoint_dir):
    
    # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('latest checlpoint: {}'.format(latest))
    status=checkpoint.restore(latest)
    status.assert_consumed()  # Optional sanity checks.
    encoder_test=checkpoint.encoder


    running_loss = 0
    running_size = 0
    slc=np.arange(250,500)
    for n, (stimuli, spikes) in test_ds.enumerate():
               
               stimuli=tf.transpose(stimuli,perm=[1,0,2,3])
               spikes=tf.squeeze(spikes,axis=[0,3])
               est_rate_t,obs_rate_t,loss_t= test_forward(stimuli, spikes,encoder_test)
               plot_psth(NUM_NEURONS,frame_rate,slc,est_rate_t.numpy(),obs_rate_t.numpy())
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_size += 1
    
    running_loss /= running_size
    test_losses=running_loss
    print(test_losses)
    return test_losses


def plot_psth(num_neurons,frame_rate,slc,est_rate,obs_rate):

    
    # Plot a slice of the true and predicted firing rates
    # slc = slice(250, 500)
    fig, axs = plt.subplots(num_neurons, 1, figsize=(8, 16), sharex=True)
    for n in range(num_neurons):
        axs[n].plot(slc/frame_rate, obs_rate[slc, n], '-k', lw=3, label="true rate")
        axs[n].plot(slc/frame_rate, est_rate[slc, n], label="CNN")
        axs[n].set_ylabel("rate [spk/s]")
        if n == 0:
            axs[n].legend()
        if n == num_neurons - 1:
            axs[n].set_xlabel("time [sec]")
    
    plt.tight_layout()

    return 0