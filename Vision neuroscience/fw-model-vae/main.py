import tensorflow as tf
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
from math import floor


# from experiment import send_out,run_exp
from RGCDataset import load_data
from train_test import train_forward,test_forward,train_backward,train_actor_vae
from my_network import forward_model_vae,encoder_vae,decoder_vae,Actor_vae,Attention
from test_evals import test_pca,test_auto_encoder,test_no_transform,test_actor,test_no_transform_vae,test_actor_vae,test_actor_vae_sampling
from dim_reduction_utils import tSNE_transform

# import tensorflow_docs.vis.embed as embed
# import glob
# import imageio

import logging
tf.get_logger().setLevel(logging.ERROR)


MAIN_DIR='../data/dataset/'



PATH = []
# _____________________
num_of_epochs = 130
num_of_epochs_actor=60
frame_rate=0.01
mean_firing_rate=0.05

n_iter=3
beta_vae=400

num_of_slots=3
latent_dims=15

noise=tf.random.normal(shape=[1000,latent_dims])
# _____________________

BATCH_SIZE=287
BATCH_SIZE_val=72
BATCH_SIZE_test=5


# -------------------------original data complete size------------------
DIR_fw_main='../data/dataset/'
DIR_fw_main_test='../data/dataset/'
loader_fw_main=load_data(DIR_fw_main,DIR_fw_main_test,'white_noise')
train_dataset_fw_main,val_dataset_fw_main,test_dataset_fw_main=loader_fw_main.make_dataset()
train_dataset_fw_main=train_dataset_fw_main.batch(1)
val_dataset_fw_main=val_dataset_fw_main.batch(1)
test_dataset_fw_main=test_dataset_fw_main.batch(1)

# -------------------------original data, one third size-----------------
DIR_fw_sub0='../data/dataset-attention-iteration/0-3ed-original/'
DIR_fw_sub_test0='../data/dataset/'
loader_fw_sub0=load_data(DIR_fw_sub0,DIR_fw_sub_test0,'white_noise')
train_dataset_fw_sub0,val_dataset_fw_sub0,_=loader_fw_sub0.make_dataset()
train_dataset_fw_sub0=train_dataset_fw_sub0.batch(1)
val_dataset_fw_sub0=val_dataset_fw_sub0.batch(1)

# -------------------------------------------------
DIR_fw_sub='../data/dataset-attention-iteration/0-3ed/'
DIR_fw_sub_test='../data/dataset/'
loader_fw_sub=load_data(DIR_fw_sub,DIR_fw_sub_test,'white_noise')
train_dataset_fw_sub,val_dataset_fw_sub,_=loader_fw_sub.make_dataset()
train_dataset_fw_sub=train_dataset_fw_sub.batch(1)
val_dataset_fw_sub=val_dataset_fw_sub.batch(1)
# test_dataset_fw_sub=test_dataset_fw_sub.batch(1)
# -------------------------------------------------


WIDTH_spikes = 1
HIGHT_spikes = loader_fw_main.num_neurons # nb_cells
WIDTH_stimuli = loader_fw_main.width
HEIGHT_stimuli = loader_fw_main.height
NUM_NEURONS= loader_fw_main.num_neurons # nb_cells


WIDTH_transformed=3
HEIGHT_transformed=3
# ------------------------------- main forward model---------------------------------
# ------------------------------- main forward model---------------------------------
checkpoint_dir_fw_main='./training_checkpoints-fwd-main/'
latest = tf.train.latest_checkpoint(checkpoint_dir_fw_main)
generator =forward_model_vae(HEIGHT_stimuli,WIDTH_stimuli,HIGHT_spikes,WIDTH_spikes,sigma=0.1,mean_firing_rate=mean_firing_rate)
encoder =encoder_vae(HEIGHT_stimuli,WIDTH_stimuli,HIGHT_spikes,WIDTH_spikes,latent_dims,sigma=0.1)
decoder =decoder_vae(HEIGHT_stimuli,WIDTH_stimuli,HIGHT_spikes,WIDTH_spikes,latent_dims,sigma=0.1,mean_firing_rate=mean_firing_rate)
print('latest checlpoint: {}'.format(latest))
checkpoint_fw_main = tf.train.Checkpoint(generator=generator,encoder=encoder,decoder=decoder)
status=checkpoint_fw_main.restore(latest)
fw_main=checkpoint_fw_main.generator
encoder_main=checkpoint_fw_main.encoder
decoder_main=checkpoint_fw_main.decoder
# -------------------------------------------------

encoder_sub =encoder_vae(HEIGHT_stimuli,WIDTH_stimuli,HIGHT_spikes,WIDTH_spikes,latent_dims,sigma=0.1)
decoder_sub =decoder_vae(HEIGHT_stimuli,WIDTH_stimuli,HIGHT_spikes,WIDTH_spikes,latent_dims,sigma=0.1,mean_firing_rate=mean_firing_rate)
fw_sub =forward_model_vae(HEIGHT_stimuli,WIDTH_stimuli,HIGHT_spikes,WIDTH_spikes,sigma=0.1,mean_firing_rate=mean_firing_rate)


actor_vae= Actor_vae(HEIGHT_stimuli,WIDTH_stimuli,WIDTH_transformed,HEIGHT_transformed)

# -------------------------------------------------

encoder_sub_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9)
decoder_sub_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9)
fw_sub_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9)

actor_vae_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9)


#--------------------------------------------------
# Keep track of the best model

checkpoint_dir_fw_sub= './training_checkpoints_fwd_vae/'
checkpoint_fw_sub = tf.train.Checkpoint(encoder_sub_optimizer=encoder_sub_optimizer,
                                 decoder_sub_optimizer=decoder_sub_optimizer,
                                 fw_sub_optimizer=fw_sub_optimizer,
                                  encoder=encoder_sub,decoder=decoder_sub,generator=fw_sub)
manager_fwd_sub_vae=tf.train.CheckpointManager(
    checkpoint_fw_sub, checkpoint_dir_fw_sub, max_to_keep=5)


checkpoint_dir_actor_vae = './training_checkpoints_actor_vae/'
checkpoint_actor_vae = tf.train.Checkpoint(
                                  actor_vae_optimizer=actor_vae_optimizer,
                                  actor_vae=actor_vae
                                  )
manager_actor_vae=tf.train.CheckpointManager(
    checkpoint_actor_vae, checkpoint_dir_actor_vae , max_to_keep=5)


def fit(train_ds, val_ds,test_ds,encoder_sub_local,decoder_sub_local,fw_sub_local,
        encoder_sub_optimizer_local,decoder_sub_optimizer_local,fw_sub_optimizer_local,num_of_epochs,iter_i):

    # Keep track of the best model
    manager_fwd_sub_vae.save()
    best_loss = 1e8
    best_corr=-1e8

    # Track the train and validation loss
    train_losses = []
    val_losses = []
    test_losses = []
    gen_losses = []
    
    train_corrs=[]
    val_corrs=[]
    test_corrs=[]
    gen_corrs=[]
    for epoch in range(num_of_epochs):
        print(epoch)
        # for phase in ['train', 'val', 'test','generation']:
        for phase in ['train', 'val', 'test']:    
            # track the running loss over batches
          running_loss = 0
          running_corr = 0
          running_size = 0
          if phase =='train':
            for n, (stimuli, spikes) in train_ds.enumerate():
               # print(n.numpy())
               
               
               stimuli=tf.transpose(stimuli,perm=[1,0,2,3])
               spikes=tf.squeeze(spikes,axis=[0,3])
               est_rate_t,obs_rate_t,loss_t,corr_t= train_forward(stimuli, spikes, epoch + num_of_epochs*(iter_i),'train',
                                                        encoder_sub_local,decoder_sub_local,fw_sub_local,
                                                        encoder_sub_optimizer_local,decoder_sub_optimizer_local,
                                                        fw_sub_optimizer_local,beta_vae)
               # print(loss_t.numpy())
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr +=corr_t.numpy()
               running_size += 1
          elif phase == 'val':
              
            for n, (stimuli, spikes) in val_ds.enumerate():
               stimuli=tf.transpose(stimuli,perm=[1,0,2,3])
               spikes=tf.squeeze(spikes,axis=[0,3])
               est_rate_t,obs_rate_t,loss_t,corr_t= train_forward(stimuli, spikes,epoch + num_of_epochs*(iter_i),'val',
                                                        encoder_sub_local,decoder_sub_local,fw_sub_local,
                                                        encoder_sub_optimizer_local,decoder_sub_optimizer_local,
                                                        fw_sub_optimizer_local,beta_vae)
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr +=corr_t.numpy()
               running_size += 1
          elif phase == 'test':
              
            for n, (stimuli, spikes) in test_ds.enumerate():
               stimuli=tf.transpose(stimuli,perm=[1,0,2,3])
               spikes=tf.squeeze(spikes,axis=[0,3])
               est_rate_t,obs_rate_t,loss_t,corr_t= train_forward(stimuli, spikes, epoch + num_of_epochs*(iter_i),'test',
                                                        encoder_sub_local,decoder_sub_local,fw_sub_local,
                                                        encoder_sub_optimizer_local,decoder_sub_optimizer_local,
                                                        fw_sub_optimizer_local,beta_vae)
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr +=corr_t.numpy()
               running_size += 1
            # compute the train/validation loss and update the best
            # model parameters if this is the lowest validation loss yet

          elif phase == 'generation':
              
            for n, (stimuli, spikes) in test_ds.enumerate():
               stimuli=tf.transpose(stimuli,perm=[1,0,2,3])
               spikes=tf.squeeze(spikes,axis=[0,3])
               est_rate_t,obs_rate_t,loss_t,corr_t= test_forward(stimuli, spikes, epoch + num_of_epochs*(iter_i),'generation',encoder_sub_local,
                                                                 decoder_sub_local,fw_sub_local,
                                                                  beta_vae)
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr +=corr_t.numpy()
               running_size += 1
            # compute the train/validation loss and update the best
            # model parameters if this is the lowest validation loss yet
          running_loss /= running_size
          running_corr/=running_size
          
          if phase == "train":
                train_losses.append(running_loss)
                train_corrs.append(running_corr)
 
          elif phase == 'val':
                val_losses.append(running_loss)
                print('running loss:'+ str(running_loss)+'best loss:'+str(best_loss))
                val_corrs.append(running_corr)
                # if running_loss < best_loss:
                if running_corr > best_corr:
                    best_loss = running_loss
                    best_corr = running_corr
                    manager_fwd_sub_vae.save()
          elif phase == 'test':
                test_losses.append(running_loss)
                test_corrs.append(running_corr)
                
          # elif phase == 'generation':
          #       gen_losses.append(running_loss)
          #       gen_corrs.append(running_corr)



    plt.figure()
    plt.plot(range(len(train_losses)),train_losses)
    plt.plot(range(len(val_losses)),val_losses)
    plt.plot(range(len(test_losses)),test_losses)
    plt.plot(range(len(gen_losses)),gen_losses)
    plt.title('fwd loss iter {}'.format(iter_i))
    
    plt.figure()
    
    plt.plot(range(len(train_corrs)),train_corrs)
    plt.plot(range(len(val_corrs)),val_corrs)
    plt.plot(range(len(test_corrs)),test_corrs)
    plt.plot(range(len(gen_corrs)),gen_corrs)
    plt.title('fwd corr iter {}'.format(iter_i))
    
    
    
    
    return [train_losses,train_corrs],[val_losses,val_corrs],[test_losses,test_corrs]


def fit_actor_vae(train_ds, val_ds,test_ds,actor_vae_local,actor_vae_optimizer_local,
                                                 
                                                                encoder_fixed,decoder_fixed, fw_sub_fixed,num_of_epochs,iter_i):

    # Keep track of the best model
    # manager_2.save()
    best_loss = 1e8

    # Track the train and validation loss
    train_losses = []
    val_losses = []
    test_losses = []
    
    train_corrs = []
    val_corrs = []
    test_corrs= []
    

    for epoch in range(num_of_epochs):
        print(epoch)
        # for phase in ['train', 'val','test']:
        for phase in ['train', 'val','test']:
            
            # track the running loss over batches
          running_loss = 0
          running_corr = 0

          running_size = 0
          if phase =='train':
            for n, (stimuli, spikes) in train_ds.enumerate():
               
               stimuli=tf.transpose(stimuli,perm=[1,0,2,3])
               spikes=tf.squeeze(spikes,axis=[0,3])
               spikes=spikes
               est_rate_t,obs_rate_t,loss_t,corr_t= train_actor_vae(stimuli, spikes, epoch + num_of_epochs*(iter_i),'train',
                                                                   actor_vae_local,actor_vae_optimizer_local,
                                                             
                                                                   encoder_fixed,decoder_fixed,fw_sub_fixed,beta_vae)

               running_loss += loss_t.numpy()
               running_corr+=np.array(corr_t)

               running_size += 1
          elif phase == 'val':
              
            for n, (stimuli, spikes) in val_ds.enumerate():
                
               stimuli=tf.transpose(stimuli,perm=[1,0,2,3])
               spikes=tf.squeeze(spikes,axis=[0,3])
               spikes=spikes
               est_rate_t,obs_rate_t,loss_t,corr_t= train_actor_vae(stimuli, spikes, epoch + num_of_epochs*(iter_i),'val',
                                                                   actor_vae_local,actor_vae_optimizer_local,
                                                           
                                                                   encoder_fixed,decoder_fixed,fw_sub_fixed,beta_vae)
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr+=np.array(corr_t)

               running_size += 1
            # compute the train/validation loss and update the best
            # model parameters if this is the lowest validation loss yet
          elif phase == 'test':
              
            for n, (stimuli, spikes) in test_ds.enumerate():
               stimuli=tf.transpose(stimuli,perm=[1,0,2,3])
               spikes=tf.squeeze(spikes,axis=[0,3])
               spikes=spikes
               est_rate_t,obs_rate_t,loss_t,corr_t= train_actor_vae(stimuli, spikes, epoch + num_of_epochs*(iter_i),'test',
                                                                   actor_vae_local,actor_vae_optimizer_local,
                                                        
                                                                   encoder_fixed,decoder_fixed,fw_sub_fixed,beta_vae)
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_corr+=np.array(corr_t)

               running_size += 1
            # compute the train/validation loss and update the best
            # model parameters if this is the lowest validation loss yet
          running_loss /= running_size
          running_corr /= running_size

            
          if phase == "train":
                train_losses.append(running_loss)
                train_corrs.append(running_corr)

                
                
          elif phase == "val":
                val_losses.append(running_loss)
                val_corrs.append(running_corr)

                if running_loss < best_loss:
                    best_loss = running_loss
                    # manager_actor_attention.save()
                    manager_actor_vae.save()
          elif phase == "test":
                test_losses.append(running_loss)
                test_corrs.append(running_corr)


        # Update the learning rate
        # scheduler.step()


    plt.figure()
    plt.plot(range(len(train_losses)),train_losses)
    plt.plot(range(len(val_losses)),val_losses)
    plt.plot(range(len(test_losses)),test_losses)
    plt.title('actor_vae loss iter {}'.format(iter_i))
    plt.figure()
    plt.plot(range(len(train_corrs)),train_corrs)
    plt.title('actor_vae TRAIN corr iter {}'.format(iter_i))
    plt.figure()
    plt.plot(range(len(val_corrs)),val_corrs)
    plt.title('actor_vae VAL corr iter {}'.format(iter_i))
    plt.figure()
    plt.plot(range(len(test_corrs)),test_corrs)
    plt.title('actor_vae TEST corr iter {}'.format(iter_i))
    

    
    return [train_losses,train_corrs],[val_losses,val_corrs],[test_losses,test_corrs]




def actor_vae_iter(n_iter,encoder_sub,decoder_sub,fw_sub,actor_vae,actor_vae_optimizer,
                   encoder_sub_optimizer,decoder_sub_optimizer,fw_sub_optimizer):
    corr_AC=[]
    fev_AC=[]
    loss_AC=[]
    latent_AC=[]
    latent_AC_emb=[]
    
    latent_AC_fixed=[]
    latent_AC_emb_fixed=[]
    for i in range(0,n_iter):
        
        if i == 0:
            
            iter_train_ds=train_dataset_fw_sub0
            iter_val_ds=val_dataset_fw_sub0
            
            
            # latest = tf.train.latest_checkpoint(checkpoint_dir_fw_sub)
            # print('latest checlpoint: {}'.format(latest))
            # status=checkpoint_fw_sub.restore(latest)
            # fw_sub=checkpoint_fw_sub.generator
            # encoder_sub=checkpoint_fw_sub.encoder
            # decoder_sub=checkpoint_fw_sub.decoder
            # encoder_sub_optimizer=checkpoint_fw_sub.encoder_sub_optimizer
            # decoder_sub_optimizer=checkpoint_fw_sub.decoder_sub_optimizer
            # fw_sub_optimizer=checkpoint_fw_sub.fw_sub_optimizer

        else:
            

            
            latest = tf.train.latest_checkpoint(checkpoint_dir_actor_vae)
            print('latest checlpoint: {}'.format(latest))
            status=checkpoint_actor_vae.restore(latest)
            actor_vae=checkpoint_actor_vae.actor_vae
            actor_vae_optimizer=checkpoint_actor_vae.actor_vae_optimizer

            iter_train_ds=train_dataset_fw_sub
            iter_val_ds=val_dataset_fw_sub
        
        # # # # input('press something')
        train_losses_fwd,val_losses_fwd,test_losses_fwd =fit(iter_train_ds, iter_val_ds,test_dataset_fw_main,
                                                                encoder_sub,decoder_sub,fw_sub,
                                                                encoder_sub_optimizer,decoder_sub_optimizer,fw_sub_optimizer,num_of_epochs,i)
        latest = tf.train.latest_checkpoint(checkpoint_dir_fw_sub)
        print('latest checlpoint: {}'.format(latest))
        status=checkpoint_fw_sub.restore(latest)
        fw_sub=checkpoint_fw_sub.generator
        encoder_sub=checkpoint_fw_sub.encoder
        decoder_sub=checkpoint_fw_sub.decoder
        encoder_sub_optimizer=checkpoint_fw_sub.encoder_sub_optimizer
        decoder_sub_optimizer=checkpoint_fw_sub.decoder_sub_optimizer
        fw_sub_optimizer=checkpoint_fw_sub.fw_sub_optimizer


        train_losses,val_losses,test_losses=fit_actor_vae(train_dataset_fw_sub0,val_dataset_fw_sub0,test_dataset_fw_main,
                                                                actor_vae,actor_vae_optimizer,
                                                                    encoder_sub,decoder_sub,fw_sub, num_of_epochs_actor,i)

        write_dir=[DIR_fw_sub + 'train_stimuli.bin',DIR_fw_sub + 'train_spikes.bin']
        t1,t2,t3,corr_all_AC1= test_actor_vae(train_dataset_fw_sub0,checkpoint_fw_main,checkpoint_dir_fw_main, checkpoint_dir_actor_vae,
                                        beta_vae,WIDTH_transformed=WIDTH_transformed,HEIGHT_transformed= HEIGHT_transformed,write_dir=write_dir)
        write_dir=[DIR_fw_sub + 'val_stimuli.bin',DIR_fw_sub + 'val_spikes.bin']
        tt1,tt2,tt3,corr_all_AC2= test_actor_vae(val_dataset_fw_sub0,checkpoint_fw_main,checkpoint_dir_fw_main, checkpoint_dir_actor_vae,
                                        beta_vae, WIDTH_transformed=WIDTH_transformed,HEIGHT_transformed= HEIGHT_transformed,write_dir=write_dir)
        write_dir=[DIR_fw_sub + 'test_stimuli.bin',DIR_fw_sub + 'test_spikes.bin']
        ttt1,ttt2,ttt3,corr_all_AC3= test_actor_vae(test_dataset_fw_main,checkpoint_fw_main,checkpoint_dir_fw_main, checkpoint_dir_actor_vae,
                                        beta_vae, WIDTH_transformed=WIDTH_transformed,HEIGHT_transformed= HEIGHT_transformed,write_dir=write_dir,noise=noise)
        loss_AC.append([t1,tt1,ttt1])
        corr_AC.append([t2,tt2,ttt2])
        fev_AC.append([t3,tt3,ttt3])
        

  
        
        # write_dir=[DIR_fw_sub + 'test_stimuli.bin',DIR_fw_sub + 'test_spikes.bin']

        # tttt1,tttt2,tttt3,_,tttt4= test_actor_vae_sampling(test_dataset_fw_main,checkpoint_fw_main,checkpoint_dir_fw_main, checkpoint_dir_actor_vae,
        #                                 beta_vae, WIDTH_transformed=WIDTH_transformed,HEIGHT_transformed= HEIGHT_transformed,write_dir=write_dir,noise=noise)
        # LA_test2=np.array(tttt4)[0,:,:]
        # for _ in range(len(tttt4)-1):
        #     LA_test2=np.concatenate((LA_test2,np.array(tttt4)[_+1,:]),axis=0)
        # latent_AC_fixed.append([LA_test2])
        
        
        
        # LA_test_emb2=tSNE_transform(LA_test2,n_components=2)
        # latent_AC_emb_fixed.append([LA_test_emb2])
        
    plt.figure()
    plt.plot(range(len(np.array(loss_AC)[:,0])),np.array(loss_AC)[:,0],'*')
    plt.plot(range(len(np.array(loss_AC)[:,1])),np.array(loss_AC)[:,1],'*')
    plt.plot(range(len(np.array(loss_AC)[:,2])),np.array(loss_AC)[:,2],'*')
    plt.title('loss progress')
    plt.figure()
    plt.plot(range(len(np.array(corr_AC)[:,0])),np.array(corr_AC)[:,0],'*')
    plt.plot(range(len(np.array(corr_AC)[:,1])),np.array(corr_AC)[:,1],'*')
    plt.plot(range(len(np.array(corr_AC)[:,2])),np.array(corr_AC)[:,2],'*')
    # plt.title('corr progress')
    

    return loss_AC,corr_AC,latent_AC_fixed

loss,corr,latent_AC_fixed=actor_vae_iter(n_iter,encoder_sub,decoder_sub,fw_sub,actor_vae,actor_vae_optimizer,
                   encoder_sub_optimizer,decoder_sub_optimizer,fw_sub_optimizer)



loss=np.squeeze(np.array(loss))
loss.tofile('./loss_{}_{}'.format(beta_vae,latent_dims))

corr=np.squeeze(np.array(corr))
corr.tofile('./corr_{}_{}'.format(beta_vae,latent_dims))


latent_AC_fixed=np.squeeze(np.array(latent_AC_fixed))
latent_AC_fixed.tofile('./latent_AC_fixed_{}_{}'.format(beta_vae,latent_dims))


        



