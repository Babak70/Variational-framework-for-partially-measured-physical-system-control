import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import entropy
from sklearn.decomposition import PCA
from numpy.random import default_rng
import tikzplotlib 

import tensorflow as tf

# from experiment import send_out,run_exp
from RGCDataset import load_data
from my_network import forward_model_vae,encoder_vae,decoder_vae,Actor_vae
from dim_reduction_utils import tSNE_transform
from test_evals import test_no_transform_vae,test_actor_vae_sampling

beta=400
latent_dims=15
emb_dims=2

Tsne=True

n_iter=3
num_repeats=1
repeat_instance=1
shift=0
batch_size=1000
batch_num=5
latnet_size=15
# MAIN_dir='./results/beta {}-la {}/'.format(beta,latent_dims)
MAIN_dir='./results/beta 400-with fwd 500/'

LA=np.empty((num_repeats,n_iter,batch_size*batch_num,latnet_size))
mean=np.empty((num_repeats,n_iter,batch_size*batch_num,latnet_size))
logvar=np.empty((num_repeats,n_iter,batch_size*batch_num,latnet_size))

for i in range(num_repeats):
    sub_dir= '{}/'.format(repeat_instance+shift)
    temp  = np.fromfile(MAIN_dir + sub_dir + 'LA',np.float32)
    LA[i] = np.reshape(temp,(n_iter,batch_size*batch_num,latnet_size))
    
    # temp  = np.fromfile(MAIN_dir + sub_dir + 'mean',np.float32)
    # mean[i] = np.reshape(temp,(n_iter,batch_size*batch_num,latnet_size))
    
    # temp  = np.fromfile(MAIN_dir + sub_dir + 'logvar',np.float32)
    # logvar[i] = np.reshape(temp,(n_iter,batch_size*batch_num,latnet_size))

LA=np.squeeze(LA)
mean=np.squeeze(mean)
logvar=np.squeeze(logvar)



# -------------------------original data complete size------------------
DIR_fw_main='../data/dataset/'
DIR_fw_main_test='../data/dataset/'
loader_fw_main=load_data(DIR_fw_main,DIR_fw_main_test,'white_noise')
train_dataset_fw_main,val_dataset_fw_main,test_dataset_fw_main=loader_fw_main.make_dataset()
train_dataset_fw_main=train_dataset_fw_main.batch(1)
val_dataset_fw_main=val_dataset_fw_main.batch(1)
test_dataset_fw_main=test_dataset_fw_main.batch(1)

WIDTH_spikes = 1
HIGHT_spikes = loader_fw_main.num_neurons # nb_cells
WIDTH_stimuli = loader_fw_main.width
HEIGHT_stimuli = loader_fw_main.height
NUM_NEURONS= loader_fw_main.num_neurons # nb_cells
WIDTH_transformed=3
HEIGHT_transformed=3
mean_firing_rate=0.05

#### ----------------------Actor-trained ==> forward model----------
checkpoint_dir_fw_main='./training_checkpoints-fwd-main/'
latest = tf.train.latest_checkpoint(checkpoint_dir_fw_main)
generator =forward_model_vae(HEIGHT_stimuli,WIDTH_stimuli,HIGHT_spikes,WIDTH_spikes,sigma=0.1,mean_firing_rate=mean_firing_rate)
encoder =encoder_vae(HEIGHT_stimuli,WIDTH_stimuli,HIGHT_spikes,WIDTH_spikes,latent_dims,sigma=0.1)
decoder =decoder_vae(HEIGHT_stimuli,WIDTH_stimuli,HIGHT_spikes,WIDTH_spikes,latent_dims,sigma=0.1,mean_firing_rate=mean_firing_rate)
print('latest checlpoint: {}'.format(latest))
checkpoint_fw_main = tf.train.Checkpoint(generator=generator,encoder=encoder,decoder=decoder)


_,_,_,_,temp,temp1,temp2=test_no_transform_vae(test_dataset_fw_main,checkpoint_fw_main,checkpoint_dir_fw_main,beta)
LA_fw_main=np.array(temp)[0,:,:]
mean_fw_main=np.array(temp1)[0,:,:]
logvar_fw_main=np.array(temp2)[0,:,:]
for _ in range(len(temp)-1):
            LA_fw_main=np.concatenate((LA_fw_main,np.array(temp)[_+1,:]),axis=0)
            mean_fw_main=np.concatenate((mean_fw_main,np.array(temp1)[_+1,:]),axis=0)
            logvar_fw_main=np.concatenate((logvar_fw_main,np.array(temp2)[_+1,:]),axis=0)




#### ----------------------Actor-UN-trained ==> forward model----------
checkpoint_dir_actor_vae = './'


noise=tf.random.normal(shape=[1000,latent_dims])
_,_,_,_,temp,temp1,temp2= test_actor_vae_sampling(test_dataset_fw_main,checkpoint_fw_main,checkpoint_dir_fw_main, checkpoint_dir_actor_vae,
                                        beta, WIDTH_transformed=WIDTH_transformed,HEIGHT_transformed= HEIGHT_transformed,noise=noise)
LA_fw_main_untrained=np.array(temp)[0,:,:]
mean_fw_main_untrained=np.array(temp1)[0,:,:]
logvar_fw_main_untrained=np.array(temp2)[0,:,:]
for _ in range(len(temp)-1):
       LA_fw_main_untrained=np.concatenate((LA_fw_main_untrained,np.array(temp)[_+1,:]),axis=0)
       mean_fw_main_untrained=np.concatenate((mean_fw_main_untrained,np.array(temp1)[_+1,:]),axis=0)
       logvar_fw_main_untrained=np.concatenate((logvar_fw_main_untrained,np.array(temp2)[_+1,:]),axis=0)
        

#### ---------------------------------------------------------------

LA_emb=np.empty((n_iter,LA.shape[1],emb_dims),np.float32)
LA_emb_fw_main=np.empty((LA_fw_main.shape[1],emb_dims),np.float32)
LA_emb_fw_main_untrained=np.empty((LA_fw_main_untrained.shape[1],emb_dims),np.float32)

early_exaggeration=20

if Tsne==True:
    
    for i in range(n_iter):
        LA_emb[i,:,:]=tSNE_transform(LA[i,...],n_components=emb_dims,early_exaggeration=early_exaggeration)

    LA_emb_fw_main=tSNE_transform(LA_fw_main,n_components=emb_dims,early_exaggeration=early_exaggeration)
    LA_emb_fw_main_untrained=tSNE_transform(LA_fw_main_untrained,n_components=emb_dims,early_exaggeration=early_exaggeration)

else:
    
    for i in range(n_iter):
        pca=PCA(emb_dims).fit(LA[i,...]) #learning the manifold
        LA_emb[i,:,:]=pca.transform(LA[i,...]) # projection of high res samples onto the low-dim basis vectors
    
    pca=PCA(emb_dims).fit(LA_fw_main) #learning the manifold
    LA_emb_fw_main=pca.transform(LA_fw_main) # projection of high res samples onto the low-dim basis vectors

    pca=PCA(emb_dims).fit(LA_fw_main_untrained,) #learning the manifold
    LA_emb_fw_main_untrained=pca.transform(LA_fw_main_untrained,) # projection of high res samples onto the low-dim basis vectors



# # ------------------------------
# cor_la=[None]*n_iter
# mse_la=[None]*n_iter
# kl_la=[None]*n_iter

# for j in range(n_iter):
#     temp=[None]*5000
#     temp2=[None]*5000
#     for i in range(LA.shape[1]):
#         temp[i]=np.corrcoef(LA[j,i,:],LA_fw_main[i,:])[0,1]
#         temp2[i]=entropy(LA[j,i,:],LA_fw_main[i,:])


        
#     cor_la[j]=sum(temp)/5000
#     mse_la[j]=np.mean(np.square(LA[j,:,:]-LA_fw_main[:,:]))
#     kl_la[j]=sum(temp2)/5000







# ----------------------------------
intvl=100

plt.figure()
M=500
rng = default_rng(seed=1)
numbers = rng.choice(5000, size=M, replace=False)


# plt.scatter(LA_emb_fw_main_untrained[0:M,0],LA_emb_fw_main_untrained[0:M,1])
# for i in range(n_iter):
#     # plt.figure()
#     plt.scatter(LA_emb[i,0:M,0],LA_emb[i,0:M,1])
# plt.scatter(LA_emb_fw_main[0:M,0],LA_emb_fw_main[0:M,1])
# plt.xlim([-intvl,intvl])
# plt.ylim([-intvl,intvl])



plt.figure()
plt.scatter(LA_emb_fw_main_untrained[numbers,0],LA_emb_fw_main_untrained[numbers,1])
plt.scatter(LA_emb_fw_main[numbers,0],LA_emb_fw_main[numbers,1])
plt.xlim([-intvl,intvl])
plt.ylim([-intvl,intvl])
tikzplotlib .save("emb0.tex")

plt.figure()
plt.scatter(LA_emb[0,numbers,0],LA_emb[0,numbers,1])
plt.scatter(LA_emb_fw_main[numbers,0],LA_emb_fw_main[numbers,1])
plt.xlim([-intvl,intvl])
plt.ylim([-intvl,intvl])
tikzplotlib .save("emb1.tex")

plt.figure()
plt.scatter(LA_emb[1,numbers,0],LA_emb[1,numbers,1])
plt.scatter(LA_emb_fw_main[numbers,0],LA_emb_fw_main[numbers,1])
plt.xlim([-intvl,intvl])
plt.ylim([-intvl,intvl])
tikzplotlib .save("emb2.tex")

plt.figure()
plt.scatter(LA_emb[2,numbers,0],LA_emb[2,numbers,1])
plt.scatter(LA_emb_fw_main[numbers,0],LA_emb_fw_main[numbers,1])
plt.xlim([-intvl,intvl])
plt.ylim([-intvl,intvl])
tikzplotlib .save("emb3.tex")







# ----------------------------------Z---------------------------
plt.figure()

M=1
shift=0
for _ in range(M):
    plt.plot(LA[0,_+shift,:])
    plt.plot(LA_fw_main[_+shift,:])

    
  


plt.figure()
for _ in range(M):
    plt.plot(LA[1,_+shift,:])
    plt.plot(LA_fw_main[_+shift,:])
  


plt.figure()
for _ in range(M):
    plt.plot(LA[2,_+shift,:])
    plt.plot(LA_fw_main[_+shift,:])

# ----------------------------------MEAN---------------------------
plt.figure()

M=1
shift=2162
for _ in range(M):
    plt.plot(mean[0,_+shift,:])
    plt.plot(mean_fw_main[_+shift,:])
plt.title('mean{}'.format(1))
plt.ylim([-2,2])

plt.figure()
for _ in range(M):
    plt.plot(mean[1,_+shift,:])
    plt.plot(mean_fw_main[_+shift,:])
plt.title('mean{}'.format(2))
plt.ylim([-2,2])

plt.figure()
for _ in range(M):
    plt.plot(mean[2,_+shift,:])
    plt.plot(mean_fw_main[_+shift,:])
plt.title('mean{}'.format(3))
plt.ylim([-2,2])
    
# ----------------------------------LOGVAR---------------------------
plt.figure()

M=1
shift=2162
for _ in range(M):
    plt.plot(np.exp(logvar[0,_+shift,:]))
    plt.plot(np.exp(logvar_fw_main[_+shift,:]))
plt.title('variance{}'.format(1))
plt.ylim([0,2])

plt.figure()
for _ in range(M):
    plt.plot(np.exp(logvar[1,_+shift,:]))
    plt.plot(np.exp(logvar_fw_main[_+shift,:]))
plt.title('variance{}'.format(2))
plt.ylim([0,2])

plt.figure()
for _ in range(M):
    plt.plot(np.exp(logvar[2,_+shift,:]))
    plt.plot(np.exp(logvar_fw_main[_+shift,:]))
plt.title('variance{}'.format(3))
plt.ylim([0,2])