import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
# from skimage.transform import resize
import tensorflow as tf
from my_network import Auto_encoder,backward_model





def single_img_svd(img,n_component):
    
    
     u,s,vh=np.linalg.svd(img)
     
     img_rec=u[:,0:n_component]@np.diag(s[0:n_component])@vh[0:n_component,:]
     return img_rec
 
def pca_transform(images,n_component):
    
    pca=PCA(n_component).fit(images) #learning the manifold
    coeffs=pca.transform(images) # projection of high res samples onto the low-dim basis vectors
    rec_imgs=pca.inverse_transform(coeffs)

    plt.plot(np.cumsum(pca.explained_variance_ratio_),'--')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    return rec_imgs
    

def tSNE_transform(features,n_components,early_exaggeration=None):
    
    features_embedded = TSNE(n_components=n_components,early_exaggeration=early_exaggeration).fit_transform(features)

    return features_embedded


def auto_encoder_transform(images,checkpoint_dir,HEIGHT_stimuli,WIDTH_stimuli,WIDTH_transformed,HEIGHT_transformed):
    
    AE=Auto_encoder(HEIGHT_stimuli,WIDTH_stimuli,WIDTH_transformed,HEIGHT_transformed)
    checkpoint=tf.train.Checkpoint(AE=AE)
    # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('latest checlpoint: {}'.format(latest))
    status=checkpoint.restore(latest)
    # status.assert_consumed()  # Optional sanity checks.
    AE_test=checkpoint.AE
    rec_imgs,_= AE_test(images, training=False)


    return rec_imgs


def actor_transform(images,checkpoint_dir,HEIGHT_stimuli,WIDTH_stimuli,WIDTH_transformed,HEIGHT_transformed):
    
    actor=backward_model(HEIGHT_stimuli,WIDTH_stimuli,WIDTH_transformed,HEIGHT_transformed)
    checkpoint=tf.train.Checkpoint(actor=actor)
    # load best model weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('latest checlpoint: {}'.format(latest))
    status=checkpoint.restore(latest)
    # status.assert_consumed()  # Optional sanity checks.
    actor_test=checkpoint.actor
    tsnf_stimuli= actor_test(images, training=False)
    tsnf_stimuli=tf.transpose(tsnf_stimuli,perm=[0,2,3,1])
    # tsnf_stimuli= stimuli
    tsnf_stimuli=tf.image.resize(tsnf_stimuli,(HEIGHT_stimuli,WIDTH_stimuli),method='nearest')
    tsnf_stimuli=tf.transpose(tsnf_stimuli,perm=[0,3,1,2])


    return tsnf_stimuli
