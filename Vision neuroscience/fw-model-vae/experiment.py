import tensorflow as tf
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
from scipy.io import savemat


def send_out(stim,skip_frames,direc):


    
      stim=tf.squeeze(stim).numpy()
       # plt.imshow(stim[0,:,:])
       #  plt.colorbar()
      stim=np.transpose(stim, [1,2,0])
      
      pre_pad=np.repeat(np.expand_dims(stim[:,:,0],axis=2),skip_frames//2,axis=2)
      post_pad=np.repeat(np.expand_dims(stim[:,:,-1],axis=2),skip_frames//2,axis=2)
      
      stim=np.concatenate((pre_pad,stim),axis=2)
      stim=np.concatenate((stim,post_pad),axis=2)

      stim=stim*255.
      stim=stim.astype(np.uint8)
      savemat('b.mat', {'b':stim[:,:,0:100]})
      
      print(stim[:,:,0])
      
      plt.figure()
      plt.imshow(stim[:,:,0])
      plt.colorbar()


      stim.tofile(direc+'./stim.bin')


def read_in(NUM_frames,HEIGHT,WIDTH,direc):
   
   train_dataR=np.fromfile(direc+'train_dataF.bin',dtype=np.uint8)/255.
#   train_data0=np.asarray(train_dataR).astype(np.float32)
   train_data=np.reshape(train_dataR, [NUM_frames,HEIGHT,WIDTH,1]).astype(np.float32)
   
   return train_data


def run_exp(NUM_frames,HEIGHT,WIDTH,direc):
    
      eng = matlab.engine.start_matlab()
      eng.cd(direc)

      eng.eval('load_data',nargout=0)
      eng.quit()
      # train_data=read_in(NUM_frames,HEIGHT,WIDTH)
      # return train_data

# (train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
# train_images=np.random.rand(960,20,20)
# send_out(train_images,40)



def save2matlab(name,mdict):
    
 savemat(name, mdict)


 return  0