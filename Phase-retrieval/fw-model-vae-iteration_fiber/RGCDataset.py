# import h5py
# import numpy as np
import tensorflow as tf
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import normalize
# 

class load_data():

    def __init__(self,MAIN_DIR,MAIN_DIR_test,stim_type,num_frames,num_frames_test):
        self.MAIN_DIR=MAIN_DIR
        self.MAIN_DIR_test=MAIN_DIR_test
        self.stim_type=stim_type
        # self.batch_size=batch_size
        # self.batch_size_val=batch_size_val
        # self.batch_size_test=batch_size_test
        self.num_frames=num_frames
        self.num_frames_test=num_frames_test
        self.height=51
        self.width=51
        self.width_out=100# 200
        self.height_out=100# 200
        
    def batching(self,data_files,record_bytes,first_dim,phase):
        
        
        if phase =='train':
            # BATCH_SIZE=self.batch_size
            duration=self.num_frames
        elif phase=='val':
            # BATCH_SIZE= self.batch_size_val
            duration=self.num_frames
        elif phase=='test':
            # BATCH_SIZE= self.batch_size_test
            duration=self.num_frames_test
        else:
            print('file not found')
            
        NUM_files=len(data_files)
        DATASETS=[None]*NUM_files
        
        for i, data_file in enumerate(data_files):
          train_dataset_portion = tf.data.FixedLengthRecordDataset(data_file, record_bytes=record_bytes[i]*4)
          train_dataset_portion = train_dataset_portion.map( lambda x: load(x,record_bytes[i],first_dim[i],duration),
                                          num_parallel_calls=None)
        
          DATASETS[i]=train_dataset_portion
        # train_dataset = tf.data.Dataset.zip(tuple(DATASETS)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        # train_dataset = tf.data.Dataset.zip(tuple(DATASETS)).batch(BATCH_SIZE)
        train_dataset = tf.data.Dataset.zip(tuple(DATASETS))
        

        # plt.plot(train_dataset.take(1)[0,:,0,0])
        # plt.show()
        
        return train_dataset

    def make_dataset(self):

        

        record_bytes_spikes=self.width_out*self.height_out*self.num_frames
        record_bytes_stimuli=self.width *self.height *self.num_frames
        record_bytes_targets=self.width_out*self.height_out*self.num_frames

        record_bytes=[record_bytes_stimuli,record_bytes_spikes,record_bytes_targets]
        first_dim=[self.height,self.height_out,self.height_out]

        #Train
        data_files=[self.MAIN_DIR+'train_stimuli.bin',self.MAIN_DIR+'train_spikes.bin',self.MAIN_DIR+'train_targets.bin']
        train_dataset=self.batching(data_files,record_bytes,first_dim,'train')
        
        #Val
        data_files=[self.MAIN_DIR+'val_stimuli.bin',self.MAIN_DIR+'val_spikes.bin',self.MAIN_DIR+'val_targets.bin']
        val_dataset=self.batching(data_files,record_bytes,first_dim,'val')
        
        #Test
        data_files=[self.MAIN_DIR_test+'test_stimuli.bin',self.MAIN_DIR_test+'test_spikes.bin',self.MAIN_DIR_test+'test_targets.bin']
        test_dataset=self.batching(data_files,record_bytes,first_dim,'test')
        
        return  train_dataset,val_dataset,test_dataset
    
def load(value,record_bytes,first_dim,duration):

  record= tf.io.decode_raw(value, tf.float32)   
  depth_label_major = (tf.reshape(record,
      [duration,int(first_dim), int(record_bytes/(first_dim*duration))]))
  image=tf.transpose(depth_label_major,[0,1,2])
  # image=normalize(tf.cast(image,tf.float32))
  image=image
  return image