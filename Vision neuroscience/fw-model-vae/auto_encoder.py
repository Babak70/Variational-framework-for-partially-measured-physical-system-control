import tensorflow as tf
from my_network import Auto_encoder
from RGCDataset import load_data
from losses import generator_loss
import datetime
MAIN_DIR='./dataset/'

num_of_epochs=100

log_dir="logs-AE/"
# -------------------------------------------------
loader=load_data(MAIN_DIR,'white_noise')
train_dataset,val_dataset,test_dataset=loader.make_dataset()

train_dataset=train_dataset.batch(1)
val_dataset=val_dataset.batch(1)
test_dataset=test_dataset.batch(1)
# -------------------------------------------------


WIDTH_spikes = 1
HIGHT_spikes = loader.num_neurons # nb_cells
WIDTH_stimuli = loader.width
HEIGHT_stimuli = loader.height
NUM_NEURONS= loader.num_neurons # nb_cells

WIDTH_transformed=1
HEIGHT_transformed=1

AE=Auto_encoder(HEIGHT_stimuli,WIDTH_stimuli,WIDTH_transformed,HEIGHT_transformed)
AE_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9)


checkpoint_dir_3 = './training_checkpoints_3/'
checkpoint_3 = tf.train.Checkpoint(AE_optimizer=AE_optimizer,
                                  AE=AE)
manager_3=tf.train.CheckpointManager(
    checkpoint_3, checkpoint_dir_3, max_to_keep=5)

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_auto_encoder(stimuli, epoch,train_what):

   if train_what=='train':
         
      with tf.GradientTape() as gen_tape:
            rec_stimuli,red_stimuli= AE(stimuli, training=True)
        
            gen_total_loss = generator_loss(rec_stimuli,stimuli,"mse")
            gen_total_loss=tf.math.reduce_mean(gen_total_loss)
      
      AE_gradients = gen_tape.gradient(gen_total_loss,
                                              AE.trainable_variables)
      AE_optimizer.apply_gradients(zip(AE_gradients,
                                              AE.trainable_variables))
    
      with summary_writer.as_default():
            tf.summary.scalar('loss-AE-train', gen_total_loss, step=epoch)
            tf.summary.image('stimuli-AE', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('rec-stimuli-AE', tf.transpose(rec_stimuli, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('red-stimuli-AE', tf.transpose(red_stimuli, perm=[0,2,3,1]),step=epoch)
        
   elif train_what=='val':
      rec_stimuli,red_stimuli= AE(stimuli, training=False)

        
      gen_total_loss = generator_loss(rec_stimuli,stimuli,"mse")
      gen_total_loss=tf.math.reduce_mean(gen_total_loss)

      with summary_writer.as_default():
            tf.summary.scalar('loss-AE-val', gen_total_loss, step=epoch)
            tf.summary.image('stimuli-val-AE', tf.transpose(stimuli, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('rec-stimuli-val-AE', tf.transpose(rec_stimuli, perm=[0,2,3,1]),step=epoch)
            tf.summary.image('red-stimuli-val-AE', tf.transpose(red_stimuli, perm=[0,2,3,1]),step=epoch)
    
   return gen_total_loss,red_stimuli

def fit_auto_encoder(train_ds, val_ds,epochs):

    # Keep track of the best model
    manager_3.save()
    best_loss = 1e8

    # Track the train and validation loss
    train_losses = []
    val_losses = []
    for epoch in range(num_of_epochs):
        print(epoch)
        for phase in ['train', 'val']:
            
            # track the running loss over batches
          running_loss = 0
          running_size = 0
          if phase =='train':
            for n, (stimuli, spikes) in train_ds.enumerate():
               
               stimuli=tf.transpose(stimuli,perm=[1,0,2,3])

               loss_t,_= train_auto_encoder(stimuli, epoch,'train')
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_size += 1
          else:
              
            for n, (stimuli, spikes) in val_ds.enumerate():
               stimuli=tf.transpose(stimuli,perm=[1,0,2,3])

               loss_t,_= train_auto_encoder(stimuli, epoch,'val')
               assert tf.math.is_finite(loss_t)
               running_loss += loss_t.numpy()
               running_size += 1
            # compute the train/validation loss and update the best
            # model parameters if this is the lowest validation loss yet
          running_loss /= running_size
            
          if phase == "train":
                train_losses.append(running_loss)
                
          else:
                val_losses.append(running_loss)
                if running_loss < best_loss:
                    best_loss = running_loss
                    manager_3.save()

        # Update the learning rate
        # scheduler.step()
    
    return train_losses,val_losses



for IT in range(1):

        train_losses,val_losses=fit_auto_encoder(train_dataset,val_dataset, num_of_epochs)
