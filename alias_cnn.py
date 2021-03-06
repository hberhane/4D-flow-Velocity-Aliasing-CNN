import tensorflow as tf
import numpy as np
from tqdm import tqdm
from colorama import Fore
import random
import os
import matplotlib.pyplot as plt

"""
Aliasing CNN uses just the phase images with no velocity aliasing 
the simulated aliasing and a ground-truth locating all aliased voxels is automatically done in the script

The CNN archecture is a 3D hybrid UNet and DenseNet and has been previously described here:  https://doi.org/10.1002/mrm.28257
"""

def parser(tfrecord):

    features = tf.parse_single_example(tfrecord,{'test/image': tf.FixedLenFeature([], tf.string),
               #'test/label': tf.FixedLenFeature([], tf.string),
               #'test/mask': tf.FixedLenFeature([], tf.string),
               #'test/og': tf.FixedLenFeature([], tf.string),
               'test/venc': tf.FixedLenFeature([], tf.int64),
               'test/depth': tf.FixedLenFeature([], tf.int64),
               'test/height': tf.FixedLenFeature([], tf.int64),
               'test/width': tf.FixedLenFeature([], tf.int64)})
    height = tf.cast(features["test/height"], tf.int32)
    width = tf.cast(features["test/width"], tf.int32)
    depth = tf.cast(features["test/depth"], tf.int32)
    venc = tf.cast(features["test/venc"], tf.float32)
    venc = tf.divide(venc,100)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['test/image'], tf.float32)

    

    # Reshape image data into the original shape
    image = tf.reshape(image, [height, width, depth])
    

    image = tf.image.resize_image_with_crop_or_pad(image, 128, 96)
    image = tf.cast(image,tf.float32)
    image2 = image
    #label = tf.image.resize_image_with_crop_or_pad(label, 160, 96)
    #og = tf.image.resize_image_with_crop_or_pad(og, 128, 96)

    #random factor to reduce the venc, between [0.3 0.7] times the venc of the scan
    r = tf.random_uniform(shape=[], minval=0.3, maxval=0.7, dtype=tf.float32)

    """
    Generate mask of voxels >sim_venc and < sim_venc 
    """
    mask1 = image2 > tf.multiply(r,venc)
    mask2 = image2 < tf.multiply(-r,venc)
    new_venc = tf.multiply(r,venc)
    
    mask1 = tf.cast(mask1,tf.float32)
    mask2 = tf.cast(mask2,tf.float32)
    

   
    """
    simulate velocity aliasing based on the mask determined above
    """
    v_mask1 = tf.multiply(mask1,tf.multiply(2.,new_venc))
    image2 = tf.subtract(image2,v_mask1)
    v_mask2 = tf.multiply(mask2,tf.multiply(2.,new_venc))
    image2 = tf.add(image2,v_mask2)

    
    #Generate ground-truth through abs of velocity data > sim_venc
    truth1 = tf.abs(image) > new_venc

    
    truth = tf.cast(truth1, tf.int32)
    
    
    #Nomralize input between (0,1)
    image = (image2 - tf.reduce_min(image2))/(tf.reduce_max(image2) - tf.reduce_min(image2))

    
    label = tf.one_hot(truth, depth=2,on_value=1,off_value=0,axis=-1 )
    
    return image, label,r

def encoder_layer(x_con, channels, name,training, pool=True):
    
    with tf.name_scope("encoder_block_{}".format(name)):
        for i in range(channels):
            
            x = tf.layers.conv3d(x_con,12,kernel_size=[3,3,3],padding='SAME')
            x = tf.layers.dropout(x,rate=0.1,training=training)
            x = tf.layers.batch_normalization(x,training=training)
            x = tf.nn.relu(x)
            x_con = tf.concat([x,x_con], axis = 4)
        if pool is False:
            return x_con
        
        #x = tf.layers.conv3d(x_con,12,kernel_size=[1,1,1],padding='SAME')
        #x = tf.layers.dropout(x,rate=0.1,training=training)
        #x = tf.layers.batch_normalization(x,training=training,renorm=True)
        #x = tf.nn.relu(x)
        pool = tf.layers.max_pooling3d(x_con,pool_size = [2,2,1], strides=[2,2,1],data_format='channels_last')

        return x_con, pool
def decoder_layer(input_, x, ch, name, upscale = [2,2,2]):
        

    up = tf.layers.conv3d_transpose(input_,filters=12,kernel_size = [2,2,1],strides = [2,2,1],padding='SAME',name='upsample'+str(name), use_bias=False)
    up = tf.concat([up,x], axis=-1, name='merge'+str(name))
    return up

    

class DUnet():
    def __init__(self, x, training):
        #self.filters = filters
        self.training = training
        self.model = self.DU_net(x)

    
    def DU_net(self,input_):
        skip_conn = []


        conv1, pool1 = encoder_layer(input_,channels=2,name="encode_"+str(1),training=self.training, pool=True)
        conv2, pool2 = encoder_layer(pool1,channels=4,name="encode_"+str(2),training=self.training, pool=True)
        conv3, pool3 = encoder_layer(pool2,channels=6,name="encode_"+str(3),training=self.training, pool=True)
        conv4, pool4 = encoder_layer(pool3,channels=8,name="encode_"+str(4),training=self.training, pool=True)
        conv5, pool5 = encoder_layer(pool4,channels=10,name="encode_"+str(5),training=self.training, pool=True)
        conv6 = encoder_layer(pool5,channels=12,name="encode_"+str(5),training=self.training, pool=False)
        up1 = decoder_layer(conv6,conv5,10,name=1)
        conv7 = encoder_layer(up1,channels=10,name="conv"+str(6),training=self.training, pool=False)
        up2 = decoder_layer(conv7,conv4,8,name=2)
        conv8 = encoder_layer(up2,channels=8,name="encode_"+str(7),training=self.training, pool=False)
        up3 = decoder_layer(conv8,conv3,6,name=3)
        conv9 = encoder_layer(up3,channels=6,name="encode_"+str(8),training=self.training, pool=False)
        up4 = decoder_layer(conv9,conv2,4,name=4)
        conv10 = encoder_layer(up4,channels= 4,name="encode_"+str(9),training=self.training, pool=False)
        up5 = decoder_layer(conv10,conv1,2,name=5)
        conv11 = encoder_layer(up5,channels= 2,name="encode_"+str(10),training=self.training, pool=False)



        score = tf.layers.conv3d(conv11,2,(1,1,1),name='logits',padding='SAME')

        return score
def cost_dice(logits, labels,name='cost'):
    with tf.name_scope('cost'):
        eps = 1e-5
        
        logits = tf.cast(logits,tf.float32)
        labels1 = labels[...,1]
        logits1 = logits[...,1]
        log1 = tf.reshape(logits1,[1,-1])
        
        labels1 = tf.reshape(labels1,[1,-1])
        
        inte1 = tf.multiply(log1,labels1)
        inter1 = eps + tf.reduce_sum(inte1)
        union1 =  tf.reduce_sum(log1) + tf.reduce_sum(labels1)+eps

        

        loss = 1-tf.reduce_mean(2* inter1/ (union1))
        return loss

            
            
            
        
    


    
   
def DUnet_train():
    image_batch_placeholder = tf.placeholder("float32", shape=[1, 128,96,None, 1])
    label_batch_placeholder = tf.placeholder(tf.float32, shape=[1, 128,96,None,2])
    labels_pixels = tf.reshape(label_batch_placeholder, [-1, 2])
    
    
    training_flag = tf.placeholder(tf.bool)

    logits = DUnet(x = image_batch_placeholder, training=training_flag).model
    N,H,W,C,S = logits.get_shape().as_list()
    logit = tf.reshape(logits,(-1, 2))
    
    h = tf.squeeze(logits)
    
    

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_pixels, logits=logit)) #Loss function of softmax cross-entropy
    

    soft = tf.nn.softmax(h)


    regularzation_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    l = tf.squeeze(label_batch_placeholder)

    
    cost_loss = cost_dice(logits=soft, labels = l) #Loss function of dice-loss
    
    


    global_step = tf.Variable(0, name='global_step', trainable=False)

    
    learning_rate = 0.0001 #Static leanring rate
    tf.summary.scalar('learning_rate', learning_rate)

    

    saver = tf.train.Saver(max_to_keep=50)
    all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])



    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    
    total_loss = cost_loss + loss
    tf.summary.scalar('total_loss', total_loss)
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(total_loss, global_step=global_step)


    summary_op = tf.summary.merge_all()  # merge all summaries into a single "operation" which we can execute in a session


    filenames = []
    a = '/media/haben/D4CC01B2CC018FC2/aliasing_data/alis_train_1.tfrecords'
    b= '/media/haben/D4CC01B2CC018FC2/aliasing/alis_train_2.tfrecords'
    c = '/media/haben/D4CC01B2CC018FC2/aliasing/alis_train_3.tfrecords'
    filenames = [a, b, c]
    ########### Data input pipeline
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(map_func=parser, num_parallel_calls=3)
    dataset = dataset.batch(1)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.prefetch(100)
    dataset = dataset.repeat(200)

    iterator = dataset.make_one_shot_iterator()

    next_element = iterator.get_next()


    summary_writer = tf.summary.FileWriter("./log", sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    accuracy_accu = 0


    checkpoint = tf.train.get_checkpoint_state("./alias_weights") #Folder that weights are loaded from
    if(checkpoint != None):
        tf.logging.info("Restoring full model from checkpoint file %s",checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)

    coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(coord=coord, sess = sess)


    check_points = 800
    for epoch in range(200):

       
        for check_point in tqdm(range(2997),    
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
            #image, label, image2, mask, og, venc
            image_out, truth, r = sess.run(next_element)
            
            image_out = np.expand_dims(image_out,axis=4)
            
            
            _, training_loss, other_loss, _global_step, summary = sess.run([train_step, total_loss, cost_loss, global_step, summary_op],
                feed_dict={image_batch_placeholder: image_out,
                label_batch_placeholder: truth,
                training_flag: True})


            if(bool(check_point%1498 == 0) & bool(check_point != 0)):
                print(_)
                print("global step: ", _global_step)
                print("training loss: ", training_loss)
                print("other_loss:", other_loss)

                summary_writer.add_summary(summary, _global_step)

        

        saver.save(sess, "./alias_weights/hb.ckpt", _global_step) #Folder that weights are saved
        
        
        
        


    coord.request_stop()
    #coord.join(threads)
    sess.close()
    return 1
    



def main():
    tf.reset_default_graph()
    DUnet_train()



if __name__ == '__main__':
    main()


