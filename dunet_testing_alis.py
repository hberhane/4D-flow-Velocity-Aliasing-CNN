from dense_unet_v6_alis import DUnet
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
import tensorlayer as tl
import scipy.io as io
import statistics
import pickle
import scipy.ndimage.morphology as scipy

"""
Current Implementation of the testing script generates the unwrapped 4d flow data and saves it as a mat file;
Dice scores are calculated outside

If people would like, I can update the script to have it (1) save the mask of detected alias voxels and (2) calculate dice scores
"""


def feed_data():
    data_path = '/media/haben/My Passport1/Autopreprocessing/alis_sim_40_1.tfrecords'  # address to save the hdf5 file
    feature = {'test/image': tf.FixedLenFeature([], tf.string),
               #'test/label': tf.FixedLenFeature([], tf.string),
               'test/depth': tf.FixedLenFeature([], tf.int64),
               'test/height': tf.FixedLenFeature([], tf.int64),
               'test/width': tf.FixedLenFeature([], tf.int64),
               'test/venc': tf.FixedLenFeature([], tf.int64),
               #'test/seg': tf.FixedLenFeature([], tf.string),
               'test/phases': tf.FixedLenFeature([], tf.int64)}
    
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path])
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    height = tf.cast(features["test/height"], tf.int32)
    venc = tf.cast(features["test/venc"], tf.int64)
    width = tf.cast(features["test/width"], tf.int32)
    depth = tf.cast(features["test/depth"], tf.int32)
    phases = tf.cast(features["test/phases"], tf.int32)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['test/image'], tf.float32)
    #seg = tf.decode_raw(features['test/seg'], tf.uint8)

    # Cast label data into data type
    #label = tf.decode_raw(features['test/label'], tf.uint8)
    

    # Reshape image data into the original shape
    image = tf.reshape(image, [height, width, depth])

    image = tf.image.resize_image_with_crop_or_pad(image, 160, 96)

    
    image = tf.cast(image,tf.float32)
    #seg = tf.cast(seg,tf.float32)


    image = tf.expand_dims(image,axis = 0)

    #label = tf.expand_dims(label,axis = 0)
    image2 = image
    image = tf.expand_dims(image,axis = 4)
    image = (image - tf.reduce_min(image))/(tf.reduce_max(image) - tf.reduce_min(image)) #input to CNN; image2 used as basis of unwrapping

    #label = tf.expand_dims(label,axis = 4)
    q = tf.FIFOQueue(capacity=50, dtypes=[tf.float32,tf.float32, tf.float32])
    enqueue_op = q.enqueue_many([image,image2])
    #image, label = q.dequeue()
    qr = tf.train.QueueRunner(q,[enqueue_op])

    return image,image2, depth, venc, phases

    ##Dice calculation: currently not used
    #Same as training script except dice value is outputted instead of 1-dice
def cost_dice(logits, labels, seg, name='cost'):
    with tf.name_scope('cost'):
        eps = 1e-5
        #N,H,W,C,J = labels.get_shape()
        logits = tf.nn.softmax(logits)
        logits = logits[...,1]>=0.2
        #logits = logits[...,1]
        logits = tf.cast(logits,tf.float32)
        logits = tf.multiply(logits,seg)
        labels = labels
        labels = tf.multiply(labels,seg)

        log = tf.reshape(logits,[1,-1])
        
        labels = tf.reshape(labels,[1,-1])
        
        inte = tf.multiply(log,labels)
        inter = eps + tf.reduce_sum(inte)
        union =  tf.reduce_sum(labels)+eps
        loss = tf.reduce_mean(inter/ (union))
        #loss = 1- loss
        return loss


def DUnet_test():

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, 160,96,None, 1])
    image_batch_placeholder2 = tf.placeholder(tf.float32, shape=[None, 160,96,None])

    
    

    
    training_flag = tf.placeholder(tf.bool)
    image_batch,image_batch2,depth, venc, phase = feed_data()


    

 

    logits = DUnet(x = image_batch_placeholder, training=training_flag).model
    
    llogits = tf.nn.softmax(logits)
    
    

    checkpoint = tf.train.get_checkpoint_state('/media/haben/D4CC01B2CC018FC2/aliasing/alias_train')
    saver = tf.train.Saver()

    d = 0

    config = tf.ConfigProto(log_device_placement=False)
    all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])

    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tf.logging.info("Restoring full model from checkpoint file %s",checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        accuracy_accu = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)
        ave = []
        ma = []
        tru = []
        use = []
        gtruth = []
        low = []
        low_true = []
        low_loss = []
        new_data = []
        dice = []
        phases = sess.run(phase)
        print(phases)
        for i in tqdm(range(int(phases))):
            image_out,alias, _, Venc = sess.run([image_batch,image_batch2, depth, venc])
            
            _, llogit = sess.run([logits, llogits], feed_dict={image_batch_placeholder: image_out,

                                                                                    #label_batch_placeholder: truth,
                                                                                    image_batch_placeholder2: alias,
                                                                                    training_flag: True})
            
            input_ = np.squeeze(image_out)
            test = np.squeeze(alias)
            #print(input_.dtype)
            #print(test.dtype)
            #plt.imshow(input_[...,12])
            #plt.show()

            #plt.imshow(test[...,12])
            #plt.show()
            
            infer_out = llogit[...,1]
            
            data = np.squeeze(infer_out)
            
           
            h = data>0.5
            h = scipy.binary_fill_holes(h.astype(int))
            #print(np.max(alias))
            alias = np.squeeze(alias)
            new_alis = alias
            #plt.imshow(im)
            #plt.show()
            #plt.imshow(data)
            #plt.show()
            
            #Unrapping data based on mask
            
            for i in range(alias.shape[2]):
                for x in range(alias.shape[0]):
                    for y in range(alias.shape[1]):
                        if h[x,y,i] == 1:
                            value = alias[x,y,i]
                            new = value - (np.sign(value) * Venc*2/100)
                            new_alis[x,y,i] = new

                        else:
                            continue
            new_data.append(new_alis)
            
            

            io.savemat('./new_vel.mat',{'data':new_data})
            
        print(sess.run(all_trainable_vars))
        


        

        tf.train.write_graph(sess.graph_def, 'graph/', 'my_graph.pb', as_text=False)

        coord.request_stop()
        coord.join(threads)
        sess.close()
    return 0



def main():
    tf.reset_default_graph()

    DUnet_test()



if __name__ == '__main__':
    main()
