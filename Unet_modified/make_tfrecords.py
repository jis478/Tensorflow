import tensorflow as tf
import numpy as np
import os
import os.path
import glob
from PIL import Image

class TFrecord_Create_For_Unet():

    def __init__(self,train_test,img_folder,label_name,img_type, tf_record_pre_fix, nx, ny):

        self.train_test = train_test
        self.img_folder = img_folder
        self.img_type = img_type
        self.label_name = label_name
        self.a_min= -np.inf
        self.a_max= np.inf
        self.nx = nx
        self.ny = ny

        files_examples = [name for name in glob.glob(os.path.join(self.img_folder,'*.'+ self.img_type)) if self.label_name not in name]
        files_examples.sort()
        
        files_gts = [name for name in glob.glob(os.path.join(self.img_folder,'*.'+ self.img_type)) if self.label_name in name]
        files_gts.sort()

        print ('original images: ', len(files_examples))
        print ('ground truth images: ', len(files_gts))
             
        _examples = np.asarray(files_examples)
        _gts = np.asarray(files_gts)

        output_directory=os.path.join(os.getcwd(),'unet_tfrecord')
         
        if not os.path.exists(output_directory) or os.path.isfile(output_directory):
                os.makedirs(output_directory)

        filename = os.path.join(output_directory, tf_record_pre_fix + '_{}.tfrecords'.format(train_test))
         
        writer = tf.python_io.TFRecordWriter(filename)
         
        for image, gt in zip(_examples, _gts):
            image = Image.open(image).resize((self.ny, self.nx))
            image = np.array(image, np.float32)
            image = self._process_data(image)
            image_raw = image.tostring()
            
            gt = np.array(Image.open(gt).convert("L").resize((self.ny, self.nx)), np.bool) 
            gt = self._process_labels(gt)
            gt_raw = gt.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': self._bytes_feature(image_raw),   # string
                'gt_raw': self._bytes_feature(gt_raw),         # string
            }))
            writer.write(example.SerializeToString())

        writer.close()
        print("Tfrecord generation finished")
  
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
      
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
 
    def _process_data(self, data): 
        data = data / 255.      
        return data

    def _process_labels(self, label): 
        nx = label.shape[0]
        ny = label.shape[1]
        labels = np.zeros((nx, ny, 2), dtype=np.float32)               
        labels[..., 1] = label
        labels[..., 0] = ~label
        return labels[..., 1]
       