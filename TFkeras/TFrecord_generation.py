class TFrecordCreate():

    def __init__(self, tfrecord_name, images, labels, output_folder, x_dim, y_dim):

        s = time.time()
        self.tfrecord_name = tfrecord_name
        self.images = images
        self.labels = labels 
        self.x_dim = x_dim
        self.y_dim = y_dim
                
        print ('The number of images to be convereted into TFrecord: ', len(images))           

        if not os.path.exists(output_folder) or os.path.isfile(output_folder):
                os.makedirs(output_folder)

        tfrecord_filename = os.path.join(output_folder, tfrecord_name)         
        writer = tf.python_io.TFRecordWriter(tfrecord_filename)
           
        for image, label in zip(self.images, self.labels):
            
            image = Image.open(image).resize((self.x_dim, self.y_dim))
            image = np.array(image, np.float32)
            image = image / 255.
            image = image.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image': self._bytes_feature(image),   # image
                'label': self._int64_feature(label),         # label
            }))
            writer.write(example.SerializeToString())

        writer.close()
        print("Tfrecord generation finished")
        e = time.time() - s
        print("Total {} seconds".format(e))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
 
    def _read_csv(self, value):
        labels = pd.read_csv(self.label_file_name)
        labels['id'] = labels['id'].apply(lambda x: x + '.tif')
        return labels
