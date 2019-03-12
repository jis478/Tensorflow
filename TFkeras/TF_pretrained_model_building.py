import tensorflow.keras.applications
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K

def keras_DenseNet(input_shape):
    base_DenseNet = tf.keras.applications.densenet.DenseNet201(include_top=False, 
                                            weights='imagenet', 
                                            input_tensor=Input(shape=input_shape),
                                            input_shape=input_shape,
                                            pooling = 'avg')
    
    x = base_DenseNet.output
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_DenseNet.input, outputs=predictions)

    return model
