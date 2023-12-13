from skimage.io import imread
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from segmentCode import modelUnet
from classifyCode import modelClassify
import tensorflow as tf

model = modelClassify()
model.load_weights('efficientnetb3-sahil-weights.h5')

modelSeg = modelUnet()
modelSeg.load_weights('unet_model_weights.h5')

def segment_image(image_path):
    img = imread(image_path)[:,:,:3]
    img = resize(img, (256, 256), mode='constant', preserve_range=True)
    predMask = modelSeg.predict(np.expand_dims(img, axis=0),verbose=0)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title('Segmented Image')
    ax1.imshow(np.squeeze(predMask))
    ax1.axis('off')
    fig.savefig('segmented.png')

def classify_image(image_path):
    classes = {
        0: 'dyed-lifted-polyps',
        1: 'dyed-resection-margins',
        2: 'esophagitis',
        3: 'normal-cecum',
        4: 'normal-pylorus',
        5: 'normal-z-line',
        6: 'polyps',
        7: 'ulcerative-colitis'
    }
    img = imread(image_path)[:,:,:3]
    img = resize(img, (224,224), mode='constant', preserve_range=True)
    pred = model.predict(np.expand_dims(img, axis=0),verbose=0)
    predicted_class = np.argmax(pred[0])
    return classes[predicted_class]