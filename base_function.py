import numpy as np 
import tensorflow as tf

image = tf.keras.utils.load_img(r"E:\Heart\FINAL Cropped\Cropped\ABASNIA, GAFAR_Run30_1.jpg", grayscale=False)

def resize_with_pad(image, target_width, target_height, pad_range:tuple):
    """
    This functions checks out the width & height in target & current image and ratio. We wanna make sure that the ratio is intact.
    1- Get the current width and height. 
    2- Now all you have to do is to checkout if the width and height are bigger/smaller than the target value given in the input of our func.
    3- Needless to say that we leave the 3rd axis be as it does not effect the information of the image.
    4- transform this to 

    """
    height, width, depth = tf.shape(image)
    pad = np.random.random()*pad_range
    h_pad = pad * target_height/target_width
    w_pad = pad * target_width/target_height
    np.pad(image, ((int(h_pad/2), int(h_pad/2)), (int(w_pad/2), int(w_pad/2)), (0, 0)),
            mode='constant',constant_values=0)
    img = tf.image.resize(image, [target_height, target_width, depth], preserve_aspect_ratio=True)
    return img

resize_with_pad(image, 200, 50, (10,25))