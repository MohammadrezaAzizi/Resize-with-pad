import numpy as np 
import tensorflow as tf
import cv2

image = tf.keras.utils.load_img(r"E:\Heart\FINAL Cropped\Cropped\ABASNIA, GAFAR_Run30_1.jpg", grayscale=False)

def resize_with_pad(image, target_width, target_height, pad_min, pad_max):
    """
    This functions checks out the width & height in target & current image and ratio. We wanna make sure that the ratio is intact.
    1- Get the current width and height. 
    2- Now all you have to do is to checkout if the width and height are bigger/smaller than the target value given in the input of our func.
    3- Needless to say that we leave the 3rd axis be as it does not effect the information of the image.
    4- transform this to 

    """
    height, width, depth = tf.shape(image)
    pad = np.random.randint(pad_min, pad_max)
    print(pad)
    h_pad = pad * (target_height/target_width)
    w_pad = pad * (target_width/target_height)
    img = np.pad(image, ((int(h_pad/2), int(h_pad/2)), (int(w_pad/2), int(w_pad/2)), (0, 0)),
            mode='constant',constant_values=0)
    img = tf.image.resize(img, [target_height, target_width], preserve_aspect_ratio=True)
    img = tf.image.per_image_standardization(img)
    cv2.imshow("img", np.array(img))
    cv2.imshow("image", np.array(image))
    cv2.waitKey(0)
    cv2.destroy_all_windows()

resize_with_pad(image, 200, 500, 50, 100)