#importing the relevant libraries
import cv2
import numpy as np 
import pywt
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt



#padd and crop function for images that are not sqare images
def make_square(img, color=(0, 0, 0)):
    height, width = img.shape[:2]
    size = min(height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return img[top:top+size, left:left+size]
#arnold transform
def arnold_transform(image,iterations=1):
    if image.shape[0] != image.shape[1]:
        print("WARNING: Not a sqaure image. Has been croppped.")
        image = make_square(image)
        #raise ValueError("Arnold Transform requires a square image")

    n = image.shape[0]
    transformed_image = np.zeros_like(image)
    for _ in range(iterations):
        for x in range(n):
            for y in range(n):
                new_x = (2*x + y) % n
                new_y = (x + y) % n
                transformed_image[new_x,new_y] = image[x,y]
        image = transformed_image.copy()
    return transformed_image

image_path = './images/sample.jpeg'
image = cv2.imread(image_path)
transform_image = arnold_transform(image)

plt.imshow(transform_image)
plt.savefig("./output/transform_ample.png")

