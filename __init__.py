# Academics: Carlos Henrique Ponciano da Silva e Vinicius Luis da Silva
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange


# load image and set standard size
def get_images(paths):
    _images = []
    for img in paths:
        _images.append(cv2.copyMakeBorder(cv2.imread(f'data/{img}.jpg'), 400, 400, 400, 400, cv2.BORDER_CONSTANT))
    return _images

# convert from RGB to grayscale
def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# get only the points with the best distance relationship
def get_only_good_matches(matches):
    return np.asarray(list(filter(lambda x: x[0].distance < 0.7 * x[1].distance, matches)))

# assembles to the final image
def get_mosaic(imageA, imageB, source, destine, _threshold=5.0):
    M, _masked = cv2.findHomography(source, destine, cv2.RANSAC, _threshold)
    h, w = to_gray(imageA).shape
    _new = cv2.warpPerspective(imageB, M, (w, h))
    return cv2.addWeighted(imageA, 0.5, _new, 0.8, 1)

# crop the final image for presentation
def crop(image, H=1000, W=500, X=350, Y=400):
    return image[Y:Y+H, X:X+W]

# plot the screen image and / or save it to disk
def plot(image, _show=True, _filename='output.jpg', _saveDisk=True, _crop=True):
    if _crop:
        image = crop(image)
    
    if _saveDisk:
        cv2.imwrite(_filename, image)

    if _show:   
        plt.imshow(image)
        plt.show()

if __name__ == "__main__":
    # name of the images in the data folder
    _images = get_images(['img1', 'img2', 'img3'])

    # loads Sift into memory
    _sift = cv2.xfeatures2d.SIFT_create()

    # repeats n pictures - 1
    for i in range(len(_images) - 1):
        # takes the last two vector images
        _imgA = _images.pop()
        _imgB = _images.pop()

        # obtain keypoints and descriptors
        _keypoints1, _descriptors1 = _sift.detectAndCompute(_imgB, None)
        _keypoints2, _descriptors2 = _sift.detectAndCompute(_imgA, None)
        
        # Get valid points
        _matcher = cv2.BFMatcher()
        _matches = get_only_good_matches(_matcher.knnMatch(_descriptors1, _descriptors2, k=2))

        # validates, converts and assembles the final mosaic
        if len(_matches[:,0]) >= 4:
            _src = np.float32([_keypoints1[m.queryIdx].pt for m in _matches[:, 0]]).reshape(-1,1,2)
            _dst = np.float32([_keypoints2[m.trainIdx].pt for m in _matches[:, 0]]).reshape(-1,1,2)
            _mosaic = get_mosaic(_imgA, _imgB, _src, _dst)
            _images.append(_mosaic) # adding the image results in the array and continues the loop
        else:
            raise AssertionError('Can’t find enough keypoints.')
    
    plot(_images[0])