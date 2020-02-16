import cv2
import numpy as np
import os
import glob

MATCHES = 500
MIN_PERCENT = 0.15
BASE_ORIGINAL_PATH = 'Image_Processing_Challenge/augmented_new/Original/original.jpg'
BASE_TEST_PATH = 'Image_Processing_Challenge/augmented_new/Test_images/*.jpg'

#get aligned Images

def alignImages(image1, image2):

    #print(image1, image2)
    #grayscale conversion
    im1Gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    #ORB and descriptors
    orb = cv2.ORB_create(MATCHES)
    key1, des1 = orb.detectAndCompute(im1Gray, None)
    key2, des2 = orb.detectAndCompute(im2Gray, None)

    #Compare features via points    
    find = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = find.match(des1, des2, None)

    #Sort matches
    goodMatches = int(len(matches) * MIN_PERCENT)
    matches = matches[:goodMatches]

    #Draw Top Matches
    matched = cv2.drawMatches(image1, key1, image2, key2, matches, None)
    cv2.imwrite('matched.jpg', matched)

    #Good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = key1[match.queryIdx].pt
        points2[i, :] = key2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
       
    # Use homography
    height, width, channels = image2.shape
    im1Reg = cv2.warpPerspective(image1, h, (width, height))
    
    return im1Reg, h

if __name__ == '__main__':
   
    print('Starting process')
    originalImage = BASE_ORIGINAL_PATH
    original = cv2.imread(originalImage, cv2.IMREAD_COLOR)
    print(originalImage) 

    files = glob.glob(BASE_TEST_PATH)
    print(*files, sep='\n')
    
    image = 1
    for f in files:
       testImage = f
    
       test = cv2.imread(testImage, cv2.IMREAD_COLOR)

       print(str(image) +'-> Test image : ' + testImage)
       imReg, h = alignImages(test, original)
       alignedImage = 'aligned/'+'aligned'+str(image) + '.jpg'
       print('Aligned Image : '+ alignedImage)
       cv2.imwrite(alignedImage, imReg)

       image += 1
