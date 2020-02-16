import cv2
import numpy as np
import os
import glob

MATCHES = 500
MIN_PERCENT = 90
BASE_ORIGINAL_PATH = 'Image_Processing_Challenge/augmented_new/Original/original.jpg'
BASE_TEST_PATH = 'Image_Processing_Challenge/augmented_new/Test_images/*.jpg'

#get aligned Images

def alignImages(testImg, orgImg, image):

    #print(image1, image2)
    #grayscale conversion
    testGray = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
    orgGray = cv2.cvtColor(orgImg, cv2.COLOR_BGR2GRAY)
    height, width  = orgGray.shape

    #ORB keypoints and descriptors
    orb = cv2.ORB_create(MATCHES)
    key1, des1 = orb.detectAndCompute(testGray, None)
    key2, des2 = orb.detectAndCompute(orgGray, None)

    #Compare features via Hamming Distance 
    #find = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    find = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = find.match(des1, des2, None)

    #Sort matches
    matches.sort(key = lambda x: x.distance)
    goodMatches = int(len(matches) * MIN_PERCENT)
    matches = matches[:goodMatches]

    #Draw Top Matches
    matched = cv2.drawMatches(testImg, key1, orgImg, key2, matches, None)
    cv2.imwrite('matched/matches'+str(image)+'.jpg', matched)

    #Good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = key1[match.queryIdx].pt
        points2[i, :] = key2[match.trainIdx].pt

    # Find homography
    homo, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
       
    # Use homography
    aligned = cv2.warpPerspective(testImg, homo, (width, height))
    
    return aligned

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

        print(str(image) +'-> Test image : ' + testImage, end=' ')
       
        aligned = alignImages(test, original, image)

        alignedImage = 'aligned/'+'aligned'+str(image) + '.jpg'
       
        print(' -> '+ alignedImage)
       
        cv2.imwrite(alignedImage, aligned)

        image += 1
