#Pierre Coiron, Matt and Manuel
"""
Created on Thu Mar 26 16:59:11 2020

@author: peter
"""

import numpy as np
import cv2

#GBR NOT RGB
img = cv2.imread('FreemanHouseViennaVA.JPG')
'''
"Flatten" the 3D matrix to a 2D matrix,
every row is a pixel. Since there are over 1.7M pixels, there are 1.7M rows
The columns are Green, Blue and Red respectivly for every pixel
'''
flattenedImg = img.reshape((-1,3))

# convert to float32. The documentation demands it.
flattenedImg = np.float32(flattenedImg)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(flattenedImg,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
#we don't need the "center" matrix, as we will be replacing it with the "colors" matrix
#unfortunatly, I get erros if I don't have it

# Now convert back into uint8, and make original image
colors= np.array([[255, 0, 0],
                  [0, 255, 0],
                  [0, 0, 255]], np.uint8)

#flatten colors and reshape
kFitMatrix=colors[label.flatten()]
kFitImage=kFitMatrix.reshape((img.shape))

#graph it!
cv2.imshow('Original',img)
cv2.imshow('Segmented via pixels',kFitImage)
cv2.waitKey(0)
cv2.destroyAllWindows()