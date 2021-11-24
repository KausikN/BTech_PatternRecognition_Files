'''
Assignment Q5
Q5. Consider the following images. Obtain the histograms for each of the images. Using a
suitable distance measure, find the distance between the query image and reference images.
'''

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Main Functions
def ReadImage(path, display=False):
    I = np.array(cv2.imread(path))
    if display:
        plt.imshow(I)
        plt.show()
    return I

def ImageHistogram(I, bins=list(range(0, 256)), display=True):
    if I.ndim == 3:
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    
    bins = sorted(bins)

    hist = {}
    for b in bins:
        hist[str(b)] = 0

    valMap = {}
    for i in tqdm(range(I.shape[0])):
        for j in range(I.shape[1]):
            if str(I[i, j]) in valMap.keys():
                hist[str(valMap[str(I[i, j])])] += 1
            else:
                minDist = -1
                minI = -1
                for bi in range(len(bins)):
                    dist = np.abs(bins[bi] - I[i, j])
                    if minI == -1 or minDist > dist:
                        minDist = dist
                        minI = bi
                    else:
                        break
                valMap[str(I[i, j])] = bins[minI]
                hist[str(valMap[str(I[i, j])])] += 1

    if display:
        histVals = []
        for b in bins:
            histVals.append(hist[str(b)])
        
        plt.title('Image Histogram')
        plt.subplot(1, 2, 1)
        plt.imshow(I, 'gray')
        plt.subplot(1, 2, 2)
        plt.bar(bins, histVals, width=1)
        plt.show()
    
    return hist

def BhattacharyyaDistance(P1, P2):
    dist = 0

    P1 = np.array(P1)
    P2 = np.array(P2)

    dist = -np.log(np.sum(np.sqrt(np.multiply((P1 / np.sum(P1)), (P2 / np.sum(P2))))))

    return dist

# Driver Code
# Params
Query_Path = 'Assignment1/Data/Query_image.jpg'
Ref1_Path = 'Assignment1/Data/Reference_image1.jpg'
Ref2_Path = 'Assignment1/Data/Reference_image2.jpg'

display = False
# Params

# RunCode
I_Query = ReadImage(Query_Path, display=False)
I_Ref1 = ReadImage(Ref1_Path, display=False)
I_Ref2 = ReadImage(Ref2_Path, display=False)

# Get Histograms
H_Query_dict = ImageHistogram(I_Query, bins=list(range(0, 256)), display=display)
H_Ref1_dict = ImageHistogram(I_Ref1, bins=list(range(0, 256)), display=display)
H_Ref2_dict = ImageHistogram(I_Ref2, bins=list(range(0, 256)), display=display)

H_Query = []
H_Ref1 = []
H_Ref2 = []
for i in range(0, 256):
    H_Query.append(H_Query_dict[str(i)])
    H_Ref1.append(H_Ref1_dict[str(i)])
    H_Ref2.append(H_Ref2_dict[str(i)])

# Bhattacharyya Distance
print("")
BhattacharyyaDist_1 = BhattacharyyaDistance(H_Query, H_Ref1)
print("Bhattacharyya Distance between Query and Reference 1:", BhattacharyyaDist_1)
BhattacharyyaDist_2 = BhattacharyyaDistance(H_Query, H_Ref2)
print("Bhattacharyya Distance between Query and Reference 2:", BhattacharyyaDist_2)