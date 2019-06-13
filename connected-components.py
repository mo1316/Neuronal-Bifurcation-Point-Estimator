import numpy as np
import cv2 as cv
from skimage.morphology import label
from skimage.morphology import skeletonize_3d
from math import sqrt
from math import pi
import xlsxwriter

# Temporary array for interface storage
arrays = []
# List of soma keypoints
kpList = []
# List of CCL keypoint values
valList = []
# List of endpoint values
epList = []


# Read in images from stack:
# CHANGE PATH ACCORDINGLY
for number in range(403, 435):
    strr = "/Users/markolchanyi/Desktop/Imperial_Year_3/NeuroProject/Ch2_Stitched_401_600_TIFF_50/Stitched_Z" + str(number) + ".tif"
    a = cv.imread(strr, cv.IMREAD_GRAYSCALE)
    arrays.append(a)

data = np.array(arrays)
data2 = data.copy()

# Simple blob detector to determine number, size, and location of soma points
# NEED TO SPECIFY:
# MIN/MAX AREA
# CIRCULARITY
for num in range(0, data.__len__()-1):
    params = cv.SimpleBlobDetector_Params()
    params.minThreshold = 200
    params.maxThreshold = 255
    # Filter by Area
    params.filterByArea = True
    params.minArea = 65
    params.maxArea = 500
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.2
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.8
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.43

    detector = cv.SimpleBlobDetector_create(params)
    # Current slice of 3d numpy array
    img = data[num, :, :]

    cv.medianBlur(img, 21, img)
    cv.bitwise_not(img, img)

    # Detect blobs.
    keypoints = detector.detect(img)
    # 'x' represents x coordinate of soma
    # 'y' represents y coordinate of soma
    for keyPoint in keypoints:
        x = keyPoint.pt[0]
        y = keyPoint.pt[1]
        s = keyPoint.size
        kpList.append([int(x), int(y), num, (sqrt(s/pi) + 2)])

# Canny edge detection to isolate axonal/dendritic branches...NEED TO SPECIFY MIN
# EXPECTED INTENSITY OF BRANCHES
for num in range(0, data2.__len__()-1):
    imgTmp = cv.Canny(data2[num, :, :], 55, 210, apertureSize=3, L2gradient=False)
    cv.threshold(imgTmp, 0, 255, cv.THRESH_OTSU, imgTmp)
    kernel = np.ones((3, 3), np.uint8)
    cv.dilate(imgTmp, kernel, imgTmp, iterations=1)
    cv.medianBlur(imgTmp, 3, imgTmp)
    data2[num, :, :] = imgTmp
skel = skeletonize_3d(data2)
skel = np.delete(skel, -1, axis=0)

# Post-edge detection, dilation and median filtering, locations with soma pts get hollowed out
# therefore filling pts with circles of radii similar to radii of original somas (from blob detection)
# is required:
for num in range(0, len(kpList)):
    cv.circle(skel[kpList[num][2], :, :], (kpList[num][0], kpList[num][1]), 9, (255, 255, 255), -1)

lb, num = label(skel, neighbors=8, return_num=True)
dout = np.array(lb)
dout = dout.astype(np.uint16)

# Post-labelling involves appending list of keypoints with associated intensity labels
# and creation of separate list of strictly labels
for num in range(0, len(kpList)):
    kpList[num] = [kpList[num][0], kpList[num][1], kpList[num][2], dout[kpList[num][2], kpList[num][1], kpList[num][0]]]
    valList.append(dout[kpList[num][2], kpList[num][1], kpList[num][0]])
valList = list(set(valList))

# Removes all labeled objects that are not associated with soma pts (neurones)
for num1 in range(0, dout.shape[0]):
    for num2 in range(0, dout.shape[1]):
        for num3 in range(0, dout.shape[2]):
            if dout[num1, num2, num3] not in valList:
                dout[num1, num2, num3] = 0

# Calculates endpoints based on 26 nearest neighbors:
# If pixel of certain label (intensity) contains one 26-connected neighbor
# with the same label then it gets labelled as an endpoint
for num1 in range(1, (dout.shape[0]-1)):
    for num2 in range(1, (dout.shape[1]-1)):
        for num3 in range(1, (dout.shape[2]-1)):
            if dout[num1, num2, num3] > 0:
                nbVal = 0
                for nb1 in range(num1-1, num1+2):
                    for nb2 in range(num2-1, num2+2):
                        for nb3 in range(num3-1, num3+2):
                            nbVal = nbVal + dout[nb1, nb2, nb3]
                if nbVal == 2*(dout[num1, num2, num3]):
                    cv.rectangle(dout[num1, :, :], (num3-2, num2-2), (num3+2, num2+2), (255, 255, 255), 1)
                    epList.append([dout[num1, num2, num3]])

# OUTPUT TO EXCEL SHEET:
# Workbook() takes one, non-optional, argument
# which is the filename that we want to create.
workbook = xlsxwriter.Workbook('bifrucation_points.xlsx')

worksheet = workbook.add_worksheet()

worksheet.write('A1', 'Cell Label')
for row in range(0, len(valList)):
    worksheet.write(row+1, 0, valList[row])
    worksheet.write(row+1, 2, epList.count(valList[row]))

worksheet.write('B1', 'Location')
worksheet.write('C1', 'Bifurcation Pt. Number')


workbook.close()

# Export image to desired path:
# CHANGE DESTINATION PATH ACCORDINGLY
for num in range(0, dout.shape[0]):
   cv.imwrite('/Users/markolchanyi/Desktop/testImages/img' + str(num) + '.jpg', dout[num, :, :])

# Create and export binary image with each neuron displayed with
# max intensity for visualization purposes:
for num1 in range(0, dout.shape[0]):
    for num2 in range(0, dout.shape[1]):
        for num3 in range(0, dout.shape[2]):
            if dout[num1, num2, num3] > 0:
                dout[num1, num2, num3] = 65536

for num in range(0, dout.shape[0]):
   cv.imwrite('/Users/markolchanyi/Desktop/binary/img' + str(num) + '.jpg', dout[num, :, :])