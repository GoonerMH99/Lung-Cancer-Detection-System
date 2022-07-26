import SimpleITK as sitk
import cv2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import skimage
import os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing
from skimage.morphology import reconstruction, binary_closing
from skimage.measure import label, regionprops, perimeter
# from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
# from skimage import measure, feature
from skimage.segmentation import clear_border
# from skimage import data
from scipy import ndimage as ndi
from scipy import ndimage
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
import pydicom
import skimage


def get_segmented_lungs(im, plot=False):
    """
    This function segments the lungs from the given 2D slice.
    """
    plots = []
    if plot:
        f, plots = plt.subplots(8, 1, figsize=(5, 20))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < 604
    if plot:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = skimage.segmentation.clear_border(binary)
    if plot:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    separate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)
        plt.show()
    return im


def pre(imagepath):
    img = pydicom.dcmread(imagepath)
    # Get Spacing
    # spacing = np.array(list(img.PixelSpacing))

    # Calculate Resize Factor
    # RESIZE_SPACING = [2, 2]
    # resize_factor = spacing / RESIZE_SPACING
    # new_real_shape = img.pixel_array.shape * resize_factor
    # new_shape = np.round(new_real_shape)
    # real_resize = new_shape / img.pixel_array.shape
    # new_spacing = spacing / real_resize

    # Resizing
    # lung_img = scipy.ndimage.zoom(np.array(img.pixel_array), real_resize)

    # Segment the lung structure
    # lung_img = lung_img + 1024
    # lung_img = lung_img - 1024
    # plt.imshow(lung_mask, cmap=plt.cm.bone)
    # plt.show()
    # plot_ct_scan(lung_mask)
    # plot_ct_scan(lung_img)
    lung_mask = get_segmented_lungs(img.pixel_array)
    return lung_mask


path = 'E:/UNI/GraduationProject/DSB3/20CenterSlicesFromEachPatient/RawCancerSample'
cancerslices = [path + "/" + filename for filename in os.listdir(path)]
pixeldata = []
slice_data = []
slicecounter = 0
c = 0
for slc in cancerslices:
    ppimg = pre(slc)
    ppimg = cv2.resize(ppimg, (50, 50))

    slice_data.append(np.array(ppimg))

    if slicecounter >= 19:
        pixeldata.append([np.array(slice_data), np.array([1, 0])])
        print("slc" + str(len(slice_data)))
        print("pix" + str(len(pixeldata)))
        slice_data = []
        slicecounter = 0
        c += 1
        print(c)
        continue
    c += 1
    slicecounter += 1
    print(c)


path = 'E:/UNI/GraduationProject/DSB3/20CenterSlicesFromEachPatient/RawNormalSample'
normalslices = [path + "/" + filename for filename in os.listdir(path)]
c = 0
slicecounter = 0
slice_data = []
for slc in normalslices:
    ppimg = pre(slc)
    ppimg = cv2.resize(ppimg, (50, 50))

    slice_data.append(np.array(ppimg))

    if slicecounter >= 19:
        pixeldata.append([np.array(slice_data), np.array([0, 1])])
        print("slc" + str(len(slice_data)))
        print("pix" + str(len(pixeldata)))
        slice_data = []
        slicecounter = 0
        c += 1
        print(c)
        continue
    c += 1
    slicecounter += 1
    print(c)

np.save('PreProcessed20Center-pixdata-50-50-20.npy', pixeldata)

"""------------------------------------------------------------------------------------------------------------------"""
lst = []
for i in range(50):
    lst.append([0]*50)
blank_slc = np.array(lst)

newpx_normal = []
newpx_cancer = []
n = 0
for i in range(len(pixeldata)):
    if list(pixeldata[i][1]) == [0, 1]:
        for j in range(20):
            if np.array(pixeldata[i][0][j]).any() == np.array(blank_slc).any():
                continue
            newpx_cancer.append(pixeldata[i][0][j])
    else:
        if n < 8380:
            for j in range(20):
                if np.array(pixeldata[i][0][j]).any() == np.array(blank_slc).any():
                    continue
                newpx_normal.append(pixeldata[i][0][j])
                n += 1


np.save('PreProcessed-Normal-SingleSlices-pixdata(centered)(extended).npy', newpx_normal)
np.save('PreProcessed-Cancer-SingleSlices-pixdata(centered)(extended).npy', newpx_cancer)
