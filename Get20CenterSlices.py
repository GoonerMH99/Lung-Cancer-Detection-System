import SimpleITK as sitk
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import pydicom

# Read Folder That Contains all patients
data_dir = 'E:/UNI/GraduationProject/DSB3/OriginalData'
patients = os.listdir(data_dir)
# Read The label(cancer or not) excel sheet
labels = pd.read_csv('E:/UNI/GraduationProject/DSB3/stage1_labels.csv', index_col=0)


# "normal" list contains the ids of all normal(label = 0) patients
normal = []
for patient in patients:
    label = labels._get_value(patient, 'cancer')
    if label == 0:
        normal.append(patient)

# Get 20 center slices from each patient
i = 0
c = 0
while i < len(normal):
    path = 'E:/UNI/GraduationProject/DSB3/OriginalData/' + normal[i]   # Folder path for each normal patient
    slices = [pydicom.dcmread(path + "/" + filename) for filename in os.listdir(path)]  # Read all slices of the patient
    slices.sort(key=lambda x: int(x.InstanceNumber))    # Sort slices so that the clear ones are in the middles
    # Random number within the range of the 20 middle slices
    # rand = random.randrange((len(slices) // 2) - 10, (len(slices) // 2) + 10)
    j = (len(slices) // 2) - 10
    while j < (len(slices) // 2) + 10:
        img = sitk.GetImageFromArray(slices[j].pixel_array)  # Get the pixels array from the selected slice
        save_path = 'E:/UNI/GraduationProject/DSB3/20CenterSlicesFromEachPatient/RawNormalSample/P' + str(c + 1) + '.dcm'
        sitk.WriteImage(img, save_path)     # Save the selected pixels array as a dicom image in a new folder
        c += 1
        j += 1
    print(i)
    i += 1


# "cancer" list contains the ids of all cancerous(label = 1) patients
cancer = []
for patient in patients:
    label = labels._get_value(patient, 'cancer')
    if label == 1:
        cancer.append(patient)

# Get 20 center slices from each patient
i = 0
c = 0
for i in range(len(cancer)):
    path = 'E:/UNI/GraduationProject/DSB3/OriginalData/' + cancer[i]   # Folder path for each cancerous patient
    slices = [pydicom.dcmread(path + "/" + filename) for filename in os.listdir(path)]  # Read all slices of the patient
    slices.sort(key=lambda x: int(x.InstanceNumber))    # Sort slices so that the clear ones are in the middles
    # Random number within the range of the 20 middle slices
    # rand = random.randrange((len(slices) // 2) - 10, (len(slices) // 2) + 10)
    j = (len(slices) // 2) - 10
    while j < (len(slices) // 2) + 10:
        img = sitk.GetImageFromArray(slices[j].pixel_array)  # Get the pixels array from the selected slice
        save_path = 'E:/UNI/GraduationProject/DSB3/20CenterSlicesFromEachPatient/RawCancerSample/P' + str(c + 1) + '.dcm'
        sitk.WriteImage(img, save_path)     # Save the selected pixels array as a dicom image in a new folder
        c += 1
        j += 1
    print(i)
    i += 1
