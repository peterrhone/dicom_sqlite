# Description and Usage of sqlite_dicom2.py

## General overview

The sqlite_dicom2.py program's main purpose is to extract information from the dicom files and store it
in a way that is easily accessible for the AI training algorithm.

It works directly with the archived zip files, one zip for each patient. Each zip file contains a 
subdirectory labeled with patient ID (e.g. P001).
This directory contains multiple dicom files from several different CBCT scans/series, as well as the 
planing CT and labeled structures for each series.

The program extracts one or more medial saggital images from the dicom files that will be used for training the algorithm,
along with the corresponding contours for the organs-at-risk (OAR) - in this case bladder and rectum (as a binary mask).
It stores the extracted images and OAR masks in an sqlite database file.

Each entry into the sqlite database contains the following information:

1. *pat_id* = patient's ID, e.g. P001
2. *cbct_id* = the ID of the CBCT
3. *sag_slice_num* =  the slice number
4. *cbct_image* = blob containing the saggital image
5. *cbct_bladder_mask* = binary blob of same dim as *cbct_image* indicating whether pixel is in bladder
6. *cbct_rectum_mask* = binary blob of same dim as *cbct_image* indicating whether pixel is in rectum


## Detailed description of function

The program begins by presenting a chooser to select the folder containing the data (zip files).

For each zip, it

1. opens and sorts the info in an instance of class ZipFileContents. 
    Each series is stored within ZipFileContents in array series_dict of type DicomSeries.
    **DicomSeries** contains the following info for each series:
    - series UID,
    - files associated with the series.
        - image files (sorted by Instance Number)
        - a dictionary that uses the image ID (image UID) as key and the image filename as value
        - the related contour file containing all the structures
        - registration file if found
    - the number of slices (=the number of image files, 1 per slice).
    - it also has a info function (get_info()) that prints a report about the series

    E.g. for P001.zip contains:
    Total number of dicom files in P001.zip = 603.
    Total series in P001.zip = 6.
    Series UID: 1.2.752.243.1.1.20240426194524432.5260.40524
    Number of image files: 88, number of contour files: 1, number of registration files: 0
    Number of files in series = 89
    Series UID: 1.2.752.243.1.1.20240426194524327.4350.72003
    Number of image files: 88, number of contour files: 1, number of registration files: 0
    Number of files in series = 89
    Series UID: 1.2.752.243.1.1.20240426194524219.3440.41110
    Number of image files: 88, number of contour files: 1, number of registration files: 0
    Number of files in series = 89
    Series UID: 1.2.752.243.1.1.20240426194523821.2530.71308
    Number of image files: 88, number of contour files: 1, number of registration files: 0
    Number of files in series = 89
    Series UID: 1.2.752.243.1.1.20240426194523440.6000.50247
    Number of image files: 152, number of contour files: 1, number of registration files: 5
    Number of files in series = 158
    Series UID: 1.2.752.243.1.1.20240426194523710.1620.86402
    Number of image files: 88, number of contour files: 1, number of registration files: 0
    Number of files in series = 89
  
2. it loops through each series in each zip file
    1. it starts with the *bladder* and gets several medial sagittal images and stores these into the database.
    to do this:
        - ImageAndMaskExtractor is instantiated with one of the DicomSeries objects found and the zip reference
        - The function get_sag_contour_pixels is called, which returns an array of sagittal images and corresponding mask images
    2. it repeats the same process with the *rectum*.


BADFILES 
P004
ZipFileContents.get_image_given_img_ID: Image with ID 1.2.752.243.1.1.20240418152502255.7155.37736 not found.
coord2pixels(): Image with ID 1.2.752.243.1.1.20240418152502255.7155.37736 not found.
Error extracting contour for Bladder in series 1.2.752.243.1.1.20240418152502191.7100.21837: coord2pixels(): Image with ID 1.2.752.243.1.1.20240418152502255.7155.37736 not found.

P014
ZipFileContents.get_contour_data: There should be only one contour file per series. Found 0 files.
ImageAndMaskExtractor: No contour data found for series 1.2.752.243.1.1.20240418150848694.3935.76578
Series 1.2.752.243.1.1.20240418150848694.3935.76578 has no contours. Skipping...
ZipFileContents.get_contour_data: There should be only one contour file per series. Found 0 files.
ImageAndMaskExtractor: No contour data found for series 1.2.752.243.1.1.20240418150849519.4542.43116
Series 1.2.752.243.1.1.20240418150849519.4542.43116 has no contours. Skipping...
ZipFileContents.get_contour_data: There should be only one contour file per series. Found 0 files.
ImageAndMaskExtractor: No contour data found for series 1.2.752.243.1.1.20240418150849408.4451.24423
Series 1.2.752.243.1.1.20240418150849408.4451.24423 has no contours. Skipping...
ZipFileContents.get_contour_data: There should be only one contour file per series. Found 0 files.
ImageAndMaskExtractor: No contour data found for series 1.2.752.243.1.1.20240418150849302.4360.88404
Series 1.2.752.243.1.1.20240418150849302.4360.88404 has no contours. Skipping...
ZipFileContents.get_contour_data: There should be only one contour file per series. Found 0 files.
ImageAndMaskExtractor: No contour data found for series 1.2.752.243.1.1.20240418150849195.4269.70104
Series 1.2.752.243.1.1.20240418150849195.4269.70104 has no contours. Skipping...
ZipFileContents.get_contour_data: There should be only one contour file per series. Found 0 files.
ImageAndMaskExtractor: No contour data found for series 1.2.752.243.1.1.20240418150849089.4178.70825
Series 1.2.752.243.1.1.20240418150849089.4178.70825 has no contours. Skipping...
ZipFileContents.get_contour_data: There should be only one contour file per series. Found 0 files.
ImageAndMaskExtractor: No contour data found for series 1.2.752.243.1.1.20240418150848978.4087.47261
Series 1.2.752.243.1.1.20240418150848978.4087.47261 has no contours. Skipping...

P038
ZipFileContents:_get_all_series_from_zip(): Unknown SOPClassUID: RT Image Storage
ZipFileContents:_get_all_series_from_zip(): Unknown SOPClassUID: RT Image Storage
ZipFileContents:_get_all_series_from_zip(): Unknown SOPClassUID: RT Image Storage
ZipFileContents:_get_all_series_from_zip(): Unknown SOPClassUID: RT Image Storage

P039
ZipFileContents.get_contour_data: There should be only one contour file per series. Found 5 files.
ImageAndMaskExtractor: No contour data found for series 1.2.752.243.1.1.20240416153123388.3047.40456
Series 1.2.752.243.1.1.20240416153123388.3047.40456 has no contours. Skipping...

P040
ZipFileContents.get_contour_data: There should be only one contour file per series. Found 0 files.
ImageAndMaskExtractor: No contour data found for series 1.2.752.243.1.1.20240416153925717.6000.48285
Series 1.2.752.243.1.1.20240416153925717.6000.48285 has no contours. Skipping...

P042
ZipFileContents:_get_all_series_from_zip(): Unknown SOPClassUID: RT Image Storage
ZipFileContents:_get_all_series_from_zip(): Unknown SOPClassUID: RT Image Storage
ZipFileContents:_get_all_series_from_zip(): Unknown SOPClassUID: RT Image Storage
ZipFileContents:_get_all_series_from_zip(): Unknown SOPClassUID: RT Image Storage

P043
ZipFileContents.get_contour_data: There should be only one contour file per series. Found 5 files.
ImageAndMaskExtractor: No contour data found for series 1.2.752.243.1.1.20240416164414571.6000.46472
Series 1.2.752.243.1.1.20240416164414571.6000.46472 has no contours. Skipping...

P061
Series 1.2.752.243.1.1.20240423104709270.3382.48776 has no image files. Skipping...
ZipFileContents.get_contour_data: There should be only one contour file per series. Found 0 files.
ImageAndMaskExtractor: No contour data found for series 1.2.752.243.1.1.20240423104708108.2502.55781
Series 1.2.752.243.1.1.20240423104708108.2502.55781 has no contours. Skipping...


