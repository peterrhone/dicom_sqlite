import pydicom as dicom
# from dicom_contour.contour import *
from scipy.sparse import csc_matrix
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from dicomseries import DicomSeries
from zipfilecontents import ZipFileContents
from PIL import Image
from matplotlib import pyplot as plt
from skimage.draw import polygon
from skimage.draw import line
import numpy as np
import numba as nb
from shapely.geometry import Polygon
import alphashape

class ImageAndMaskExtractor:
    def __init__(self, series: DicomSeries, zip_contents: ZipFileContents):
        self.zip_contents = zip_contents
        self.series = series
        self.image_and_mask = {}
        # self.__extract_image_and_mask()
        self.contour_data = self.zip_contents.get_contour_data(self.series)

        #check if contour data is empty
        self.has_contours = False
        if self.contour_data and hasattr(self.contour_data, 'ROIContourSequence'):
            self.roi_names_dict = self.get_roi_names(self.contour_data)
            self.has_contours = True
        else:
            print(f"ImageAndMaskExtractor: No contour data found for series {self.series.get_series_id()}")
            self.roi_names_dict = {}

        self.ctr = 0

    def get_roi_names(self, contour_data):
        roi_seq_dict = {}
        if 'ROIContourSequence' in contour_data:
            for roi_seq in contour_data.StructureSetROISequence:
                roi_name = roi_seq.ROIName
                roi_number = roi_seq.ROINumber
                if roi_name not in roi_seq_dict.keys():
                    # print(f"roi_name: {roi_name}, roi_number: {roi_number}")
                    roi_seq_dict[roi_name] = roi_number
        return roi_seq_dict

    def get_sag_aspect_ratio(self):
        if self.series.get_num_image_files() == 0:
            print(f"get_sag_aspect_ratio(): No image files found in the series")
            return None
        img = self.zip_contents.get_image_given_filename(list(self.series.get_image_files())[0])
        x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])
        layer_spacing = float(img.SliceThickness)
        sagittal_aspect_ratio = y_spacing / (2*layer_spacing) #I don't know why I need to multiply layer spacing by a factor, but it works
        # print(f"y_spacing: {y_spacing}, layer_spacing: {layer_spacing}, ratio: {sagittal_aspect_ratio}")
        return sagittal_aspect_ratio
    
    def __find_midline(self, series_id):
        pass

    def __extract_image_and_mask(self):
        pass

    def get_voxel_spacing(self):
        if self.series.get_num_image_files() == 0:
            print(f"get_voxel_spacing(): No image files found in the series")
            return None
        img = self.zip_contents.get_image_given_filename(list(self.series.get_image_files())[0])
        x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])
        layer_spacing = float(img.SliceThickness)
        voxel_spacing = (x_spacing, y_spacing, layer_spacing)
        return voxel_spacing

    def get_contour_names(self):
        return self.roi_names_dict
    
    def generate_filled_in_mask_surface_from_contour(self, closed_surface_mask_img):
        #use scipy.ndimage.binary_fill_holes
        return binary_fill_holes(closed_surface_mask_img)
        # pass  

    def get_sag_contour_pixels(self, roi_name, total_layers=88):
        """
        Extract sagittal contour pixels with context layers, 
        with contour centered in volume
        Args:
            roi_name: Name of ROI to extract
            total_layers: Total number of layers to return
        """

        # Get full volume
        ct_img_vol, mask_vol, img_ids = self.get_img_and_mask_volumes(roi_name)

        # convert the arrays to np arrays
        ct_img_vol = np.array(ct_img_vol)
        mask_vol = np.array(mask_vol)

        # Find contour bounds in cc direction
        contour_layers = np.any(mask_vol > 0, axis=(1, 2))
        min_layer = np.where(contour_layers)[0][0]
        max_layer = np.where(contour_layers)[0][-1] 

        # contour center and padding
        contour_center = (min_layer + max_layer) // 2
        half_window = total_layers // 2

        # Calculate layer range
        start_layer = max(0, contour_center - half_window) 
        end_layer = min(start_layer + total_layers, ct_img_vol.shape[0])

        if end_layer - start_layer < total_layers:
            # Add padding to the caudal side
            start_layer = max(0, end_layer - total_layers)

        # Extract layers
        ct_img_vol = ct_img_vol[start_layer:end_layer]
        mask_vol = mask_vol[start_layer:end_layer]

        # swap the axes to get the correct orientation
        ct_img_vol = np.swapaxes(ct_img_vol, 0, 2)
        mask_vol = np.swapaxes(mask_vol, 0, 2)
        # rotate the images by 90 degrees and flip horizontally so that it's a saggital image 
        #  with the following orientation: (typical orientation in RT)
        #           Cranial ⭡
        # posterior     ⭤     anterior
        #           Caudal  ⭣
        ct_img_vol = np.rot90(ct_img_vol, k=1, axes=(2, 1))
        mask_vol = np.rot90(mask_vol, k=1, axes=(2, 1))
        ct_img_vol = np.flip(ct_img_vol, axis=1)
        mask_vol = np.flip(mask_vol, axis=1)

        #process each mask slice to ensure contiguous mask
        processed_masks = []
        for i in range(mask_vol.shape[0]):
            processed_mask = self.connect_contour_points_and_fill(mask_vol[i])
            processed_masks.append(processed_mask)
        
        processed_masks = np.array(processed_masks)

        # create a new list of tuples (ct_img, mask_img, img_id), but now with the sagittal slices
        ids = range(len(ct_img_vol))
        sag_img_contour_arrays = [(ct_img_vol[i], processed_masks[i], ids) \
                                  for i in range(ct_img_vol.shape[0])]
        return sag_img_contour_arrays
    
    # def get_img_and_mask_volumes(self, roi_name):
    #     roi_seq = [roi_seq for roi_seq in self.contour_data.ROIContourSequence \
    #                if roi_seq.ReferencedROINumber == self.roi_names_dict[roi_name]]
    #     contours = [contour for contour in roi_seq[0].ContourSequence]
    #     img_contour_arrays = [self.coord2pixels(contour ) for contour in contours]
    #     ct_img_vol = []
    #     mask_vol = []
    #     img_ids = []
    #     for i in range(len(img_contour_arrays)):
    #         ct_img,mask_img,img_id = img_contour_arrays[i]
    #         ct_img_vol.append(ct_img)
    #         mask_vol.append(mask_img)
    #         img_ids.append(img_id)

    #     return ct_img_vol, mask_vol, img_ids 
    def get_img_and_mask_volumes(self, roi_name):
        # Get contour data
        roi_seq = [roi_seq for roi_seq in self.contour_data.ROIContourSequence 
                   if roi_seq.ReferencedROINumber == self.roi_names_dict[roi_name]]
        contours = [contour for contour in roi_seq[0].ContourSequence]
    
        # Get all image files sorted by position
        all_images = list(self.series.get_image_files())
        all_image_ids = [self.zip_contents.get_image_given_filename(img).SOPInstanceUID 
                        for img in all_images]
    
        # Create volumes with all slices
        ct_img_vol = []
        mask_vol = []
    
        # Process contours to get mask info
        contour_data = {}
        for contour in contours:
            img_arr, mask_img, img_id = self.coord2pixels(contour)
            contour_data[img_id] = (img_arr, mask_img)
    
        # Fill volumes including empty slices
        for img_file, img_id in zip(all_images, all_image_ids):
            img = self.zip_contents.get_image_given_filename(img_file)
            if img_id in contour_data:
                # Use existing contour data
                ct_img_vol.append(contour_data[img_id][0])
                mask_vol.append(contour_data[img_id][1])
            else:
                # Create empty mask
                ct_img_vol.append(img.pixel_array)
                mask_vol.append(np.zeros_like(img.pixel_array))
    
        return ct_img_vol, mask_vol, all_image_ids

    def get_contour_pixels(self, roi_name): # = cfile2pixels function eqivalent
        roi_seq = [roi_seq for roi_seq in self.contour_data.ROIContourSequence \
                   if roi_seq.ReferencedROINumber == self.roi_names_dict[roi_name]]
        # print(f"type(roi_seq): {type(roi_seq)}, of length: {len(roi_seq)}")
        contours = [contour for contour in roi_seq[0].ContourSequence]
        img_contour_arrays = [self.coord2pixels(contour) for contour in contours]
        # print(f"get_contour_pixels: len(img_contour_arrays): {len(img_contour_arrays)}")
        return img_contour_arrays

    def coord2pixels(self, contour):
        self.ctr += 1
        
        contour_coord = contour.ContourData

        len_contour = len(contour_coord)
        x0 = contour_coord[len_contour-3]
        y0 = contour_coord[len_contour-2]
        z0 = contour_coord[len_contour-1]
    #    print(f"x0: {x0}, y0: {y0}, z0: {z0}")
        coord = []
        for i in range(0, len_contour, 3):
            x = contour_coord[i]
            y = contour_coord[i+1]
            z = contour_coord[i+2]
            l = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
            l = int(np.ceil(l*2)+1)
            for j in range(1, l+1):
                coord.append([(x-x0)*j/l+x0, (y-y0)*j/l+y0, (z-z0)*j/l+z0])
            x0 = x
            y0 = y
            z0 = z
            
            #get the corresponding image for given contour
        img_ID = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
        img =  self.zip_contents.get_image_given_img_ID(self.series, img_ID)
        if img is None:
            print(f"coord2pixels(): Image with ID {img_ID} not found.")
            raise ValueError(f"coord2pixels(): Image with ID {img_ID} not found.")
        img_arr = img.pixel_array

        x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

        origin_x, origin_y, _ = img.ImagePositionPatient
        
        # aspect_ratio = x_spacing / y_spacing
        layer_spacing = float(img.SliceThickness)
        # voxel_spacing = (x_spacing, y_spacing, layer_spacing)
        # sagittal_aspect_ratio = y_spacing / layer_spacing    # 0.90802035... / 1.98848901...  = 0.456638

        pixel_coords = [(np.round((y - origin_y) / y_spacing), np.round((x - origin_x) / x_spacing)) for x, y, _ in coord]
        # remove duplicate pixel coordinates
        pixel_coords = list(set(pixel_coords))
        valid_pixel_coords = [(i, j) for i, j in set(pixel_coords) if 0 <= i < img_arr.shape[0] and 0 <= j < img_arr.shape[1]]
        if(len(valid_pixel_coords) < len(pixel_coords)):
            print(f"coord2pixels(): Literally impossible: Found {len(pixel_coords) - len(valid_pixel_coords)} contour coords that are outside the image bounds")
        pixel_coords = valid_pixel_coords

        rows = []
        cols = []
        for i,j in pixel_coords:
            rows.append(i)
            cols.append(j)
        try:
            contour_arr = csc_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int8, shape=(img_arr.shape[0], img_arr.shape[1])).toarray()
            contour_mask = contour_arr

        except Exception as e:
            print(f"coord2pixels(): CRAP! Exception: {e}")

        return img_arr, contour_mask, img_ID

    def convex_hull_polygon_fill(self, contour_arr, points):

        # Get convex hull of contour points
        try:
            # hull = ConvexHull(points,)
            # make the hull use all the points for a tighter fit
            hull = ConvexHull(points, qhull_options='QJ')   # qhull_options='QJ' is used to prevent qhull from crashing
        except Exception as e:
            print(f"convex_hull_polygon_fill(): Exception: {e}")
            return contour_arr
        
        hull_points = points[hull.vertices]
    
        # Create empty mask
        mask = np.zeros_like(contour_arr)
    
        # Draw filled polygon using hull points
        rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], mask.shape)
        mask[rr, cc] = 1
    
        return mask

    # def convex_hull_polygon_fill(self, contour_arr, points):
    #     try:
    #         # Perform Delaunay triangulation
    #         tri = Delaunay(points)
            
    #         # Create empty mask
    #         mask = np.zeros_like(contour_arr)
            
    #         # Draw filled triangles
    #         for simplex in tri.simplices:
    #             triangle = points[simplex]
    #             rr, cc = polygon(triangle[:, 0], triangle[:, 1], mask.shape)
    #             mask[rr, cc] = 1
    
    #         return mask
    
    #     except Exception as e:
    #         print(f"convex_hull_polygon_fill(): Exception: {e}")
    #         return contour_arr

    def connect_contour_points_and_fill(self, contour_arr):
        # Get contour points
        points = np.column_stack(np.where(contour_arr > 0))
        if len(points) < 3:
            return contour_arr

        filled_mask_polygon = self.convex_hull_polygon_fill(contour_arr, points)

        #quick and dirty display of the filled contour
        # plt.imshow(filled_polygon, cmap='gray')
        # plt.title("Filled contour")
        # plt.show()

        return filled_mask_polygon
           
        





