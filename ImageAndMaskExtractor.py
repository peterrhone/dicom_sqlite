import pydicom as dicom
from scipy.ndimage import binary_fill_holes
from collections import defaultdict
from dicomseries import DicomSeries
from zipfilecontents import ZipFileContents
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.path import Path
import numpy as np
import os

class ImageAndMaskExtractor:
    def __init__(self, series: DicomSeries, zip_contents: ZipFileContents):
        self.zip_contents = zip_contents
        self.series = series
        self.image_and_mask = {}
        self.contour_data = self.zip_contents.get_contour_data(self.series)

        # initialize slope and intercept for Hounsfield unit conversion
        self.slope = 1.0 # default slope
        self.intercept = 0.0 # default intercept
        if self.series.get_num_image_files() > 0:
            sample_img = self.zip_contents.get_image_given_filename(list(self.series.get_image_files())[0])
            self.slope = float(sample_img.RescaleSlope)
            self.intercept = float(sample_img.RescaleIntercept)
        
        # check if contour data is empty
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
        return sagittal_aspect_ratio

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

    def fill_3d_contour_volume(self, mask_vol):
        """Apply 3D binary filling to create a closed volume from contour points."""
        try:
            # Ensure mask is binary (0 or 1)
            mask_vol = (mask_vol > 0).astype(np.uint8)
            # Apply 3D binary fill to close gaps in the volume
            filled_mask_vol = binary_fill_holes(mask_vol)
            return filled_mask_vol.astype(np.uint8)
        except Exception as e:
            print(f"fill_3d_contour_volume(): Exception: {e}")
            return mask_vol  # Return original mask if filling fails

    def get_sag_contour_pixels(self, roi_name, total_layers=88):
        # Get full volume (raw pixel values)
        ct_img_vol, mask_vol, _, _, _ = self.get_img_and_mask_volumes(roi_name)
    
        # Convert to numpy arrays
        ct_img_vol = np.array(ct_img_vol)
        mask_vol = np.array(mask_vol)
    
        # Find contour bounds in cc direction
        contour_layers = np.any(mask_vol > 0, axis=(1, 2))
        min_layer = np.where(contour_layers)[0][0] if np.any(contour_layers) else 0
        max_layer = np.where(contour_layers)[0][-1] if np.any(contour_layers) else len(mask_vol) - 1
    
        # Contour center and padding
        contour_center = (min_layer + max_layer) // 2
        half_window = total_layers // 2
    
        # Calculate layer range
        start_layer = max(0, contour_center - half_window) 
        end_layer = min(start_layer + total_layers, ct_img_vol.shape[0])
    
        if end_layer - start_layer < total_layers:
            start_layer = max(0, end_layer - total_layers)
    
        # Extract layers
        ct_img_vol = ct_img_vol[start_layer:end_layer]
        mask_vol = mask_vol[start_layer:end_layer]
    
        # Swap axes to get sagittal orientation
        ct_img_vol = np.swapaxes(ct_img_vol, 0, 2)
        mask_vol = np.swapaxes(mask_vol, 0, 2)
        ct_img_vol = np.rot90(ct_img_vol, k=1, axes=(2, 1))
        mask_vol = np.rot90(mask_vol, k=1, axes=(2, 1))
        ct_img_vol = np.flip(ct_img_vol, axis=1)
        mask_vol = np.flip(mask_vol, axis=1)
    
        # Return list of tuples (ct_img, mask_img)
        sag_img_contour_arrays = [(ct_img_vol[i], mask_vol[i]) for i in range(ct_img_vol.shape[0])]
        return sag_img_contour_arrays
    
    def get_img_and_mask_volumes(self, roi_name):
        # Create directory for mask plots
        plot_dir = "/home/peter/Data/md_data/processed/mask_plots"
        os.makedirs(plot_dir, exist_ok=True)
        pat_id = self.zip_contents.get_patient_id()
        series_id = self.series.get_series_id()
    
        # Get contour data
        roi_seq = [roi_seq for roi_seq in self.contour_data.ROIContourSequence 
                   if roi_seq.ReferencedROINumber == self.roi_names_dict[roi_name]]
        contours = [contour for contour in roi_seq[0].ContourSequence]
    
        # Get all image files sorted by position
        all_images = list(self.series.get_image_files())
        all_image_ids = []
        for img in all_images:
            try:
                img_ds = self.zip_contents.get_image_given_filename(img)
                all_image_ids.append(img_ds.SOPInstanceUID)
            except Exception as e:
                print(f"get_img_and_mask_volumes: Failed to read image {img}: {e}")
                continue
    
        # Debug: Count contours per image ID
        contour_counts = defaultdict(int)
        for contour in contours:
            try:
                img_id = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
                contour_counts[img_id] += 1
            except Exception as e:
                print(f"get_img_and_mask_volumes: Invalid contour data: {e}")
                continue
        for img_id, count in contour_counts.items():
            print(f"get_img_and_mask_volumes: ROI={roi_name}, Image ID={img_id}, Contours={count}")
    
        # Create volumes with all slices
        ct_img_vol = []
        mask_vol = []
        plot_count = 0
    
        # Process contours to get mask info, accumulating multiple contours per slice
        contour_data = defaultdict(lambda: (None, None))
        for contour in contours:
            try:
                img_arr, temp_mask, img_id = self.coord2pixels(contour)
                print(f"coord2pixels: ROI={roi_name}, Image ID={img_id}, Mask Pixels={np.sum(temp_mask)}")
                if contour_data[img_id][0] is None:
                    contour_data[img_id] = (img_arr, temp_mask.astype(np.uint8))
                else:
                    _, slice_mask = contour_data[img_id]
                    slice_mask = np.maximum(slice_mask, temp_mask)  # Combine disconnected regions
                    contour_data[img_id] = (img_arr, slice_mask)
            except ValueError as e:
                print(f"get_img_and_mask_volumes: Skipping contour due to error: {e}")
                continue
    
        # Fill volumes including empty slices and plot non-zero masks
        for idx, (img_file, img_id) in enumerate(zip(all_images, all_image_ids)):
            try:
                img = self.zip_contents.get_image_given_filename(img_file)
                img_arr = img.pixel_array  # Store raw pixel values
                if img_id in contour_data:
                    ct_img_vol.append(img_arr)
                    mask = contour_data[img_id][1]
                    mask_vol.append(mask)
                    # Plot non-zero masks
                    # if np.sum(mask) > 0:
                    #     plot_count += 1
                    #     plt.figure(figsize=(10, 5))
                    #     plt.subplot(1, 2, 1)
                    #     plt.imshow(img_arr, cmap='gray')
                    #     plt.title(f"Patient {pat_id}, Series {series_id[:10]}..., Slice {idx}")
                    #     plt.subplot(1, 2, 2)
                    #     plt.imshow(mask, cmap='binary')
                    #     plt.title(f"Mask (Pixels: {np.sum(mask)})")
                    #     # Unique filename with patient ID, series ID, and slice index
                    #     plot_filename = os.path.join(plot_dir, f"mask_{pat_id}_{series_id[:10]}_slice_{idx}.png")
                    #     plt.savefig(plot_filename)
                    #     plt.close()
                    #     print(f"get_img_and_mask_volumes: Saved plot {plot_filename}, Plot Count={plot_count}")
                else:
                    ct_img_vol.append(img_arr)
                    mask_vol.append(np.zeros(img_arr.shape, dtype=np.uint8))
            except Exception as e:
                print(f"get_img_and_mask_volumes: Failed to process image {img_file}: {e}")
                continue
    
        # Convert to numpy array and fill the 3D mask volume
        mask_vol = np.array(mask_vol)
        if mask_vol.size > 0:
            mask_vol = self.fill_3d_contour_volume(mask_vol)
    
        # Calculate volume in voxels and milliliters
        total_voxels = np.sum(mask_vol) if mask_vol.size > 0 else 0
        voxel_spacing = self.get_voxel_spacing()
        total_volume_ml = 0.0
        if voxel_spacing:
            voxel_volume_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
            total_volume_ml = (total_voxels * voxel_volume_mm3) / 1000.0
            print(f"get_img_and_mask_volumes: ROI={roi_name}, total_voxels={total_voxels}, "
                  f"voxel_volume_mm3={voxel_volume_mm3}, total_volume_ml={total_volume_ml}, "
                  f"Total Plots Saved={plot_count}")

        return ct_img_vol, mask_vol, all_image_ids, total_voxels, total_volume_ml

    def get_contour_pixels(self, roi_name):
        roi_seq = [roi_seq for roi_seq in self.contour_data.ROIContourSequence 
                   if roi_seq.ReferencedROINumber == self.roi_names_dict[roi_name]]
        contours = [contour for contour in roi_seq[0].ContourSequence]
        img_contour_arrays = [self.coord2pixels(contour) for contour in contours]
        return img_contour_arrays

    def coord2pixels(self, contour):
        self.ctr += 1
        contour_coord = contour.ContourData
    
        # Get the corresponding image
        img_ID = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
        img = self.zip_contents.get_image_given_img_ID(self.series, img_ID)
        if img is None:
            print(f"coord2pixels(): Image with ID {img_ID} not found.")
            raise ValueError(f"coord2pixels(): Image with ID {img_ID} not found.")
    
        img_arr = img.pixel_array  # Use raw pixel values
    
        x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])
        origin_x, origin_y, _ = img.ImagePositionPatient
    
        # Convert physical coordinates to pixel coordinates using raw contour points
        pixel_coords = [(float(y - origin_y) / y_spacing, float(x - origin_x) / x_spacing) 
                        for x, y, _ in zip(contour_coord[::3], contour_coord[1::3], contour_coord[2::3])]
        if len(pixel_coords) < 3:
            print(f"coord2pixels(): Insufficient points for polygon fill ({len(pixel_coords)} points)")
            contour_mask = np.zeros(img_arr.shape, dtype=np.uint8)
            return img_arr, contour_mask, img_ID
    
        # Ensure closed contour (DICOM contours should be closed, but verify)
        if pixel_coords[0] != pixel_coords[-1]:
            pixel_coords.append(pixel_coords[0])
            print(f"coord2pixels(): Appended first point to close contour")
    
        # Validate contour points
        valid_coords = [(i, j) for i, j in pixel_coords if 0 <= i < img_arr.shape[0] and 0 <= j < img_arr.shape[1]]
        if len(valid_coords) < len(pixel_coords):
            print(f"coord2pixels(): Found {len(pixel_coords) - len(valid_coords)} contour coords outside image bounds")
        if len(valid_coords) < 3:
            print(f"coord2pixels(): Insufficient valid points for polygon fill ({len(valid_coords)} points)")
            contour_mask = np.zeros(img_arr.shape, dtype=np.uint8)
            return img_arr, contour_mask, img_ID
    
        # Create Path for point-in-polygon test
        try:
            path = Path(valid_coords)
            # Create grid of pixel coordinates (centers of pixels)
            rows, cols = np.indices(img_arr.shape)
            points = np.vstack((rows.ravel() + 0.5, cols.ravel() + 0.5)).T
            # Test which pixels are inside the polygon
            mask = path.contains_points(points, radius=0.5).reshape(img_arr.shape)
            contour_mask = mask.astype(np.uint8)
            # Fill holes within the contour
            contour_mask = binary_fill_holes(contour_mask).astype(np.uint8)
            # Debug: Check path validity
            print(f"coord2pixels(): Path vertices={len(valid_coords)}, Closed={path.vertices[0] == path.vertices[-1]}")
        except Exception as e:
            print(f"coord2pixels(): Exception in point-in-polygon test: {e}")
            contour_mask = np.zeros(img_arr.shape, dtype=np.uint8)
    
        return img_arr, contour_mask, img_ID
