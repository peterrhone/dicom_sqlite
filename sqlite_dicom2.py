import zipfile
from PIL import Image
import collections
import numpy as np
import math
import warnings
import tkinter as tk
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib
import sys
import threading
import os

from datadirectorychooser import DataDirectoryChooser
from sqlitedata import SqliteData
from sqlite_viewer import ImageViewer
from zipfilecontents import ZipFileContents
from ImageAndMaskExtractor import ImageAndMaskExtractor
from imagescroller import ImageScroller
import matplotlib.pyplot as plt

global should_continue
global end_mark
should_continue = True

def show_warning_dialog(message):
    dialog = Gtk.MessageDialog(
        transient_for=None,
        flags=0,
        message_type=Gtk.MessageType.WARNING,
        buttons=Gtk.ButtonsType.OK,
        text="Warning",
    )
    dialog.format_secondary_text(message)
    dialog.run()
    dialog.destroy()

# Get the path to the data directory
while True:
    chooser_dialog = DataDirectoryChooser()
    chooser_dialog.show_all()
    result = chooser_dialog.run()
    print(f"Result = {result}")
    print(f"Response type Save = {Gtk.ResponseType.OK}")
    print(f"Response type Cancel = {Gtk.ResponseType.CANCEL}")
    if result == Gtk.ResponseType.CANCEL or result == Gtk.ResponseType.DELETE_EVENT:
        print("No directory selected. Exiting...")
        sys.exit()

    path = chooser_dialog.get_path()
    mode = chooser_dialog.get_mode()
    save_location = chooser_dialog.get_save_path()
    sqlite_file = chooser_dialog.get_sqlite_fullpath()
    zipfiles = chooser_dialog.get_zip_filenames()
    wanted_ROI = None
    if mode == "save":
        wanted_ROI = chooser_dialog.get_selected_structures()[0].lower()
        # Ensure save_location is a valid .sqlite file path
        if not save_location.endswith('.sqlite'):
            save_location = os.path.join(os.path.dirname(save_location), f"images_{wanted_ROI}.sqlite")
        # Fix double slashes and normalize path
        save_location = os.path.normpath(save_location)
        # Check if directory is writable
        save_dir = os.path.dirname(save_location)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not os.access(save_dir, os.W_OK):
            show_warning_dialog(f"Cannot write to directory {save_dir}. Please check permissions.")
            continue
    chooser_dialog.destroy()

    if mode == "save" and len(zipfiles) == 0:
        show_warning_dialog("No zip files found in the selected directory. Please select a valid directory.")
    else:
        break

print(f"Mode = {mode}")
print(f"Save location = {save_location}")

match mode:
    case "save":
        print(f"Saving images to database at {save_location}")
        print(f"Working with {len(zipfiles)} zip files")
    case "view":
        print("Viewing images")
        sql_viewer = ImageViewer(sqlite_file)
        sql_viewer.run()
        sys.exit()
    case _:
        print("Invalid mode selected. Exiting...")
        sys.exit()

if len(zipfiles) == 0:
    print("No zip files found in the directory. Exiting...")
    exit()

# Create a progress bar window
progress_window = Gtk.Window(title="Processing Zip Files")
progress_window.set_default_size(400, 200)
vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
progress_window.add(vbox)

progress_label = Gtk.Label(label="Processing zip files...")
vbox.pack_start(progress_label, False, False, 0)

progress_bar = Gtk.ProgressBar()
vbox.pack_start(progress_bar, False, False, 0)

scrolled_window = Gtk.ScrolledWindow()
scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
vbox.pack_start(scrolled_window, True, True, 0)

textview = Gtk.TextView()
textview.set_editable(False)
textbuffer = textview.get_buffer()
end_mark = textbuffer.create_mark("end", textbuffer.get_end_iter(), False)

def check_mark():
    global end_mark
    if end_mark.get_deleted():
        print("Mark deleted")
        end_mark = textbuffer.create_mark("end", textbuffer.get_end_iter(), False)
    return end_mark
    
scrolled_window.add(textview)

def on_close_button_clicked(widget):
    global should_continue, processing_thread
    should_continue = False
    Gtk.main_quit()
    if 'processing_thread' in globals() and processing_thread.is_alive():
        processing_thread.join(timeout=5)
    sys.exit(0)

close_button = Gtk.Button(label="Close")
close_button.connect("clicked", on_close_button_clicked)
vbox.pack_start(close_button, False, False, 0)

progress_window.show_all()

progress_bar.set_fraction(0)
progress_bar.set_text("0%")
progress_window.show_all()

def process_zip_file(zip_file_num):
    db = SqliteData(save_location)
    progress_bar.set_fraction((zip_file_num + 1) / len(zipfiles))
    progress_bar.set_text(f"{int((zip_file_num + 1) / len(zipfiles) * 100)}%")
    progress_label.set_text(f"Processing zip file {zip_file_num + 1} of {len(zipfiles)}. {(zip_file_num + 1)/len(zipfiles)*100:.0f}% complete")
    progress_window.show_all()

    try:
        zip_content = ZipFileContents(path, zipfiles[zip_file_num])
    except FileNotFoundError as e:
        GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), 
            f"Failed to open zip file {zipfiles[zip_file_num]}: {e}\n"))
        GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
        return
    if not zip_content.contains_dicom_files:
        GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), 
            f"No dicom files found in zip file {zipfiles[zip_file_num]}. Skipping...\n"))
        GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
        return
    pat_id = zip_content.get_patient_id()
    GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), 
        f"Processing patient {pat_id}...\n"))
    GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))

    for series in zip_content.get_all_series():
        if not should_continue:
            db.close()
            return
        studyseries = ImageAndMaskExtractor(series, zip_content)
        if not studyseries.has_contours:
            GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), 
                f"Series {series.get_series_id()} has no contours. Skipping...\n"))
            GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
            continue
        if studyseries.series.get_num_image_files() == 0:
            GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), 
                f"Series {series.get_series_id()} has no image files. Skipping...\n"))
            GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
            continue
        if studyseries.series.get_num_image_files() > 88:
            continue
        series_id = series.get_series_id()
        aspect_ratio = studyseries.get_sag_aspect_ratio()

        # Get voxel spacing for volume calculation
        voxel_spacing = studyseries.get_voxel_spacing()
        if voxel_spacing is None:
            GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), 
                f"Series {series.get_series_id()} has no voxel spacing. Skipping...\n"))
            GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
            continue
        x_spacing, y_spacing, layer_spacing = voxel_spacing

        roi_names = []
        match wanted_ROI:
            case "bladder":
                roi_names = [name for name in studyseries.roi_names_dict.keys() if "bladder" in name.lower()]
            case "rectum":
                roi_names = [name for name in studyseries.roi_names_dict.keys() if "rectum" in name.lower()]
            case "prostate":
                roi_names = [name for name in studyseries.roi_names_dict.keys() 
                            if ("prostate_sb" in name.lower() or "prostata_sb" in name.lower() or 
                                "prostata_sbb" in name.lower() or "prostatasbb" in name.lower()) 
                            and "ptv" not in name.lower() and "58gy" not in name.lower() and '68gy' not in name.lower()]
            case _:
                roi_names = [wanted_ROI]
        roi_names = list(set(roi_names))
        if len(roi_names) > 1:
            GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), 
                f"Multiple ROIs found for '{wanted_ROI}' ({len(roi_names)}): {roi_names}\n"))
            GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
        valid_roi_names = [roi for roi in roi_names if roi in studyseries.roi_names_dict]
        if not valid_roi_names:
            GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), 
                f"Series {series.get_series_id()} does not contain contour '{wanted_ROI}'. Not in {studyseries.roi_names_dict.keys()}. Skipping...\n"))
            GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
            continue

        try:
            contour_arrays = []
            for roi_name in valid_roi_names:
                # Get volume and voxel count from get_img_and_mask_volumes
                ct_img_vol, mask_vol, _, total_voxels, total_volume_ml = studyseries.get_img_and_mask_volumes(roi_name)
                contour_arrays.extend(studyseries.get_sag_contour_pixels(roi_name))
        except Exception as e:
            error_message = f"Error extracting contour for '{wanted_ROI}' in series {series.get_series_id()}: {e} \n"
            GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), error_message))
            GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
            continue

        if mode == "view":
            image_arr = []
            mask_arr = []
            for i in range(len(contour_arrays)):
                image, contour = contour_arrays[i]
                mask = np.zeros_like(contour)
                mask[contour > 0] = 1
                image_arr.append(image)
                mask_arr.append(mask)

            totalmax = 0
            images = []
            for img in image_arr:
                max_val = np.max(img)
                if max_val > totalmax:
                    totalmax = max_val
            for img in image_arr:
                img = img / totalmax
                img = np.round(img * 255)
                PILImage = Image.fromarray(img.astype("uint8")).convert("RGBA")
                images.append(PILImage)

            tkroot = tk.Tk()
            tkroot.title(f"Zip filename: {zipfiles[zip_file_num]} - series id {series.get_series_id()}")
            scroller = ImageScroller(images, mask_arr, tkroot, aspect_ratio)
            scroller.start()

        elif mode == "save":
            image_arr = []
            mask_arr = []
            for i in range(len(contour_arrays)):
                image, contour = contour_arrays[i]
                image_arr.append(image)
                mask_arr.append(contour)
            mid_idx = len(image_arr) // 2
            match wanted_ROI:
                case "bladder":
                    slice_indices = [mid_idx - 5, mid_idx - 3, mid_idx, mid_idx + 3, mid_idx + 5]
                case "rectum":
                    slice_indices = [mid_idx -2, mid_idx - 1, mid_idx, mid_idx + 1, mid_idx + 2]
                case "prostate":
                    slice_indices = [mid_idx - 2, mid_idx - 1, mid_idx, mid_idx + 1, mid_idx + 2]
                case _:
                    slice_indices = [mid_idx - 1, mid_idx, mid_idx + 1]
            for slice_idx in slice_indices:
                if 0 <= slice_idx < len(image_arr):
                    if np.sum(mask_arr[slice_idx]) == 0:
                        continue
                    for roi_name in valid_roi_names:
                        db.insert_db(pat_id, series_id, slice_idx, image_arr[slice_idx], mask_arr[slice_idx], 
                                     wanted_ROI, total_volume_ml, total_voxels, x_spacing, y_spacing, layer_spacing)
    db.close()

def process_all_zip_files():
    for zip_file_num in range(len(zipfiles)):
        if not should_continue:
            return
        process_zip_file(zip_file_num)
    if mode == "save":
        GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(end_mark), "Saved all images to database\n"))
        GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
    GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(end_mark), "PROCESSING COMPLETE. CLICK 'Close' TO EXIT.\n"))
    GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
    progress_label.set_text("Processing complete. Click 'Close' to exit.")

def start_processing():
    global processing_thread
    processing_thread = threading.Thread(target=process_all_zip_files)
    processing_thread.start()

GLib.idle_add(start_processing)

Gtk.main()
