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

# class definitions are in separate files
from datadirectorychooser import DataDirectoryChooser
# from dicomseries import DicomSeries
from sqlitedata import SqliteData
from sqlite_viewer import ImageViewer
from zipfilecontents import ZipFileContents
from ImageAndMaskExtractor import ImageAndMaskExtractor
from imagescroller import ImageScroller
import matplotlib.pyplot as plt
        
global should_continue
global end_mark
should_continue = True

# ################################################################################
# # start of the main program, to test the functions
# ################################################################################

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

# get the path to the data directory
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
    zipfiles=chooser_dialog.get_zip_filenames()
    wanted_ROI = None
    if mode == "save":
        wanted_ROI = chooser_dialog.get_selected_structures()[0].lower()
    chooser_dialog.destroy()

    if mode == "save" and len(zipfiles) == 0:
        show_warning_dialog("No zip files found in the selected directory. Please select a valid directory.")
    else:
        break

print(f"Mode = {mode}")

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
# Create a mark at the end of the buffer to keep track of the end position
end_mark = textbuffer.create_mark("end", textbuffer.get_end_iter(), False)

def check_mark():
    global end_mark
    if end_mark.get_deleted():
        print("Mark deleted")
        # global end_mark
        end_mark = textbuffer.create_mark("end", textbuffer.get_end_iter(), False)
    return end_mark
    

scrolled_window.add(textview)

def on_close_button_clicked(widget):
    global should_continue, processing_thread
    should_continue = False
    Gtk.main_quit()
    if 'processing_thread' in globals() and processing_thread.is_alive():
        processing_thread.join(timeout=5)
    sys.exit(0)  # Ensure the program terminates instantly

close_button = Gtk.Button(label="Close")
close_button.connect("clicked", on_close_button_clicked)
vbox.pack_start(close_button, False, False, 0)

progress_window.show_all()

progress_bar.set_fraction(0)
progress_bar.set_text("0%")
progress_window.show_all()

def process_zip_file(zip_file_num):
    db = SqliteData(save_location)  # Create a new database connection for each thread
    progress_bar.set_fraction((zip_file_num + 1) / len(zipfiles))
    progress_bar.set_text(f"{int((zip_file_num + 1) / len(zipfiles) * 100)}%")
    progress_label.set_text(f"Processing zip file {zip_file_num + 1} of {len(zipfiles)}. {(zip_file_num + 1)/len(zipfiles)*100:.0f}% complete")
    progress_window.show_all()

    zip_content = ZipFileContents(path, zipfiles[zip_file_num])
    if not zip_content.contains_dicom_files:
        GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), "No dicom files found in the zip file. Skipping...\n"))
        GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
        return
    pat_id = zip_content.get_patient_id()
    GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), f"Processing patient {pat_id}...\n"))
    GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))

    for series in zip_content.get_all_series():
        if not should_continue:
            db.close()
            return
        studyseries = ImageAndMaskExtractor(series, zip_content)
        # print(f"Voxel spacing: {studyseries.get_voxel_spacing()}")
        if not studyseries.has_contours:
            GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), f"Series {series.get_series_id()} has no contours. Skipping...\n"))
            GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
            continue
        if studyseries.series.get_num_image_files() == 0:
            GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), f"Series {series.get_series_id()} has no image files. Skipping...\n"))
            GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
            continue
        if studyseries.series.get_num_image_files() > 88: # this would be the planing CT - skip
            # GLib.idle_add(textbuffer.insert, textbuffer.get_iter_at_mark(end_mark), f"Series {series.get_series_id()} has more than 88 image files and is thus not a CBCT. Skipping...\n")
            continue
        series_id = series.get_series_id()
        aspect_ratio = studyseries.get_sag_aspect_ratio()

        # Determine the ROI names to select
        roi_names = []
        match wanted_ROI:
            case "bladder":
                roi_names = [name for name in studyseries.roi_names_dict.keys() if "bladder" in name.lower()]
            case "rectum":
                roi_names =  [name for name in studyseries.roi_names_dict.keys() if "rectum" in name.lower()]
            case "prostate":
                roi_names = [name for name in studyseries.roi_names_dict.keys() if "prostate_sb" in name.lower()]
                roi_names.extend([name for name in studyseries.roi_names_dict.keys() if "prostata_sb" in name.lower()])
                roi_names.extend([name for name in studyseries.roi_names_dict.keys() if "prostata_sbb" in name.lower()])
                roi_names.extend([name for name in studyseries.roi_names_dict.keys() if "prostatasbb" in name.lower()])
                roi_names.extend([name for name in studyseries.roi_names_dict.keys() if "58Gy" in name])
                roi_names.extend([name for name in studyseries.roi_names_dict.keys() if "PTV_Prostate-SBB" in name])
                roi_names.extend([name for name in studyseries.roi_names_dict.keys() if "PTV1" in name]) #since these are prostate patients, PTV1 = the 58Gy PTV which includes the samenblasen
                roi_names.extend([name for name in studyseries.roi_names_dict.keys() if "PTV mit SBB 58 Gy" in name]) 
                roi_names.extend([name for name in studyseries.roi_names_dict.keys() if "PTV Prostata SBB" in name]) 
                roi_names.extend([name for name in studyseries.roi_names_dict.keys() if "PTV-Prostata-CT-2-3-mitSBB" in name]) 
            case _:
                roi_names = [wanted_ROI]
        roi_names = list(set(roi_names)) # deduplicate
        if len(roi_names)>1:
            GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), f"Multiple ROIs found for '{wanted_ROI}' ({len(roi_names)}): {roi_names}\n"))
            GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
        # Check if any of the ROI names exist in the series
        valid_roi_names = [roi for roi in roi_names if roi in studyseries.roi_names_dict]
        if not valid_roi_names:
            GLib.idle_add(lambda: textbuffer.insert(textbuffer.get_iter_at_mark(check_mark()), f"Series {series.get_series_id()} does not contain contour '{wanted_ROI}'. Not in {studyseries.roi_names_dict.keys()}. Skipping...\n"))
            GLib.idle_add(lambda: textbuffer.move_mark(end_mark, textbuffer.get_end_iter()))
            continue

        try:
            contour_arrays = []
            for roi_name in valid_roi_names:
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
                image, contour, img_id = contour_arrays[i]
                mask = Image.new("RGBA", (image.shape[1], image.shape[0]), (0, 0, 0, 0))
                for x in range(mask.width - 1):
                    for y in range(mask.height - 1):
                        if contour[y, x] > 0:
                            mask.putpixel((x, y), (255, 0, 0, 90))
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
                image, contour, img_id = contour_arrays[i]
                image_arr.append(image)
                mask_arr.append(contour)
            mid_idx = len(image_arr) // 2
            match wanted_ROI:
                case "bladder":
                    slice_indices = [mid_idx - 3, mid_idx, mid_idx + 3]
                case "rectum":
                    slice_indices = [mid_idx - 1, mid_idx, mid_idx + 1]
                case "prostate":
                    slice_indices = [mid_idx - 1, mid_idx, mid_idx + 1]
                case _:
                    slice_indices = [mid_idx - 1, mid_idx, mid_idx + 1]
            for slice_idx in slice_indices:
                if 0 <= slice_idx < len(image_arr):
                    # if the mask at the slice index is empty, skip it
                    if np.sum(mask_arr[slice_idx]) == 0:
                        continue
                    db.insert_db(pat_id, series_id, slice_idx, image_arr[slice_idx], mask_arr[slice_idx], wanted_ROI)

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
