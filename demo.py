# this is a program to demonstrate the functionality of the model.
# it is a gtk based interface
# You select a zip file and it allows you to choose a dicom series from it.
# it allows you to scroll through the image in the saggital plane
# it blends in the pre contoured mask of the organ selected on the side, 
# and generated the AI contour on the fly, using the pre trained model, which is selected from the dropdown.

# first create a class for the startup window
# this allows you to select a zip file. When the zip is selected,
# it will populate a dropdown with the dicom series in the zip file
# when a series is selected, it will display the contours associated with it
# and allow you to select one.
# then it will allow you to select a model (in the same directory) from a dropdown (e.g. all the .pth files)

# when the selections have been made, the main window will open


import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf
import os
from zipfilecontents import ZipFileContents
from ImageAndMaskExtractor import ImageAndMaskExtractor
from demo_viewer import DemoViewer

class StartupWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Select Zip File")
        self.set_default_size(400, 200)
        self.set_border_width(10)
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(self.box)

        
        self.file_chooser = Gtk.FileChooserButton(title="Select Zip File")
        self.file_chooser.set_action(Gtk.FileChooserAction.OPEN)
        self.file_chooser.add_filter(self.create_filter())
        self.file_chooser.connect("file-set", self.on_file_set)
        self.box.pack_start(self.file_chooser, True, True, 0)
        
        self.series_combo = Gtk.ComboBoxText()
        self.series_combo.connect("changed", self.on_series_changed)
        self.box.pack_start(self.series_combo, True, True, 0)

        self.contour_combo = Gtk.ComboBoxText()
        self.contour_combo.connect("changed", self.on_contour_changed)  # Corrected connection
        self.box.pack_start(self.contour_combo, True, True, 0)
        
        self.button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.box.pack_start(self.button_box, False, False, 0)

        self.quit_button = Gtk.Button(label="Quit")
        self.quit_button.connect("clicked", self.on_quit_button_clicked)
        self.button_box.pack_start(self.quit_button, True, True, 0)

        self.open_button = Gtk.Button(label="Open")
        self.open_button.connect("clicked", self.on_open_button_clicked)
        self.button_box.pack_start(self.open_button, True, True, 0)
        self.show_all()
        
        self.zipfile_contents = None
        self.series_list = None
        self.series = None
        self.studyseries = None
        self.zip_content = None 
        self.contours = None
        self.contour_name = None

    def create_filter(self):
        filter = Gtk.FileFilter()
        filter.set_name("Zip files")
        filter.add_pattern("*.zip")
        return filter
        
    def on_file_set(self, widget):
        self.series_combo.remove_all()
        zip_file = self.file_chooser.get_filename()
        self.series_list = self.get_series_list(zip_file)
        for series in self.series_list:
            self.series_combo.append_text(series.get_series_id())
        self.series_combo.set_active(0)
        
    def get_series_list(self, zip_file):
        path = os.path.dirname(zip_file) + "/"
        zname = os.path.basename(zip_file)
        self.zip_content = ZipFileContents(path, zname)
        if not self.zip_content.contains_dicom_files:
            print("No dicom files found in the zip file. Skipping...")
            return
        self.series_list = self.zip_content.get_all_series()
        return self.series_list

    def on_series_changed(self, widget):
        series_name = self.series_combo.get_active_text()
        self.series = self.zip_content.series_dict[series_name]
        
        # clear the contour combo
        self.contour_combo.remove_all()

        self.studyseries = ImageAndMaskExtractor(self.series, self.zip_content)
        if not self.studyseries.has_contours:
            print(f"Series {self.series.get_series_id()} has no contours.")
            return ["No contours found"]
        if self.studyseries.series.get_num_image_files() == 0:
            print(f"Series {self.series.get_series_id()} has no image files. Skipping...")
            return ["No image files found"]

        # Get the contour names
        self.contours = self.studyseries.get_contour_names()
        for name in self.contours:
            print(f"{name}")
            self.contour_combo.append_text(name)
        self.contour_combo.set_active(0)

    def on_contour_changed(self, widget):
        self.contour_name = self.contour_combo.get_active_text()
        # print(f"Selected contour: {self.contour_name}")

    def on_quit_button_clicked(self, widget):
        Gtk.main_quit()

    def on_open_button_clicked(self, widget):
        # Handle the open button click event
        print("Open button clicked")
        # close this window and open the demo viewer
        self.hide()
        demo_viewer_window = DemoViewer(self.zip_content, self.studyseries, self.contours, self.contour_name)
        demo_viewer_window.connect("destroy", Gtk.main_quit)
        demo_viewer_window.show_all()

if __name__ == "__main__":
    win = StartupWindow()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()