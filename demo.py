import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, Gio
import os
try:
    from zipfilecontents import ZipFileContents
except ModuleNotFoundError:
    ZipFileContents = None
    # try:
        # with open('/tmp/ai_view_missing_pydicom.txt', 'w') as f:
            # f.write('pydicom missing. Check python environment.')
    # except Exception:
        # pass
from ImageAndMaskExtractor import ImageAndMaskExtractor
from demo_viewer import DemoViewer

class StartupWindow(Gtk.Window):
    def __init__(self, application=None):
        # with open('/tmp/ai_view_started.txt', 'w') as f:
            # f.write('started')
        # pass the Gtk.Application instance into the Gtk.Window so the window
        # belongs to the application (helps GNOME group windows and use the same icon)
        try:
            Gtk.Window.__init__(self, title="Select Zip File", application=application)
        except TypeError:
            # older Gtk bindings may not accept the application kw; fall back
            Gtk.Window.__init__(self, title="Select Zip File")
        # set WM_CLASS on the Gtk.Window (deprecated API but works for matching
        # the running window to the .desktop file's StartupWMClass)
        try:
            self.set_wmclass("ai-view", "ai-view")
        except Exception:
            # if unavailable, ignore; some backends don't expose it
            pass
        # set the icon name to an icon in the icon theme (ai-view-icon)
        try:
            self.set_icon_name("ai-view-icon")
        except Exception:
            pass
        # keep realize handler only for diagnostics (don't call Gdk.set_wmclass on the Gdk.Window)
        self.connect('realize', self.on_realize)
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
        
        self.button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.box.pack_start(self.button_box, False, False, 0)

        self.quit_button = Gtk.Button(label="Quit")
        self.quit_button.connect("clicked", self.on_quit_button_clicked)
        self.button_box.pack_start(self.quit_button, True, True, 0)

        self.open_button = Gtk.Button(label="Open")
        self.open_button.connect("clicked", self.on_open_button_clicked)
        self.button_box.pack_start(self.open_button, True, True, 0)
        
        self.zipfile_contents = None
        self.series_list = None
        self.series = None
        self.studyseries = None
        self.zip_content = None 
        
        self.show_all()
        # with open('/tmp/ai_view_init_done.txt', 'w') as f:
            # f.write('init done')

    def on_realize(self, widget):
        # don't call set_wmclass on the Gdk.Window object (AttributeError on some builds)
        window = self.get_window()
        if window:
            try:
                wm = window.get_wmclass()
            except Exception:
                wm = None
            # with open('/tmp/ai_view_wmclass.txt', 'w') as f:
                # f.write(str(wm))
        # write application id and icon name for diagnostics
        try:
            app = self.get_application()
            app_id = app.get_application_id() if app else None
        except Exception:
            app_id = None
        try:
            icon_name = self.get_icon_name()
        except Exception:
            icon_name = None
        # try:
            # with open('/tmp/ai_view_realize_diag.txt', 'w') as f:
                # f.write(f"wmclass={wm}\napp_id={app_id}\nicon_name={icon_name}\n")
        # except Exception:
            # pass

    def create_filter(self):
        filter = Gtk.FileFilter()
        filter.set_name("Zip files")
        filter.add_pattern("*.zip")
        return filter
        
    def on_file_set(self, widget):
        self.series_combo.remove_all()
        zip_file = self.file_chooser.get_filename()
        if not zip_file or not zip_file.endswith('.zip'):
            print("Please select a valid ZIP file")
            return
        self.series_list = self.get_series_list(zip_file)
        if not self.series_list:
            print("No valid series found in ZIP file")
            return
        for series in self.series_list:
            self.series_combo.append_text(series.get_series_id())
        self.series_combo.set_active(0)
    
    def get_series_list(self, zip_file):
        path = os.path.dirname(zip_file) + "/"
        zname = os.path.basename(zip_file)
        self.zip_content = ZipFileContents(path, zname)
        if not self.zip_content.contains_dicom_files:
            print("No DICOM files found in the ZIP file. Skipping...")
            return []
        self.series_list = self.zip_content.get_all_series()
        return self.series_list
    
    def on_series_changed(self, widget):
        series_name = self.series_combo.get_active_text()
        if not series_name:
            print("No series selected")
            return
        self.series = self.zip_content.series_dict[series_name]
        self.studyseries = ImageAndMaskExtractor(self.series, self.zip_content)
        if not self.studyseries.has_contours:
            print(f"Series {self.series.get_series_id()} has no contours.")
            return
        if self.studyseries.series.get_num_image_files() == 0:
            print(f"Series {self.series.get_series_id()} has no image files. Skipping...")
            return
    
    def on_quit_button_clicked(self, widget):
        # explicit safe quit path
        try:
            self.destroy()
        except Exception:
            pass
        _safe_quit()
    
    def on_open_button_clicked(self, widget):
        if not self.studyseries:
            print("Please select a valid series")
            return
        print("Open button clicked")
        self.hide()
        contours = self.studyseries.get_contour_names()
        if not contours:
            print("No contours available for this series")
            return
        demo_viewer_window = DemoViewer(self.zip_content, self.studyseries, contours, self)
        demo_viewer_window.connect("destroy", self.on_viewer_destroy)
        demo_viewer_window.show_all()
    
    def on_viewer_destroy(self, widget):
        self.show_all()


def _safe_quit():
    # prefer quitting the Gtk.Application if present (works on Wayland/GNOME),
    # otherwise fall back to Gtk.main_quit for legacy usage
    app = None
    try:
        app = Gtk.Application.get_default()
    except Exception:
        app = None
    if app is not None:
        try:
            app.quit()
            return
        except Exception:
            pass
    try:
        Gtk.main_quit()
    except Exception:
        # nothing we can do
        pass
    # debug marker for verification
    # try:
        # with open('/tmp/ai_view_safe_quit_called.txt', 'w') as f:
            # f.write('safe quit called')
    # except Exception:
        # pass


if __name__ == '__main__':
    # Use a Gtk.Application so GNOME / Wayland groups windows and applies the
    # .desktop icon consistently.
    try:
        # with open('/tmp/ai_view_main_started.txt', 'w') as f:
            # f.write('main started')
        app = Gtk.Application(application_id='org.peter.ai_view')
        # set the application-level icon name (must match the icon in the theme)
        try:
            app.set_property('default-icon', 'ai-view-icon')
        except Exception:
            # older bindings may not expose set_property for default-icon
            pass

        # create and show the startup window as belonging to the application
        def on_activate(application):
            win = StartupWindow(application=application)
            win.connect('destroy', lambda w: _safe_quit())
            win.show_all()

        app.connect('activate', on_activate)
        exit_status = app.run(None)
        # explicit exit
        raise SystemExit(exit_status)
    except Exception as e:
        print(f"Exception in main: {e}")
        # with open('/tmp/ai_view_error.txt', 'w') as f:
            # f.write(str(e))