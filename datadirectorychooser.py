import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gio
from os import walk, path

class DataDirectoryChooser(Gtk.Dialog):

    def __init__(self):
        Gtk.Dialog.__init__(self)
        self.set_title("Choose Data Directory or SQLite File")
        self.add_button(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL)
        self.add_button(Gtk.STOCK_OPEN, Gtk.ResponseType.OK)

        # Initialize file chooser with OPEN action to allow both directory and file selection
        self.chooser = Gtk.FileChooserWidget()
        self.chooser.set_select_multiple(False)  # Only one selection at a time

        # Add file type filters
        self.filter_zip = Gtk.FileFilter()
        self.filter_zip.set_name("Zip files")
        self.filter_zip.add_mime_type("application/zip")
        self.chooser.add_filter(self.filter_zip)
        self.filter_sqlite = Gtk.FileFilter()
        self.filter_sqlite.set_name("SQLite files")
        self.filter_sqlite.add_pattern("*.sqlite")
        self.filter_sqlite.add_pattern("*.db")
        self.chooser.add_filter(self.filter_sqlite)
        self.filter_all = Gtk.FileFilter()
        self.filter_all.set_name("All files")
        self.filter_all.add_pattern("*")
        self.chooser.add_filter(self.filter_all)
        # Start with all files filter
        self.chooser.set_filter(self.filter_all)

        # Connect signals for handling selection
        self.chooser.connect("file-activated", self.on_item_activated)
        self.chooser.connect("selection-changed", self.on_selection_changed)

        box = self.get_content_area()
        box.add(self.chooser)
        box.set_spacing(6)

        # Save filename widgets
        self.save_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.pack_start(self.save_box, False, True, 0)

        save_label = Gtk.Label(label="Save filename")
        self.save_box.pack_start(save_label, False, True, 0)

        self.save_entry = Gtk.Entry()
        self.sqlite_file = "images.sqlite"  # Default value
        self.save_entry.set_text(self.sqlite_file)
        self.save_entry.set_editable(True)
        self.save_entry.connect("key-release-event", self.on_save_entry_changed)
        self.save_box.pack_start(self.save_entry, False, True, 0)

        # Radio buttons
        radio_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.pack_start(radio_box, False, True, 0)

        self.view_radio = Gtk.RadioButton.new_with_label_from_widget(None, "View")
        self.view_radio.set_active(False)
        self.view_radio.connect("toggled", self.on_radio_button_toggled)
        radio_box.pack_start(self.view_radio, False, True, 0)

        self.save_radio = Gtk.RadioButton.new_with_label_from_widget(self.view_radio, "Save")
        self.save_radio.set_active(True)
        self.save_radio.connect("toggled", self.on_radio_button_toggled)
        radio_box.pack_start(self.save_radio, False, True, 0)

        # Structure chooser
        structure_label = Gtk.Label(label="Select Structures to save:")
        self.save_box.pack_start(structure_label, False, True, 0)

        self.structure_listbox = Gtk.ListBox()
        self.structure_listbox.set_selection_mode(Gtk.SelectionMode.SINGLE)
        structures = ["Bladder", "Rectum", "Prostate"]
        for structure in structures:
            row = Gtk.ListBoxRow()
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
            row.add(hbox)
            label = Gtk.Label(label=structure, xalign=0)
            hbox.pack_start(label, True, True, 0)
            self.structure_listbox.add(row)
        self.structure_listbox.connect("row-selected", self.on_structure_selected)
        self.save_box.pack_start(self.structure_listbox, False, True, 0)

        # Initially hide save options
        self.save_box.hide()

        self.show_all()

        # Ensure save_box is hidden if view_radio is active
        if self.view_radio.get_active():
            self.save_box.hide()

    def on_radio_button_toggled(self, widget):
        if self.save_radio.get_active():
            self.save_box.show()
            self.chooser.set_filter(self.filter_all)
            self.chooser.set_action(Gtk.FileChooserAction.SELECT_FOLDER)  # Force directory selection in save mode
        else:
            self.save_box.hide()
            self.chooser.set_action(Gtk.FileChooserAction.OPEN)  # Allow file selection in view mode

    def on_item_activated(self, widget):
        file = widget.get_file()
        if file:
            file_type = file.query_file_type(Gio.FileQueryInfoFlags.NONE, None)
            if file_type == Gio.FileType.REGULAR and self.view_radio.get_active():
                if file.get_basename().endswith(('.sqlite', '.db')):
                    self.sqlite_file = file.get_path()
                    self.response(Gtk.ResponseType.OK)

    def on_selection_changed(self, widget):
        file = widget.get_file()
        if file:
            file_type = file.query_file_type(Gio.FileQueryInfoFlags.NONE, None)
            if file_type == Gio.FileType.DIRECTORY:
                self.path = file.get_path() + "/"
            elif file_type == Gio.FileType.REGULAR and self.view_radio.get_active():
                if file.get_basename().endswith(('.sqlite', '.db')):
                    self.sqlite_file = file.get_path()

    def on_save_entry_changed(self, widget, event):
        self.sqlite_file = widget.get_text()
        
    def get_mode(self):
        return "save" if self.save_radio.get_active() else "view"

    def get_selected_structures(self):
        selected_structures = []
        selected_rows = self.structure_listbox.get_selected_rows()
        for row in selected_rows:
            selected_structures.append(row.get_child().get_children()[0].get_text())
        return selected_structures

    def get_path(self):
        if self.chooser.get_current_folder() is None:
            return None
        self.path = path.normpath(self.chooser.get_current_folder() + '/')
        return self.path

    def get_save_path(self):
        if self.chooser.get_current_folder() is None:
            return None
        self.path = path.normpath(self.chooser.get_current_folder() + '/')
        # Ensure sqlite_file ends with .sqlite
        if not self.sqlite_file.endswith('.sqlite'):
            self.sqlite_file = self.sqlite_file + '.sqlite'
        return path.join(self.path, self.sqlite_file)

    def get_sqlite_fullpath(self):
        return self.sqlite_file if hasattr(self, 'sqlite_file') else None

    def get_zip_filenames(self):
        zfiles = []
        if self.chooser.get_current_folder() is None:
            return zfiles
        self.path = path.normpath(self.chooser.get_current_folder() + '/')
        for (dirpath, dirnames, filenames) in walk(self.path):
            for filename in filenames:
                if filename.endswith('.zip'):
                    zfiles.append(filename)
            break
        zfiles.sort()
        return zfiles

    def get_selected_files(self):
        return [self.sqlite_file] if hasattr(self, 'sqlite_file') and self.view_radio.get_active() else []

    def on_structure_selected(self, listbox, row):
        if row:
            structure_name = row.get_child().get_children()[0].get_text().lower()
            self.sqlite_file = f"images_{structure_name}.sqlite"
            self.save_entry.set_text(self.sqlite_file)
