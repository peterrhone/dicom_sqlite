import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import io
from sqlitedata import SqliteData

class DatabaseChooser(Gtk.Dialog):
    def __init__(self):
        Gtk.Dialog.__init__(self, title="Choose Database File")
        self.add_button(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL)
        self.add_button(Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        
        self.chooser = Gtk.FileChooserWidget(action=Gtk.FileChooserAction.OPEN)
        self.chooser.set_select_multiple(False)
        filter_sqlite = Gtk.FileFilter()
        filter_sqlite.set_name("SQLite files")
        filter_sqlite.add_pattern("*.sqlite")
        self.chooser.add_filter(filter_sqlite)
        
        box = self.get_content_area()
        box.add(self.chooser)
        self.show_all()

class ImageViewer:
    def __init__(self, db_path=None):
        self.root = tk.Tk()  # Ensure root is initialized first
        self.current_index = 0
        self.show_mask = True
        
        # Choose database
        if db_path:
            self.db_path = db_path
        else:
            chooser = DatabaseChooser()
            response = chooser.run()
            if response == Gtk.ResponseType.OK:
                self.db_path = chooser.chooser.get_filename()
            chooser.destroy()
        
        # Load database
        self.db = SqliteData(self.db_path)

        # Get total number of images
        self.total_images = self.db.get_total_images()
        if self.total_images == 0:
            print("No images found in database")
            return
            
        # Setup Tkinter window
        self.root.title("CBCT Image Slice and Mask Viewer")
        self.root.geometry("800x600")
        self.root.bind("<Configure>", self.on_resize)
        
        # Image display
        self.label = ttk.Label(self.root)
        self.label.pack()
        
        # Info display
        self.info_label = ttk.Label(self.root, text="")
        self.info_label.pack()
        
        self.keybindings_label = ttk.Label(self.root, text="←: Previous, →: Next, d: Delete, m: Toggle Mask, Mousewheel: Scroll")
        self.keybindings_label.pack()
        
        # Key bindings
        self.root.bind('<Left>', self.prev_image)
        self.root.bind('<Right>', self.next_image)
        self.root.bind('d', self.delete_current)
        self.root.bind('m', self.toggle_mask)
        self.root.bind('<MouseWheel>', self.on_mouse_wheel)  # Add mouse wheel binding
        self.root.bind('<Button-4>', self.on_mouse_wheel)  # For Linux
        self.root.bind('<Button-5>', self.on_mouse_wheel)  # For Linux
        
        self.show_current_image()
        
    def on_resize(self, event):
        self.show_current_image()

    def on_mouse_wheel(self, event):
        if event.num == 5 or event.delta == -120:
            self.prev_image(event)
        if event.num == 4 or event.delta == 120:
            self.next_image(event)

    def show_current_image(self):
        if self.total_images == 0:
            return

        try:
            current = self.db.read_db_by_index(self.current_index)
            # Unpack all 14 fields
            id, pat_id, cbct_id, sag_slice_idx, img_bytes, mask_bytes, height, width, structure_id, mask_volume_ml, mask_voxel_count, x_spacing, y_spacing, layer_spacing = current
            
            # Convert image bytes to numpy array
            img_arr = np.frombuffer(img_bytes, dtype=np.uint16).reshape(height, width)
            img_arr = ((img_arr - img_arr.min()) / (img_arr.max() - img_arr.min()) * 255).astype(np.uint8)
     
            # Create PIL image
            img = Image.fromarray(img_arr, mode='L')
        
            if self.show_mask and mask_bytes:
                mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(height, width)
                mask_arr = mask_arr.astype(np.uint8) * 255
                mask = Image.fromarray(mask_arr, mode='L')

                # Create red transparent overlay
                overlay = Image.new('RGBA', img.size, (255, 0, 0, 0))
                alpha_mask = mask.point(lambda p: p * 0.3)
                overlay.putalpha(alpha_mask)  
                img = Image.alpha_composite(img.convert('RGBA'), overlay)
            
            # Resize image
            available_width = self.root.winfo_width()
            available_height = self.root.winfo_height() - self.info_label.winfo_height() - self.keybindings_label.winfo_height()
            img = img.resize((available_width, available_height-20), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(img)
            self.label.configure(image=self.photo)
        
            # Update info text
            self.info_label.configure(text=f"ID: {id}, Patient: {pat_id}, CBCT: {cbct_id}, Slice: {sag_slice_idx}, Structure: {structure_id}, Volume: {mask_volume_ml} mL")
            self.keybindings_label.configure(text="Left: Previous, Right: Next, d: Delete, m: Toggle Mask, Mousewheel: Scroll")

        except Exception as e:
            print(f"Error displaying image at index {self.current_index}: {e}")
        
    def next_image(self, event):
        if self.current_index < self.total_images - 1:
            self.current_index += 1
            self.show_current_image()
            
    def prev_image(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
            
    def delete_current(self, event):
        if self.total_images > 0:
            try:
                id = self.db.read_db_by_index(self.current_index)[0]
                self.db.delete_by_id(id)
                self.total_images -= 1
                if self.current_index >= self.total_images:
                    self.current_index = max(0, self.total_images - 1)
                self.show_current_image()
            except Exception as e:
                print(f"Error deleting image at index {self.current_index}: {e}")
            
    def toggle_mask(self, event):
        self.show_mask = not self.show_mask
        self.show_current_image()
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    viewer = ImageViewer()
    viewer.run()
