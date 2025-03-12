import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
import torch
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageEnhance
from ImageAndMaskExtractor import ImageAndMaskExtractor
import matplotlib
from matplotlib import colormaps
import threading

# Add pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.upconv4 = self.upconv(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.decoder1 = self.conv_block(128, 64)
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        dec4 = self.upconv4(bottleneck)
        # dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.pad_and_concat(dec4, enc4)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        # dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.pad_and_concat(dec3, enc3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        # dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.pad_and_concat(dec2, enc2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        # dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.pad_and_concat(dec1, enc1)
        dec1 = self.decoder1(dec1)
        return self.conv_last(dec1)
    
    def pad_and_concat(self, upsampled, bypass):
        diffY = bypass.size()[2] - upsampled.size()[2]
        diffX = bypass.size()[3] - upsampled.size()[3]
        upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        return torch.cat([upsampled, bypass], dim=1)

class ImageDataset(Dataset):
    def __init__(self, images):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.images = torch.tensor(images, dtype=torch.float32, device=self.device).unsqueeze(1)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx] 
        
class DemoViewer(Gtk.Window):
    # Color definitions as class variables (tuples for immutability)
    YELLOW: tuple = (255, 255, 0, 255)
    RED: tuple = (255, 0, 0, 255)
    GREEN: tuple = (0, 255, 0, 255)
    BLUE: tuple = (0, 0, 255, 255)
    ORANGE: tuple = (255, 165, 0, 255)
    PURPLE: tuple = (128, 0, 128, 255)
    CYAN: tuple = (0, 255, 255, 255)
    MAGENTA: tuple = (255, 0, 255, 255)
    LIME: tuple = (0, 255, 0, 255)
    PINK: tuple = (255, 192, 203, 255)
    # Collection of all colors
    COLORS: tuple = (RED, YELLOW, GREEN, BLUE, ORANGE, PURPLE, CYAN, MAGENTA, LIME, PINK)

    def __init__(self, zip_content, studyseries, contours, contour_name):
        # zip_content is an instance of ZipFileContents
        # studyseries is an instance of ImageAndMaskExtractor
        # contours is a dictionary of contour names and their corresponding numeric values
        # contour_name is the name of the contour to be displayed initially
        Gtk.Window.__init__(self, title="Demo Viewer")
        self.set_default_size(800, 600)
        self.set_border_width(10)

        self.connect("configure-event", self.on_window_resize)
        
        self.zip_content = zip_content
        self.studyseries = studyseries
        self.contour_names = contours # holds the names of the contours
        self.current_contour = contour_name
        self.model = None
        self.current_image_index = 0
        self.contour_visible = True
        self.prediction_mask_visible = True
        self.model_loaded = False
        self.contrast = 1.0
        self.brightness = 1.0
        self.prediction_threshold = 0.5

        # Set up neural network model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = None #UNet(in_channels=1, out_channels=1).to(self.device)
        # self.criterion = nn.BCEWithLogitsLoss()

        # image data
        self.contour_arrays_df = pd.DataFrame() # this holds the raw sagittal images and masks from ImageAndMaskExtractor.get_sag_contour_pixels
        # self.image_arr = [] # this holds the raw sagittal images - pixels normalized during loading to range 0-1
        # self.mask_arr = [] # this holds the binary mask arrays
        self.masks_PIL = [] # this holds the masks as PIL images
        self.images_PIL = [] # this holds the PIL images
        self.predicted_masks = [] # this holds the predicted masks
        self.binary_predictions = [] # this holds the binary mask predictions

        # Main container
        self.box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.add(self.box)
        
        # Image viewer on the left side
        self.image_GTK = Gtk.Image()
        self.image_scroller = Gtk.ScrolledWindow()
        self.image_scroller.add(self.image_GTK)
        self.image_scroller.connect("scroll-event", self.on_scroll)
        self.box.pack_start(self.image_scroller, True, True, 0)
        
        # Control buttons on the right side
        self.control_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.box.pack_start(self.control_box, False, False, 0)
        
        # Dropdown list for selecting the desired contour
        self.contour_combo = Gtk.ComboBoxText()
        for contour in self.contour_names:
            self.contour_combo.append_text(contour)
        # Set current contour to the contour_name
        for i,key in enumerate(self.contour_names):
            if key == contour_name:
                self.contour_combo.set_active(i)
                break
        self.current_contour = contour_name
        self.contour_combo.connect("changed", self.on_contour_changed)
        self.control_box.pack_start(self.contour_combo, False, False, 0)
        
        # Sliders for adjusting contrast and brightness
        self.contrast_adjustment = Gtk.Adjustment(1.0, 0.0, 5.0, 0.01, 0.01, 0.0)
        self.contrast_slider = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.contrast_adjustment)
        self.contrast_slider.set_digits(2)
        self.contrast_slider.set_value_pos(Gtk.PositionType.RIGHT)
        self.contrast_slider.connect("value-changed", self.on_contrast_changed)
        self.control_box.pack_start(Gtk.Label(label="Contrast"), False, False, 0)
        self.control_box.pack_start(self.contrast_slider, False, False, 0)
        
        self.brightness_adjustment = Gtk.Adjustment(1.0, 0.0, 3.0, 0.01, 0.01, 0.0)
        self.brightness_slider = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.brightness_adjustment)
        self.brightness_slider.set_digits(2)
        self.brightness_slider.set_value_pos(Gtk.PositionType.RIGHT)
        self.brightness_slider.connect("value-changed", self.on_brightness_changed)
        self.control_box.pack_start(Gtk.Label(label="Brightness"), False, False, 0)
        self.control_box.pack_start(self.brightness_slider, False, False, 0)

        self.threshold_adjustment = Gtk.Adjustment(0.2, 0.0, 1.0, 0.01, 0.01, 0.0)
        self.threshold_slider = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.threshold_adjustment)
        self.threshold_slider.set_digits(2)
        self.threshold_slider.set_value_pos(Gtk.PositionType.RIGHT)
        self.threshold_slider.connect("value-changed", self.on_threshold_changed)
        self.control_box.pack_start(Gtk.Label(label="Threshold"), False, False, 0)
        
        # Button to turn the contour on or off
        self.contour_toggle_button = Gtk.Button(label="Toggle Contour")
        self.contour_toggle_button.connect("clicked", self.on_contour_toggle)
        self.control_box.pack_start(self.contour_toggle_button, False, False, 0)
        
        # Button to generate a contour based on the currently loaded pre-calculated model
        self.generate_contour_button = Gtk.Button(label="Generate Contour")
        # self.generate_contour_button.connect("clicked", self.on_generate_AI_contour)
        self.generate_contour_button.connect("clicked", self.generate_predictions)
        self.control_box.pack_start(self.generate_contour_button, False, False, 0)
        
        # Button to change the model with a file selection dialog
        self.change_model_button = Gtk.Button(label="Change Model")
        self.change_model_button.connect("clicked", self.on_change_model)
        self.control_box.pack_start(self.change_model_button, False, False, 0)

        # Add a label under the change_model_button that shows the current loaded model filenmae
        self.model_label = Gtk.Label(label="No model loaded")
        self.control_box.pack_start(self.model_label, False, False, 0)

        # Button to toggle prediction mask visibility
        self.prediction_mask_toggle = Gtk.Button(label="Toggle Prediction Mask")
        self.prediction_mask_toggle.connect("clicked", self.on_prediction_mask_toggle)
        self.control_box.pack_start(self.prediction_mask_toggle, False, False, 0)

        # initially retrieve the image and mask volumes for the first contour
        self.load_contours()
        self.normalize_contours
        self.load_image_volume()
        self.load_mask_volume()

        self.update_image()
        
        self.show_all()
    
    def load_contours(self):
        # get the image and mask data as a list of tuples
        contour_data = self.studyseries.get_sag_contour_pixels(self.current_contour)

        #convert to pandas dataframe, ignore the third element in the tuple
        self.contour_arrays_df = pd.DataFrame({'images':contour_data[0], 'masks':contour_data[1]})

        # Ensure that the images and masks are numpy arrays
        self.contour_arrays_df['images'] = self.contour_arrays_df['images'].apply(lambda x: np.array(x, dtype=np.float32))
        self.contour_arrays_df['masks'] = self.contour_arrays_df['masks'].apply(lambda x: np.array(x, dtype=np.uint8))

        size_pre_culling = self.contour_arrays_df.shape[0]
        # remove rows with size other than 88x512
        self.contour_arrays_df = self.contour_arrays_df[self.contour_arrays_df['images'].apply(lambda x: x.shape == (88, 512))]
        size_post_culling = self.contour_arrays_df.shape[0]
        print(f"Removed {size_pre_culling - size_post_culling} images with size other than 88x512")

        # print information about dataframe
        print(f"Contour arrays dataframe shape: {self.contour_arrays_df.shape}")
        print(f"Contour array datafram stats: {self.contour_arrays_df.describe()}")

    def normalize_contours(self):
        if self.contour_arrays_df.empty:
            print("No contours available")
            return
        # using pandas, normalize
        max_pixel = self.contour_arrays_df['images'].apply(np.max).max()
        min_pixel = self.contour_arrays_df['images'].apply(np.min).min()
        print(f"Normalization: max pixel: {max_pixel}, min pixel: {min_pixel}")
        self.contour_arrays_df['images'] = self.contour_arrays_df['images'].apply(lambda x: x / max_pixel)


    def on_threshold_changed(self, widget):
        self.prediction_threshold = widget.get_value()
        self.update_image()

    def on_prediction_mask_toggle(self, widget):
        if(self.model_loaded == False):
            print("No model loaded")
            return
        self.prediction_mask_visible = not self.prediction_mask_visible
        self.update_image()

    def on_window_resize(self, widget, event):
        # This method will be called when the window is resized
        self.update_image()
        return False  # Allow the default handler to run

    def pil_to_pixbuf(self, pil_image):
        """Convert a PIL Image to GdkPixbuf."""
        if pil_image.mode not in ['RGB', 'RGBA']:
            pil_image = pil_image.convert('RGBA')
        width, height = pil_image.size
        data = pil_image.tobytes()
        has_alpha = pil_image.mode == 'RGBA'
        return GdkPixbuf.Pixbuf.new_from_data(data, GdkPixbuf.Colorspace.RGB, has_alpha, 8, width, height, width * (4 if has_alpha else 3))

    def on_contour_changed(self, widget):
        self.current_contour = widget.get_active_text()
        self.load_contours()
        self.normalize_contours()

        self.load_image_volume
        self.load_mask_volume

        # Reset the image index (back to the middle)
        self.current_image_index = np.round(len(self.images_PIL)/2).astype(int) # set the image index to the midline sagittal image

    def load_image_volume(self):
        self.images_PIL = []

        for img in self.contour_arrays_df['images']:
            img = np.round(img * 255)
            PILImage = Image.fromarray(img.astype("uint8")).convert("RGBA")
            self.images_PIL.append(PILImage)

    def load_mask_volume(self):
        self.masks_PIL = []

        for mask in self.contour_arrays_df['masks']:
            mask = np.round(mask * 255)
            PILMask = Image.fromarray(mask.astype("uint8")).convert("RGBA")
            self.masks_PIL.append(PILMask)

    def on_contrast_changed(self, widget):
        self.contrast = widget.get_value()
        self.update_image()

    def on_brightness_changed(self, widget):
        self.brightness = widget.get_value()
        self.update_image()

    def adjust_contrast_and_brightness(self, pil_image):
        enhancer = ImageEnhance.Contrast(pil_image)
        img = enhancer.enhance(self.contrast)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(self.brightness)

        return img

    def on_contour_toggle(self, widget):
        self.contour_visible = not self.contour_visible
        self.update_image()

    def on_change_model(self, widget):
        self.model_loaded = False
        dialog = Gtk.FileChooserDialog(
            title="Select Model File",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
            buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        )
        dialog.add_filter(self.create_model_filter())
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.predicted_masks = [] # clear the predicted masks
            model_file = dialog.get_filename()
            print(f"Selected model file: {model_file}")
            self.load_model(model_file)
            model_file_name = model_file.split("/")[-1]
            # model_file_name = dialog.get_basename()
            self.model_label.set_text(model_file_name)
            self.model_loaded = True
        dialog.destroy()

    def create_model_filter(self):
        filter = Gtk.FileFilter()
        filter.set_name("Model files")
        filter.add_pattern("*.pth")
        return filter

    def load_model(self, model_file):
        print(f"Model loaded: {model_file}")
        state_dict = torch.load(model_file, map_location=self.device)
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval() # set the model to evaluation mode
        self.update_image()

    def generate_predictions(self):
        """Generate predictions using loaded model"""
        def prediction_thread():
            try:
                if self.model is None:
                    print("No model loaded")
                    return
                    
                if self.contour_arrays_df.empty:
                    print("No images loaded")
                    return

                images = self.contour_arrays_df['images']
            
                self.predicted_masks = [np.zeros_like(img) for img in images]
                self.binary_predictions = [np.zeros_like(img) for img in images]
            
                # Generate predictions for midmost slices, not whole volume (because bladder/rectum near midline)
                mid_index = len(images) // 2
                start_index = max(0, mid_index - 15)
                end_index = min(len(images), mid_index + 15)

                # Create dataset and dataloader
                dataset = ImageDataset(images[start_index:end_index])
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 

                with torch.no_grad():
                    masks = []
                    binary_preds = []
                    for images in dataloader:
                        images = images.to(self.device)
                        output = self.model(images)
                        probs = torch.sigmoid(output) # calculate probabilities
                        bin_pred = probs > self.prediction_threshold # use threshold to get binary predictions
                        masks.append(probs.detach().cpu().numpy())
                        binary_preds.append(bin_pred.detach().cpu().numpy())
                    self.predicted_masks[start_index:end_index] = masks
                    self.binary_predictions[start_index:end_index] = binary_preds

                self.predicted_masks = np.concatenate(self.predicted_masks, axis=0)
                print(f"predicted masks shape: {self.predicted_masks.shape}")
                self.binary_predictions = np.concatenate(self.binary_predictions, axis=0)
                print(f"binary predictions shape: {self.binary_predictions.shape}")
                print(f"Generated predictions for the midmost {end_index-start_index} slices")
            
            except Exception as e:
                print(f"Error generating predictions: {str(e)}")
                self.predicted_masks = []
            
            GLib.idle_add(self.update_image)
        
        # Start a new thread to generate predictions
        thread = threading.Thread(target=prediction_thread)
        thread.daemon = True
        thread.start()


    def update_image(self):
        """Update the current image based on the current index and settings."""
        if self.current_contour is None or not self.images_PIL:
            print("no current contour or images available")
            return
        
        # If no images are loaded, return
        if len(self.images_PIL) == 0:
            return

        # Get the current image from the array
        img = self.images_PIL[self.current_image_index]
        current_image = self.adjust_contrast_and_brightness(img)
        
        # Convert PIL image to GdkPixbuf
        pixbuf = self.pil_to_pixbuf(current_image)

        # Get model predictions for the masks
        if self.model_loaded and self.prediction_mask_visible:
            if len(self.predicted_masks) == 0:
                self.generate_predictions()

        # Overlay prediction if available and visible 
        if len(self.predicted_masks) > 0 and self.prediction_mask_visible and self.model_loaded:
            pred_mask = self.predicted_masks[self.current_image_index] # shape (88, 512)
            
            print("##### Prediction mask #####")
            print(f"pred_mask.ndim = {pred_mask.ndim}, pred_mask.shape = {pred_mask.shape}")
            # apply threshold
            pred_mask_binary = pred_mask > self.prediction_threshold

            print(f"pred_mask shape: {pred_mask.shape}, max/min {np.max(pred_mask)}/{np.min(pred_mask)} dtype: {pred_mask.dtype}")
            print(f"pred_mask_binary shape: {pred_mask_binary.shape}, max/min {np.max(pred_mask_binary)}/{np.min(pred_mask_binary)} dtype: {pred_mask_binary.dtype}")
            
            # Apply colormap - e.g. viridis, Blues, jet, plasma
            cmap = colormaps.get_cmap('plasma')
            heatmap = cmap(pred_mask_binary)[...,:3] # shape (88, 512, 3)
            heatmap = (heatmap * 255).astype(np.uint8)
            #DEBUG
            # print(f"Heatmap shape before concat: {heatmap.shape}")
            # print(f"Min and Max of pred_mask: {np.min(pred_mask)}, {np.max(pred_mask)}") 
            # print(f"##### Heatmap shape: {heatmap.shape}, max/min {np.max(heatmap)}/{np.min(heatmap)} dtype: {heatmap.dtype}")
            # print(f"Value counts of heatmap: {np.unique(heatmap, return_counts=True)}")

            alpha = np.where(pred_mask > 0, 128, 0).astype(np.uint8)
            alpha = alpha[..., np.newaxis] # shape (88, 512, 1)
            # print(f"Alpha shape before concat: {alpha.shape}")


            heatmap_rgba = np.concatenate((heatmap, alpha), axis=-1) # shape (88, 512, 4)
            # print(f"Final shape of heatmap_rgba after concat: {heatmap_rgba.shape}")
            print(f'heatmap_rgba shape: {heatmap_rgba.shape} max/min: {np.max(heatmap_rgba)}/{np.min(heatmap_rgba)}, dtype = {heatmap_rgba.dtype}')

            heatmap_PIL = Image.fromarray(heatmap_rgba, mode='RGBA')
            heatmap_pixbuf = self.pil_to_pixbuf(heatmap_PIL)

            # verify heatmap_pixbuf dimensions and datatype and compare to pixbuf
            print(f"heatmap_pixbuf shape: {heatmap_pixbuf.get_width()}x{heatmap_pixbuf.get_height()}, dtype: {heatmap_pixbuf.get_rowstride()}")
            print(f"pixbuf shape: {pixbuf.get_width()}x{pixbuf.get_height()}, dtype: {pixbuf.get_rowstride()}")

            GdkPixbuf.Pixbuf.composite(heatmap_pixbuf, pixbuf, 
                                   0, 0, 
                                   heatmap_pixbuf.get_width(), 
                                   heatmap_pixbuf.get_height(), 
                                   0, 0, 1, 1, 
                                   GdkPixbuf.InterpType.BILINEAR, 255)

            # debug: verify that self.predicted_masks dimensions and datatype are the same as self.mask_arr
            # print(f"predicted_masks shape: {pred_mask.shape}, dtype: {pred_mask.dtype}")
            # print(f"mask_arr shape: {self.mask_arr[self.image_index].shape}, dtype: {self.mask_arr[self.image_index].dtype}")
            # print(f"Image shape: {self.image_arr[self.image_index].shape}")

        # Apply mask if contour is visible
        if self.contour_visible and len(self.masks_PIL) > 0:
            mask = self.masks_PIL[self.current_image_index]
            mask = self.convert_PIL_mask_to_color_and_transparency(mask, self.get_color_based_on_contour(self.current_contour))
            mask_pixbuf = self.pil_to_pixbuf(mask)
            # now overlay
            GdkPixbuf.Pixbuf.composite(mask_pixbuf, pixbuf, 0, 0, 
                                            mask_pixbuf.get_width(), 
                                            mask_pixbuf.get_height(), 
                                            0, 0, 1, 1, 
                                            GdkPixbuf.InterpType.BILINEAR, 255)
        
        # Get the allocation of the image widget to determine the available space
        allocation = self.image_scroller.get_allocation()
        available_width = max(1,allocation.width)
        available_height = max(1,allocation.height)
        # print(f"Available space: {available_width}x{available_height}")

        # Calculate the scale factor while preserving aspect ratio
        original_width = pixbuf.get_width()
        original_height = pixbuf.get_height()
        # print(f"Original image size: {original_width}x{original_height}")

        scale_factor = min(available_width / original_width, available_height / original_height)

        # Scale the pixbuf to fit within the available space
        new_width = max(1, int(original_width * scale_factor))
        new_height = max(1, int(original_height * scale_factor))
        # print(f"Scaling image to {new_width}x{new_height}")

        scaled_pixbuf = pixbuf.scale_simple(new_width, new_height, GdkPixbuf.InterpType.BILINEAR)

        # Set the image in the Gtk.Image widget
        self.image_GTK.set_from_pixbuf(scaled_pixbuf)

    def convert_PIL_mask_to_color_and_transparency(self, mask_PIL, color):
        transparency_nonzero = 90
        transparency_zero = 0
        # Convert the image to RGBA mode if it isn't already
        if mask_PIL.mode != 'RGBA':
            mask_PIL = mask_PIL.convert('RGBA')
    
        # Create a new image with the same size as the original mask
        result = Image.new('RGBA', mask_PIL.size, (0, 0, 0, 0))  # Transparent black background
    
        # Get the pixel data
        pixels = mask_PIL.load()
        result_pixels = result.load()
    
        for y in range(mask_PIL.height):
            for x in range(mask_PIL.width):
                # If the pixel in the mask is non-zero, color it with the new color
                if pixels[x, y][0] > 0:  # Assuming the mask uses shades of gray or black/white
                    result_pixels[x, y] = color[:3] + (transparency_nonzero,)
                else:
                    # If the pixel is zero (or very close to zero), make it fully transparent
                    result_pixels[x, y] = (0, 0, 0, transparency_zero)
    
        return result


    def get_color_based_on_contour(self, contour_name):
        for i, key in enumerate(self.contour_names):
            if key == contour_name:
                return self.COLORS[i % len(self.COLORS)]

    def on_scroll(self, widget, event):
        """Handle scroll events to navigate through images."""
        if event.direction == Gdk.ScrollDirection.UP:
            self.current_image_index = max(0, self.current_image_index - 1)
        elif event.direction == Gdk.ScrollDirection.DOWN:
            self.current_image_index = min(len(self.images_PIL) - 1, self.current_image_index + 1)
        elif event.direction == Gdk.ScrollDirection.SMOOTH:
            if event.delta_y > 0:
                self.current_image_index = max(0, self.current_image_index - 1)
            elif event.delta_y < 0:
                self.current_image_index = min(len(self.images_PIL) - 1, self.current_image_index + 1)
        self.update_image()
