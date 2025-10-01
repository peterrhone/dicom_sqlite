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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# DISTANCE_FROM_CENTER_FOR_GENERATED_CONTOUR = 55

# class UNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1):
#         super(UNet, self).__init__()
#         self.encoder1 = self.conv_block(in_channels, 64)
#         self.encoder2 = self.conv_block(64, 128)
#         self.encoder3 = self.conv_block(128, 256)
#         self.encoder4 = self.conv_block(256, 512)
#         self.bottleneck = self.conv_block(512, 1024)
#         self.upconv4 = self.upconv(1024, 512)
#         self.decoder4 = self.conv_block(1024, 512)
#         self.upconv3 = self.upconv(512, 256)
#         self.decoder3 = self.conv_block(512, 256)
#         self.upconv2 = self.upconv(256, 128)
#         self.decoder2 = self.conv_block(256, 128)
#         self.upconv1 = self.upconv(128, 64)
#         self.decoder1 = self.conv_block(128, 64)
#         self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def upconv(self, in_channels, out_channels):
#         return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(F.max_pool2d(enc1, 2))
#         enc3 = self.encoder3(F.max_pool2d(enc2, 2))
#         enc4 = self.encoder4(F.max_pool2d(enc3, 2))
#         bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat([dec4, enc4], dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat([dec3, enc3], dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat([dec2, enc2], dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat([dec1, enc1], dim=1)
#         dec1 = self.decoder1(dec1)
#         return self.conv_last(dec1)
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def pad_and_concat(self, upsampled, bypass):
        diffY = bypass.size()[2] - upsampled.size()[2]
        diffX = bypass.size()[3] - upsampled.size()[3]
        upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        return torch.cat([upsampled, bypass], dim=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        dec4 = self.upconv4(bottleneck)
        dec4 = self.pad_and_concat(dec4, enc4)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = self.pad_and_concat(dec3, enc3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = self.pad_and_concat(dec2, enc2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = self.pad_and_concat(dec1, enc1)
        dec1 = self.decoder1(dec1)
        return self.conv_last(dec1)

class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = np.array(self.images.iloc[idx], dtype=np.float32)
        # Images are already normalized in normalize_contours(), don't normalize again
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)
        return img

class DemoViewer(Gtk.Window):
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
    COLORS: tuple = (RED, YELLOW, GREEN, BLUE, ORANGE, PURPLE, CYAN, MAGENTA, LIME, PINK)
    WINDOW_SOFT_TISSUE = (400, 40)
    WINDOW_BONE = (2000, 600)
    WINDOW_LUNG = (1500, -500)

    def __init__(self, zip_content, studyseries, contours, startup_window):
        # try to attach this window to the same Gtk.Application as the startup window
        app = None
        try:
            app = startup_window.get_application()
        except Exception:
            app = None
        try:
            Gtk.Window.__init__(self, title="Demo Viewer", application=app)
        except TypeError:
            Gtk.Window.__init__(self, title="Demo Viewer")
        # match the main window icon name so GNOME shows the same running icon
        try:
            self.set_icon_name("ai-view-icon")
        except Exception:
            pass
        # set WM_CLASS like the main window to help matching to the .desktop entry
        try:
            self.set_wmclass("ai-view", "ai-view")
        except Exception:
            pass
        self.set_default_size(800, 600)
        self.set_border_width(10)
        self.startup_window = startup_window
        self.connect("configure-event", self.on_window_resize)
        self.connect("destroy", self.on_destroy)
        
        self.zip_content = zip_content
        self.studyseries = studyseries
        self.contour_names = contours
        # Select the first contour automatically
        self.current_contour = list(contours.keys())[0] if contours else None
        self.model = None
        self.current_image_index = 0
        self.contour_visible = True
        self.prediction_mask_visible = True
        self.model_loaded = False
        self.contrast = 1.0
        self.brightness = 1.0
        self.prediction_threshold = 0.5
        self.slope = getattr(studyseries, 'slope', 1.0)
        self.intercept = getattr(studyseries, 'intercept', 0.0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.contour_arrays_df = pd.DataFrame()
        self.masks_PIL = []
        self.images_PIL = []
        self.predicted_masks = []
        self.binary_predictions = []

        self.box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.add(self.box)
        
        self.image_GTK = Gtk.Image()
        self.image_scroller = Gtk.ScrolledWindow()
        self.image_scroller.add(self.image_GTK)
        self.image_scroller.connect("scroll-event", self.on_scroll)
        self.box.pack_start(self.image_scroller, True, True, 0)
        
        self.control_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.box.pack_start(self.control_box, False, False, 0)
        
        self.contour_combo = Gtk.ComboBoxText()
        for contour in self.contour_names:
            self.contour_combo.append_text(contour)
        # Set the first contour as active
        if self.contour_names:
            self.contour_combo.set_active(0)
        self.contour_combo.connect("changed", self.on_contour_changed)
        self.control_box.pack_start(self.contour_combo, False, False, 0)
        
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
        
        self.threshold_adjustment = Gtk.Adjustment(0.5, 0.0, 1.0, 0.01, 0.01, 0.0)
        self.threshold_slider = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.threshold_adjustment)
        self.threshold_slider.set_digits(2)
        self.threshold_slider.set_value_pos(Gtk.PositionType.RIGHT)
        self.threshold_slider.connect("value-changed", self.on_threshold_changed)
        self.control_box.pack_start(Gtk.Label(label="Threshold"), False, False, 0)
        self.control_box.pack_start(self.threshold_slider, False, False, 0)

        self.window_combo = Gtk.ComboBoxText()
        self.window_combo.append_text("Soft Tissue")
        self.window_combo.append_text("Bone")
        self.window_combo.append_text("Lung")
        self.window_combo.set_active(0)
        self.window_combo.connect("changed", self.on_window_changed)
        self.control_box.pack_start(self.window_combo, False, False, 0)
        
        self.contour_toggle_button = Gtk.Button(label="Toggle Contour")
        self.contour_toggle_button.connect("clicked", self.on_contour_toggle)
        self.control_box.pack_start(self.contour_toggle_button, False, False, 0)
        
        self.generate_contour_button = Gtk.Button(label="Generate Contour")
        self.generate_contour_button.connect("clicked", self.generate_predictions)
        self.control_box.pack_start(self.generate_contour_button, False, False, 0)
        
        self.change_model_button = Gtk.Button(label="Change Model")
        self.change_model_button.connect("clicked", self.on_change_model)
        self.control_box.pack_start(self.change_model_button, False, False, 0)

        self.model_label = Gtk.Label(label="No model loaded")
        self.control_box.pack_start(self.model_label, False, False, 0)

        self.prediction_mask_toggle = Gtk.Button(label="Toggle Prediction Mask")
        self.prediction_mask_toggle.connect("clicked", self.on_prediction_mask_toggle)
        self.control_box.pack_start(self.prediction_mask_toggle, False, False, 0)

        self.slice_label = Gtk.Label(label="Slice 0/0")
        self.control_box.pack_start(self.slice_label, False, False, 0)

        self.load_contours()
        self.normalize_contours()
        self.load_image_volume()
        self.load_mask_volume()
        self.current_image_index = len(self.images_PIL) // 2
        self.update_image()
        self.show_all()
    
    def load_contours(self):
        contour_data = self.studyseries.get_sag_contour_pixels(self.current_contour)
        images = [item[0] for item in contour_data]
        masks = [item[1] for item in contour_data]
        self.contour_arrays_df = pd.DataFrame({'images': images, 'masks': masks})
        self.contour_arrays_df['masks'] = self.contour_arrays_df['masks'].apply(
            lambda x: np.clip(np.array(x), 0, 1).astype(np.uint8)
        )
        size_pre_culling = self.contour_arrays_df.shape[0]
        self.contour_arrays_df = self.contour_arrays_df[
            self.contour_arrays_df['images'].apply(lambda x: x.shape == (88, 512))
        ]
        size_post_culling = self.contour_arrays_df.shape[0]
        print(f"Removed {size_pre_culling - size_post_culling} images with size other than 88x512")

    def normalize_contours(self):
        if self.contour_arrays_df.empty:
            print("No contours available")
            return
        max_pixel = self.contour_arrays_df['images'].apply(np.max).max()
        min_pixel = self.contour_arrays_df['images'].apply(np.min).min()
        print(f"Normalization: max pixel: {max_pixel}, min pixel: {min_pixel}")
        # Clip to 0-8000 like training data, then normalize to [0,1]
        self.contour_arrays_df['images_normalized'] = self.contour_arrays_df['images'].apply(lambda x: np.clip(x, 0, 8000) / 8000.0)

    def load_image_volume(self):
        self.images_PIL = []
        for img in self.contour_arrays_df['images']:
            img_scaled = np.clip(img, 0, np.max(img)).astype(np.uint8)
            self.images_PIL.append(Image.fromarray(img_scaled, mode='L').convert("RGBA"))

    def load_mask_volume(self):
        self.masks_PIL = []
        for mask in self.contour_arrays_df['masks']:
            mask = np.clip(mask, 0, 1) * 255
            PILMask = Image.fromarray(mask.astype("uint8")).convert("RGBA")
            self.masks_PIL.append(PILMask)

    def on_threshold_changed(self, widget):
        self.prediction_threshold = widget.get_value()
        self.update_image()

    def on_prediction_mask_toggle(self, widget):
        if not self.model_loaded:
            print("No model loaded")
            return
        self.prediction_mask_visible = not self.prediction_mask_visible
        self.update_image()

    def on_window_changed(self, widget):
        self.update_image()

    def on_window_resize(self, widget, event):
        print(f"Window resized to: {event.width}x{event.height}")
        self.image_scroller.set_size_request(-1, -1)
        self.image_scroller.queue_resize_no_redraw()
        self.image_scroller.check_resize()
        GLib.idle_add(self.update_image)
        self.image_GTK.queue_draw()
        self.queue_draw()
        return False

    def pil_to_pixbuf(self, pil_image):
        if pil_image.mode not in ['RGB', 'RGBA']:
            pil_image = pil_image.convert('RGBA')
        width, height = pil_image.size
        data = pil_image.tobytes()
        has_alpha = pil_image.mode == 'RGBA'
        return GdkPixbuf.Pixbuf.new_from_data(data, GdkPixbuf.Colorspace.RGB, has_alpha, 8, width, height, width * (4 if has_alpha else 3))

    def on_contour_changed(self, widget):
        self.current_contour = widget.get_active_text()
        self.predicted_masks = []
        self.binary_predictions = []
        self.load_contours()
        self.normalize_contours()
        self.load_image_volume()
        self.load_mask_volume()
        self.current_image_index = len(self.images_PIL) // 2
        print(f"Switched to contour: {self.current_contour}, image index: {self.current_image_index}")
        if self.model_loaded:
            self.generate_predictions()
        self.update_image()

    def on_contrast_changed(self, widget):
        self.contrast = widget.get_value()
        self.update_image()

    def on_brightness_changed(self, widget):
        self.brightness = widget.get_value()
        self.update_image()

    def adjust_contrast_and_brightness(self, pil_image, contrast_factor, brightness_factor):
        np_image = np.array(pil_image)
        np_image = np.uint8(np_image)
        pil_image = Image.fromarray(np_image).convert("RGBA")
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness_factor)
        return pil_image

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
            self.predicted_masks = []
            self.binary_predictions = []
            model_file = dialog.get_filename()
            print(f"Selected model file: {model_file}")
            self.load_model(model_file)
            model_file_name = model_file.split("/")[-1]
            self.model_label.set_text(model_file_name)
            self.model_loaded = True
            self.generate_predictions()
        dialog.destroy()

    def create_model_filter(self):
        filter = Gtk.FileFilter()
        filter.set_name("Model files")
        filter.add_pattern("*.pth")
        return filter

    def load_model(self, model_file):
        try:
            print(f"Loading model: {model_file}")
            state_dict = torch.load(model_file, map_location=self.device)
            self.model = UNet(in_channels=1, out_channels=1).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Model loaded successfully: {model_file}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
            self.model_loaded = False
            self.model_label.set_text("Failed to load model")

    def generate_predictions(self, widget=None):
        def prediction_thread():
            try:
                if self.model is None:
                    print("No model loaded")
                    return
                if self.contour_arrays_df.empty:
                    print("No images loaded")
                    return
                images = self.contour_arrays_df['images_normalized']
                self.predicted_masks = [np.zeros_like(img) for img in self.contour_arrays_df['images']]
                self.binary_predictions = [np.zeros_like(img) for img in self.contour_arrays_df['images']]
                mid_index = len(images) // 2
                # Always select 10 slices centered around mid_index
                num_slices = 10
                if len(images) < num_slices:
                    print(f"Warning: not enough slices ({len(images)}) for prediction, padding with zeros.")
                    pad_shape = images.iloc[0].shape if len(images) > 0 else (88, 512)
                    pad_img = np.zeros(pad_shape, dtype=np.float32)
                    pad_count = num_slices - len(images)
                    selected_images = list(images.values) + [pad_img] * pad_count
                else:
                    start_index = max(0, mid_index - num_slices // 2)
                    end_index = start_index + num_slices
                    if end_index > len(images):
                        end_index = len(images)
                        start_index = end_index - num_slices
                    selected_images = images[start_index:end_index].values
                print(f"Selected {len(selected_images)} slices for prediction (indices: {start_index} to {end_index-1})")
                images_np = np.stack(selected_images)
                dataset = ImageDataset(pd.Series(list(selected_images)))
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                with torch.no_grad():
                    masks = []
                    binary_preds = []
                    for batch in dataloader:
                        print(f"Batch shape: {batch.shape}")
                        output = self.model(batch)
                        print(f"Model logits min: {output.min().item()}, max: {output.max().item()}")
                        probs = torch.sigmoid(output).squeeze(1)
                        bin_pred = (probs > self.prediction_threshold).float()
                        masks.append(probs.cpu().numpy())
                        binary_preds.append(bin_pred.cpu().numpy())
                        print(f"Model output probs min: {probs.min().item()}, max: {probs.max().item()}")
                        print(f"Binary pred sum: {bin_pred.sum().item()}")
                    masks = np.concatenate(masks, axis=0)
                    binary_preds = np.concatenate(binary_preds, axis=0)
                    # Only update the slices that were actually predicted
                    for i, idx in enumerate(range(start_index, end_index)):
                        if idx < len(self.predicted_masks):
                            self.predicted_masks[idx] = masks[i]
                            self.binary_predictions[idx] = binary_preds[i]
                print(f"Generated predictions for slices {start_index} to {end_index-1}")
                GLib.idle_add(self.update_image)
            except Exception as e:
                print(f"Error generating predictions: {str(e)}")
                self.predicted_masks = []
                GLib.idle_add(self.update_image)
        thread = threading.Thread(target=prediction_thread)
        thread.daemon = True
        thread.start()

    def update_image(self):
        if self.current_contour is None or not self.contour_arrays_df['images'].size:
            print("No current contour or images available")
            return
        raw_img = self.contour_arrays_df['images'].iloc[self.current_image_index]
        img_hu = raw_img * self.slope + self.intercept
        print(f"HU values: min={np.min(img_hu)}, max={np.max(img_hu)}")
        window_width, window_level = self.get_window_params()
        lower = window_level - window_width / 2
        upper = window_level + window_width / 2
        img_windowed = np.clip(img_hu, lower, upper)
        img_scaled = (img_windowed - lower) / (upper - lower) * 255
        img_scaled = np.uint8(img_scaled)
        print(f"Windowed values: min={np.min(img_windowed)}, max={np.max(img_windowed)}")
        print(f"Scaled values: min={np.min(img_scaled)}, max={np.max(img_scaled)}")
        current_image = Image.fromarray(img_scaled, mode='L').convert("RGBA")
        print(f"PIL image pixel range: {np.min(np.array(current_image))}, {np.max(np.array(current_image))}")
        current_image = self.adjust_contrast_and_brightness(current_image, self.contrast, self.brightness)
        pixbuf = self.pil_to_pixbuf(current_image)
        if self.model_loaded and self.prediction_mask_visible and len(self.predicted_masks) > 0:
            pred_mask = self.predicted_masks[self.current_image_index]
            pred_mask = np.squeeze(pred_mask)
            print(f"Prediction mask min: {np.min(pred_mask)}, max: {np.max(pred_mask)}")
            pred_mask_binary = (pred_mask > self.prediction_threshold).astype(np.uint8)
            print(f"Binary mask min: {np.min(pred_mask_binary)}, max: {np.max(pred_mask_binary)}")
            cmap = colormaps.get_cmap('plasma')
            heatmap = cmap(pred_mask_binary)[..., :3] * 255
            heatmap = heatmap.astype(np.uint8)
            alpha = np.where(pred_mask_binary > 0, 128, 0).astype(np.uint8)
            heatmap_rgba = np.dstack((heatmap, alpha))
            heatmap_PIL = Image.fromarray(heatmap_rgba, mode='RGBA')
            heatmap_pixbuf = self.pil_to_pixbuf(heatmap_PIL)
            GdkPixbuf.Pixbuf.composite(
                heatmap_pixbuf, pixbuf,
                0, 0, heatmap_pixbuf.get_width(), heatmap_pixbuf.get_height(),
                0, 0, 1, 1, GdkPixbuf.InterpType.BILINEAR, 255
            )
        if self.contour_visible and len(self.masks_PIL) > 0:
            mask = self.masks_PIL[self.current_image_index]
            mask = self.convert_PIL_mask_to_color_and_transparency(
                mask, self.get_color_based_on_contour(self.current_contour)
            )
            mask_pixbuf = self.pil_to_pixbuf(mask)
            GdkPixbuf.Pixbuf.composite(
                mask_pixbuf, pixbuf,
                0, 0, mask_pixbuf.get_width(), mask_pixbuf.get_height(),
                0, 0, 1, 1, GdkPixbuf.InterpType.BILINEAR, 255
            )
        allocation = self.image_scroller.get_allocation()
        print(f"Using allocation: {allocation.width}x{allocation.height}")
        available_width = max(1, allocation.width)
        available_height = max(1, allocation.height)
        original_width = pixbuf.get_width()
        original_height = pixbuf.get_height()
        scale_factor = min(available_width / original_width, available_height / original_height)
        new_width = max(1, int(original_width * scale_factor))
        new_height = max(1, int(original_height * scale_factor))
        scaled_pixbuf = pixbuf.scale_simple(new_width, new_height, GdkPixbuf.InterpType.BILINEAR)
        self.image_GTK.set_from_pixbuf(scaled_pixbuf)
        self.slice_label.set_text(f"Slice {self.current_image_index + 1}/{len(self.images_PIL)}")
        self.image_scroller.queue_draw()

    def convert_PIL_mask_to_color_and_transparency(self, mask_PIL, color):
        transparency_nonzero = 90
        transparency_zero = 0
        if mask_PIL.mode != 'RGBA':
            mask_PIL = mask_PIL.convert('RGBA')
        result = Image.new('RGBA', mask_PIL.size, (0, 0, 0, 0))
        pixels = mask_PIL.load()
        result_pixels = result.load()
        for y in range(mask_PIL.height):
            for x in range(mask_PIL.width):
                if pixels[x, y][0] > 0:
                    result_pixels[x, y] = color[:3] + (transparency_nonzero,)
                else:
                    result_pixels[x, y] = (0, 0, 0, transparency_zero)
        return result

    def get_window_params(self):
        active_index = self.window_combo.get_active()
        match active_index:
            case 0:
                return self.WINDOW_SOFT_TISSUE
            case 1:
                return self.WINDOW_BONE
            case 2:
                return self.WINDOW_LUNG
            case _:
                print("Invalid window type selected, defaulting to soft tissue.")
                return self.WINDOW_SOFT_TISSUE

    def get_color_based_on_contour(self, contour_name):
        for i, key in enumerate(self.contour_names):
            if key == contour_name:
                return self.COLORS[i % len(self.COLORS)]

    def on_scroll(self, widget, event):
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

    def on_destroy(self, widget):
        self.startup_window.show_all()