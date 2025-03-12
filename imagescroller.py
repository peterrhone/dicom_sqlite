from tkinter import Tk, PhotoImage, Label
from PIL import Image, ImageTk

class ImageScroller():
    def __init__(self, images=None, masks=None, root=None, aspect_ratio=1):
        self.root_passed = False
        if root is None:
            self.root = Tk()
        else:
            self.root_passed = True
            self.root = root
        self.image_arr = images
        self.mask_arr = masks
        self.index = 256
        self.indices_to_save = []
        self.show_mask = True
        self.aspect_ratio=aspect_ratio

        if self.image_arr is not None:
            self.img = self.combine_images(self.image_arr[self.index], self.mask_arr[self.index]) 
            self.imgtk = ImageTk.PhotoImage(self.img)
            self.label = Label(self.root, image=self.imgtk)
            self.label.pack()

            self.label.bind("<Button-4>", self.scroll_up)
            self.label.bind("<Button-5>", self.scroll_down)
            self.root.bind("<Configure>", self.on_window_resize)
            self.root.bind("s", self.image_to_save)
            self.root.bind("w", self.image_to_save)
            self.root.bind("m", self.toggle_mask)

            self.text_label = Label(self.root, 
                                    text = "",
                                    fg="white",
                                    bg="black",
                                    font="Helvetica 16 bold")
            self.text_label.place(x=10,y=10)
            self.original_imgw = self.img.width
            self.original_imgh = self.img.height
            self.update_dimensions_text()

            # the destructor should be called when the window is closed or q or escape is pressed
            # self.root.protocol("WM_DELETE_WINDOW", self.__del__)
            # self.root.bind("q", self.__del__)
            # self.root.bind("<Escape>", self.__del__)
    
    def update_dimensions_text(self):
        if hasattr(self, 'img'):
            self.text_label.config(text=f"Image: {self.original_imgw} x {self.original_imgh}")

    def resize_image(self, img):
        """Resize image maintaining aspect ratio"""
        width = self.root.winfo_width() - 2
        height = int(width * self.aspect_ratio)
        return img.resize((width, height))
    
    def combine_images(self, image, mask):
        if self.show_mask:
            combined_img = Image.alpha_composite(image, mask)
        else:
            combined_img = image
        return combined_img

    def toggle_mask(self, event):
        self.show_mask = not self.show_mask
        #force refresh
        self.img = self.combine_images(self.image_arr[self.index], self.mask_arr[self.index])
        self.img = self.resize_image(self.img)
        self.imgtk = ImageTk.PhotoImage(self.img)
        self.label.configure(image=self.imgtk)

    def scroll_up(self, event):
        if hasattr(self, 'image_arr'):
            if self.index < len(self.image_arr) - 1:
                self.index += 1
                self.img = self.combine_images(self.image_arr[self.index], self.mask_arr[self.index]) 
                self.img = self.resize_image(self.img)
                self.imgtk = ImageTk.PhotoImage(self.img)
                self.label.configure(image=self.imgtk)
                self.update_dimensions_text()
    
    def scroll_down(self, event):
        if hasattr(self, 'image_arr'):
            if self.index > 0:
                self.index -= 1
                self.img = self.combine_images(self.image_arr[self.index], self.mask_arr[self.index]) 
                self.img = self.resize_image(self.img)
                self.imgtk = ImageTk.PhotoImage(self.img)
                self.label.configure(image=self.imgtk)
                self.update_dimensions_text()

    def image_to_save(self, event):
        self.indices_to_save.append(self.index)
        print(f"Image {self.index} added to save list")
    
    def on_window_resize(self, event):
        if hasattr(self, 'img'):
            self.img = self.combine_images(self.image_arr[self.index], self.mask_arr[self.index])
            self.img = self.resize_image(self.img)
            self.imgtk = ImageTk.PhotoImage(self.img)
            self.label.configure(image=self.imgtk)
            self.update_dimensions_text()

    def start(self):
        self.root.mainloop()

    
