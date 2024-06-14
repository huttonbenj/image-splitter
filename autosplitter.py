import cv2
import numpy as np
from tkinter import Tk, filedialog, Label, Listbox, Scrollbar, EXTENDED
from tkinter import ttk
from ttkthemes import ThemedTk
from PIL import Image, ImageTk
import os

class AutoSplitterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AutoSplitter Photo Scanning Software")
        self.image_list = []
        self.cropped_images = []

        # Apply custom theme
        self.style = ttk.Style()
        self.style.theme_use('equilux')
        
        # Configure styles
        self.style.configure("TButton", font=("Helvetica", 12, "bold"), padding=10)
        self.style.configure("TLabel", font=("Helvetica", 14))
        self.style.configure("TFrame", padding=20)
        self.style.configure("TListbox", font=("Helvetica", 12), borderwidth=0)
        
        self.frame = ttk.Frame(root, style="TFrame")
        self.frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.label = ttk.Label(self.frame, text="AutoSplitter Photo Scanning Software", font=("Helvetica", 18, "bold"), anchor="center")
        self.label.grid(row=0, column=0, columnspan=3, pady=10, sticky="ew")

        self.button_frame = ttk.Frame(self.frame, style="TFrame")
        self.button_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky="ew")

        self.open_files_button = ttk.Button(self.button_frame, text="Open File(s)", command=self.open_files)
        self.open_files_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.open_folder_button = ttk.Button(self.button_frame, text="Open Folder", command=self.open_folder)
        self.open_folder_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.review_button = ttk.Button(self.button_frame, text="Review batch detection", command=self.review_detection)
        self.review_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.file_listbox = Listbox(self.frame, selectmode=EXTENDED, font=("Helvetica", 12), bd=0, highlightthickness=0, relief="flat")
        self.file_listbox.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.file_listbox.yview)
        self.file_listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.grid(row=2, column=3, sticky="ns")

        self.save_all_button = ttk.Button(self.frame, text="Save All", command=self.save_all)
        self.save_all_button.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.close_application_button = ttk.Button(self.frame, text="Close Application", command=self.close_application)
        self.close_application_button.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.image_panel = Label(root)
        self.image_panel.pack(pady=10, fill="both", expand=True)

        # Configure grid weights for responsiveness
        self.frame.grid_rowconfigure(2, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=1)

    def open_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        self.image_list = list(files)
        self.file_listbox.delete(0, 'end')
        for file in self.image_list:
            self.file_listbox.insert('end', os.path.basename(file))

    def open_folder(self):
        folder = filedialog.askdirectory()
        self.image_list = [os.path.join(folder, file) for file in os.listdir(folder) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        self.file_listbox.delete(0, 'end')
        for file in self.image_list:
            self.file_listbox.insert('end', os.path.basename(file))

    def review_detection(self):
        if not self.image_list:
            return
        self.cropped_images = []
        for image_path in self.image_list:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_height, img_width = img.shape[:2]
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                area = w * h
                # Filter criteria: aspect ratio, area, and relative size
                if 0.5 < aspect_ratio < 2.0 and 0.02 * img_width * img_height < area < 0.9 * img_width * img_height:
                    # Additional filter to exclude small, narrow objects
                    if w > 0.2 * img_width and h > 0.2 * img_height:
                        cropped = img[y:y+h, x:x+w]
                        self.cropped_images.append((image_path, cropped))

        self.display_images()

    def display_images(self):
        self.file_listbox.delete(0, 'end')
        for i, (path, img) in enumerate(self.cropped_images):
            self.file_listbox.insert('end', f"{os.path.basename(path)} - Crop {i+1}")

    def save_all(self):
        save_dir = filedialog.askdirectory()
        if not save_dir:
            return
        for i, (path, img) in enumerate(self.cropped_images):
            base_name = os.path.basename(path)
            name, ext = os.path.splitext(base_name)
            save_path = os.path.join(save_dir, f"{name}_crop_{i+1}{ext}")
            cv2.imwrite(save_path, img)

    def close_application(self):
        self.root.quit()

if __name__ == "__main__":
    root = ThemedTk(theme="equilux")
    app = AutoSplitterApp(root)
    root.mainloop()