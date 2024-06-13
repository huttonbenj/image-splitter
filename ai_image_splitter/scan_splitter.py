import os
import requests
import base64
import cv2
import numpy as np
import shutil

def send_image_for_processing(image_path, debug_dir):
    try:
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            response = requests.post('http://localhost:8000/api/process/', files=files)
            
            if response.status_code == 200:
                result = response.json()
                processed_images = result['processed_images']
                debug_images = result['debug_images']

                decoded_images = []
                for img_data in processed_images:
                    processed_image_data = base64.b64decode(img_data)
                    nparr = np.frombuffer(processed_image_data, np.uint8)
                    processed_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    decoded_images.append(processed_image)

                save_debug_images(debug_images, debug_dir, os.path.basename(image_path))
                
                return decoded_images
            else:
                print('Error:', response.json())
                return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def save_debug_images(debug_images, debug_dir, image_name):
    base_name = os.path.splitext(image_name)[0]
    for step, img_data in debug_images.items():
        img_path = os.path.join(debug_dir, f"{base_name}_{step}.jpg")
        img = base64.b64decode(img_data)
        nparr = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(img_path, img)
        print(f"Saved debug image to {img_path}")

def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def process_directory(directory_path, debug_dir):
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    output_directory = os.path.join(directory_path, 'processed')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Clear the output and debug directories
    clear_directory(output_directory)
    clear_directory(debug_dir)

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(directory_path, filename)
            print(f"Processing {image_path}")
            processed_images = send_image_for_processing(image_path, debug_dir)

            if processed_images is not None:
                for idx, img in enumerate(processed_images):
                    output_path = os.path.join(output_directory, f"processed_{filename.split('.')[0]}_{idx}.jpg")
                    cv2.imwrite(output_path, img)
                    print(f"Saved processed image to {output_path}")
            else:
                print(f"Failed to process {image_path}")

if __name__ == "__main__":
    directory_path = '../ai_image_splitter/images'
    debug_directory = '../ai_image_splitter/images/debug_images'
    process_directory(directory_path, debug_directory)
