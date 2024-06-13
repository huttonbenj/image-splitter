from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import cv2
import numpy as np
import base64
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ProcessImageView(APIView):
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({'error': 'No image file provided'}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image']
        image_data = image_file.read()
        
        try:
            processed_images, debug_images = self.process_image(image_data)
            return Response({'processed_images': processed_images, 'debug_images': debug_images})
        except Exception as e:
            logging.error(f"Image processing failed: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def save_debug_image(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return encoded_image

    def process_image(self, image_data):
        # Convert the image data to a numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Dictionary to hold debug images
        debug_images = {}

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        debug_images["gray"] = self.save_debug_image(gray)
        logging.debug("Converted to grayscale")

        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        debug_images["blurred"] = self.save_debug_image(blurred)
        logging.debug("GaussianBlur applied")

        # Use adaptive thresholding to create a binary image
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        debug_images["adaptive_thresh"] = self.save_debug_image(adaptive_thresh)
        logging.debug("Adaptive Thresholding applied")

        # Dilate edges to close gaps with refined parameters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(adaptive_thresh, kernel, iterations=2)
        debug_images["dilated"] = self.save_debug_image(dilated)
        logging.debug("Edges dilated")

        # Perform morphological closing to close gaps inside objects
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
        debug_images["closing"] = self.save_debug_image(closing)
        logging.debug("Morphological closing done")

        # Find contours
        contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logging.debug(f"Found {len(contours)} contours")

        # Create a list to hold encoded cropped images
        processed_images = []

        # Filter and process contours
        image_height, image_width = image.shape[:2]
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            area = cv2.contourArea(approx)
            logging.debug(f"Contour: x={x}, y={y}, w={w}, h={h}, area={area}")

            # Adjusted filtering criteria
            if area < 5000 or w < 100 or h < 100 or w > image_width * 0.9 or h > image_height * 0.9:
                logging.debug(f"Contour filtered out: x={x}, y={y}, w={w}, h={h}, area={area}")
                continue

            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image_width - x, w + 2 * padding)
            h = min(image_height - y, h + 2 * padding)

            cropped_image = image[y:y + h, x:x + w]

            _, buffer = cv2.imencode('.jpg', cropped_image)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            processed_images.append(encoded_image)

        logging.debug(f"Total processed images: {len(processed_images)}")
        return processed_images, debug_images

# If you have a main function to run this code as a script
if __name__ == "__main__":
    from django.core.management import execute_from_command_line
    execute_from_command_line()
