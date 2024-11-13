import cv2
import pytesseract
from PIL import Image
import numpy as np

# Load the image
image_path = 'PocketTrainsMap.png'
image = cv2.imread(image_path)

# Convert to grayscale for better OCR and edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding to binarize the image (optional: adjust threshold for OCR accuracy)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Use OCR to detect city names
cities_text = pytesseract.image_to_string(thresh, config='--psm 6')
cities = cities_text.splitlines()
print("Detected city names:", cities)

# Detect lines (tracks) using Hough Line Transform
edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=5)

# Draw detected lines on a copy of the image for visualization
line_image = np.copy(image)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save or display the image with detected tracks
output_path = 'detected_tracks.png'
cv2.imwrite(output_path, line_image)
print("Saved image with detected tracks to:", output_path)

# Display image with detected tracks (optional, for local execution)
# cv2.imshow("Detected Tracks", line_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
