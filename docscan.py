import imutils
from imutils import perspective
from python_imagesearch.imagesearch import imagesearch
from skimage.filters import threshold_local
import numpy as np 
import argparse
import cv2
import tkinter as tk
from tkinter import filedialog

def ask_for_image_path():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

    if not file_path:
        print("No file selected. Exiting.")
        exit()

    return file_path

# Ask the user for the image path
image_path = ask_for_image_path()

# Read the input image
image = cv2.imread(image_path)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# Display original and edge-detected images
print("1: EDGE DETECTION")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Finding contours in the edged image
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# Loop over contours
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # Screen check
    if len(approx) == 4:
        screenCnt = approx
        print("Screen contour found:", screenCnt)
        break

print("2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Applying four point transform to obtain top-down view of the original image
warped = imutils.perspective.four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# Convert warped to grayscale then threshold it
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > T).astype("uint8") * 255

# Display original and scanned images
print("3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()


print ( """
╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓
╓╓╓╓╓╓╓╓╓╓╓╓█████████████╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓
╓╓╓╓╓╓╓╓╓█████╓╓╓█╓╓╓╓╓╓███████╓╓╓╓╓╓╓╓╓
╓╓╓╓╓╓╓╓██╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓██╓╓╓╓╓╓╓
╓╓╓╓╓╓██╓╓╓╓╓█████╓╓██████╓╓╓╓╓╓██╓╓╓╓╓╓
╓╓╓╓╓╓█╓╓╓╓╓╓█╓╓╓██╓██╓╓╓█╓╓╓╓╓╓╓╓█╓╓╓╓╓
╓╓╓╓╓█╓╓╓╓╓╓╓╓█████╓╓████╓╓╓╓╓╓╓╓╓██╓╓╓╓
╓╓╓╓╓█╓╓╓╓╓╓╓╓╓╓╓╓╓█╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓█╓╓╓╓
╓╓╓╓╓█╓╓╓╓╓╓╓╓╓╓╓╓╓██╓╓╓╓╓╓╓╓╓╓╓╓╓╓█╓╓╓╓
╓╓╓╓╓█╓╓╓╓╓╓╓█╓╓╓╓╓╓███╓╓╓╓╓█╓╓╓╓╓╓█╓╓╓╓
╓╓╓╓╓██╓╓╓╓╓╓█╓╓╓╓╓╓██╓╓╓╓╓╓█╓╓╓╓╓╓█╓╓╓╓
╓╓╓╓╓╓██╓╓╓╓╓███╓╓╓╓╓╓╓╓╓╓╓██╓╓╓╓╓╓╓█╓╓╓
╓╓╓╓╓╓╓██╓╓╓╓╓╓█████████████╓╓╓╓╓╓╓╓█╓╓╓
╓╓╓╓╓╓╓╓██╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓█╓╓╓
╓╓╓╓╓╓╓╓╓███╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓█╓╓╓╓
╓╓╓╓╓╓╓╓╓╓╓███╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓██╓╓╓╓╓
╓╓╓╓╓╓╓╓╓╓╓╓╓╓████╓╓╓╓╓╓╓╓╓╓╓╓███╓╓╓╓╓╓╓
╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓█████████████╓╓╓╓╓╓╓╓╓╓
╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓╓
DocScan By DeadBrain 0_0...
""")