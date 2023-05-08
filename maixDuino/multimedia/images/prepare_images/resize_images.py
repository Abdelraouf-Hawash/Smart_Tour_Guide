import cv2
import os

input_path = "./input"
output_path = "./output"
img_names = os.listdir(input_path)
for img_name in img_names:
    print(img_name)
    img = cv2.imread(f"{input_path}/{img_name}")
    img = cv2.resize(img, (320, 240), interpolation=3)
    cv2.imwrite(f"{output_path}/{img_name}", img)
