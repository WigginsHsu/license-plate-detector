import cv2
import os

plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

image_paths = [
    "license_plate_1.png",
    "license_plate_2.png",
    "license_plate_3.png",
    "license_plate_4.png"
]

output_dir = "plate_rois"
os.makedirs(output_dir, exist_ok=True)

for path in image_paths:
    print(f"processing images: {path}")
    img = cv2.imread(path)
    if img is None:
        print("failed to read the images")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    print(f"detected {len(plates)} license plates")

    for i, (x, y, w, h) in enumerate(plates):
        plate_roi = img[y:y+h, x:x+w]
        roi_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(path))[0]}_plate_{i+1}.png")
        cv2.imwrite(roi_path, plate_roi)
        print(f"save license plate area: {roi_path}")

print("done")