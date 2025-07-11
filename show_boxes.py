import requests
from PIL import Image, ImageDraw

# 1. Send image to your API
response = requests.post(
    "http://127.0.0.1:8000/detect",
    files={"file": open("your_apple_image.jpg", "rb")}
)
results = response.json()

# 2. Draw boxes on the image
img = Image.open("your_apple_image.jpg")
draw = ImageDraw.Draw(img)

for detection in results["detections"]:
    x1, y1, x2, y2 = detection["bbox"]
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, y1-20), f"{detection['class']} {detection['confidence']:.2f}", fill="red")

img.show()