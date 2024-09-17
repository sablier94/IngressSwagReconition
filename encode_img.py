import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Path to your test image
image_path = "test.jpg"
encoded_image = encode_image(image_path)

# Write the encoded image to a file for testing
with open("event.json", "w") as event_file:
    event_file.write(f'{{"body": "{encoded_image}"}}')