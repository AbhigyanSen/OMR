from PIL import Image
import os

# Function to convert an image to black and white
def convert_to_bw(image_path):
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale first
            gray_img = img.convert("L")
            
            # Apply threshold to convert to pure black and white
            threshold = 128  # Adjust if needed (0-255)
            bw_img = gray_img.point(lambda x: 255 if x > threshold else 0, '1')
            
            # Save (it will save as 1-bit black & white)
            bw_img.save(image_path)
            print(f"Converted {image_path} to pure black & white.")
    except Exception as e:
        print(f"Error converting {image_path}: {e}")

# Path to the folder containing your images
folder_path = r'P:\PROJECTS\OMR\Images\HSOMR\23072025\Input\Batch003'

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Process each file in the folder
for file in files:
    # Check if the file is an image (you can add more image file extensions as needed)
    if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, file)
        
        # Convert the image to black and white
        convert_to_bw(image_path)