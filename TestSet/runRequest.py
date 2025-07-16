import requests
import json
import os

# --- Configuration ---
# Ensure your Flask app is running at this address
API_URL = "http://10.4.1.66:8003/predict"

# Base directory where your 'ICR Images' folder is located
# IMPORTANT: Adjust this path to match where your 'ICR Images' folder is on your CLIENT system.
# Example: If 'ICR Images' is directly in 'D:/OMR_DEV/VLM/', then BASE_IMAGE_DIR should be 'D:/OMR_DEV/VLM/ICR Images'
BASE_IMAGE_DIR = r"D:\Projects\OMR\new_abhigyan\TestSet\annotate_BE_24_Series\OMR" # <--- ADJUST THIS PATH

# Output JSON file name
OUTPUT_JSON_FILE = r"D:\Projects\OMR\new_abhigyan\TestSet\annotate_BE_24_Series\OMR\ICR_Images.json"

# List of all possible categories you expect (corresponding to your subfolder names)
ALL_CATEGORIES = ["roll_no", "reg_no", "booklet_no"]

# Dictionary to store all results before saving to JSON
# Format: { "image_name": { "booklet_no": "", "reg_no": "", "roll_no": "" } }
all_extracted_data = {}

# --- Main Logic ---
print(f"Starting API client to process images from: {BASE_IMAGE_DIR}")

if not os.path.isdir(BASE_IMAGE_DIR):
    print(f"Error: Base image directory not found at {BASE_IMAGE_DIR}. Please check the path.")
else:
    # First pass: Collect all image paths and their intended categories
    # This helps in initializing the all_extracted_data structure correctly
    # before sending requests, ensuring all images are accounted for.
    images_to_process = [] # List of tuples: (image_file_path, image_name_without_ext, category_name)

    for category_folder_name in os.listdir(BASE_IMAGE_DIR):
        category_folder_path = os.path.join(BASE_IMAGE_DIR, category_folder_name)

        if os.path.isdir(category_folder_path) and category_folder_name in ALL_CATEGORIES:
            category_to_extract = category_folder_name

            for image_filename in os.listdir(category_folder_path):
                image_file_path = os.path.join(category_folder_path, image_filename)

                if os.path.isfile(image_file_path) and image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    image_name_without_ext = os.path.splitext(image_filename)[0]
                    images_to_process.append((image_file_path, image_name_without_ext, category_to_extract))

                    # Initialize the entry for this image if it doesn't exist
                    if image_name_without_ext not in all_extracted_data:
                        all_extracted_data[image_name_without_ext] = {cat: "" for cat in ALL_CATEGORIES}
                else:
                    print(f"  Skipping non-image file or directory: {image_filename} in {category_folder_path}")
        else:
            print(f"Skipping non-category folder or non-directory item: {category_folder_name}")

    # Second pass: Send requests and populate the results
    for image_file_path, image_name_without_ext, category_to_extract in images_to_process:
        print(f"\n--- Processing image: '{image_name_without_ext}' for category: '{category_to_extract}' ---")

        try:
            with open(image_file_path, 'rb') as f:
                # Determine content type based on extension (simple heuristic)
                if image_file_path.lower().endswith('.png'):
                    content_type = 'image/png'
                else:
                    content_type = 'image/jpeg' # Default for jpg, jpeg, etc.

                files = {'image': (os.path.basename(image_file_path), f, content_type)}
                data = {'category': category_to_extract}

                response = requests.post(API_URL, files=files, data=data)

                if response.status_code == 200:
                    api_response_json = response.json()
                    print("  API Raw Response:", api_response_json)

                    # Update the specific category for this image
                    # The API is expected to return {"category_name": "extracted_value"}
                    if category_to_extract in api_response_json:
                        extracted_value = api_response_json[category_to_extract]
                        all_extracted_data[image_name_without_ext][category_to_extract] = extracted_value
                        print(f"  Extracted '{category_to_extract}': '{extracted_value}'")
                    else:
                        # Handle unexpected JSON structure from API
                        all_extracted_data[image_name_without_ext][category_to_extract] = "Error: API output format unexpected"
                        print(f"  Error: API output format unexpected for {category_to_extract}. Response: {api_response_json}")

                else:
                    error_message = f"Error: API returned status code {response.status_code}. Content: {response.text}"
                    all_extracted_data[image_name_without_ext][category_to_extract] = f"Error: {response.status_code}"
                    print(f"  {error_message}")

        except requests.exceptions.ConnectionError as e:
            all_extracted_data[image_name_without_ext][category_to_extract] = "Error: Connection Failed"
            print(f"  Error: Could not connect to the API at {API_URL}. Is the Flask server running and accessible? {e}")
            # Consider if you want to break here or continue to try other images
            # For now, it will mark this specific category for this image as failed.
        except json.JSONDecodeError as e:
            all_extracted_data[image_name_without_ext][category_to_extract] = "Error: Invalid JSON response"
            print(f"  Error: Could not decode JSON response from API for {image_name_without_ext}, {category_to_extract}. {e}")
            print(f"  Raw response text: {response.text}")
        except Exception as e:
            all_extracted_data[image_name_without_ext][category_to_extract] = f"Error: {str(e)}"
            print(f"  An unexpected error occurred while processing {image_filename}: {e}")

# --- Save Results to JSON File ---
print(f"\n--- All image processing complete. Saving results to {OUTPUT_JSON_FILE} ---")
try:
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_extracted_data, f, indent=4, ensure_ascii=False)
    print(f"Successfully saved all extracted data to {OUTPUT_JSON_FILE}")
    print("\n--- Final Consolidated Data ---")
    print(json.dumps(all_extracted_data, indent=2))
except Exception as e:
    print(f"Error: Could not save data to JSON file {OUTPUT_JSON_FILE}: {e}")