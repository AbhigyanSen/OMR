import cv2
import numpy as np
import os

def visualize_yolo_annotations(image_path, annotations_data, classes_data, output_image_name="del.jpg"):
    """
    Visualizes YOLO format bounding box annotations and class names on an image.

    Args:
        image_path (str): The path to the input image.
        annotations_data (str): A string containing the YOLO annotation data (e.g., "0 0.5 0.5 0.1 0.1\n...").
        classes_data (str): A string containing class names, one per line (e.g., "class1\nclass2\n...").
        output_image_name (str): The name to save the output image as.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}. Please check the path and file existence.")
        return

    img_height, img_width, _ = img.shape

    # Parse class names
    class_names = [name.strip() for name in classes_data.split('\n') if name.strip()]

    # Parse annotations
    annotations = []
    for line in annotations_data.split('\n'):
        if not line.strip():
            continue
        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])
        x_center_norm, y_center_norm, width_norm, height_norm = parts[1:]
        annotations.append({
            'class_id': class_id,
            'x_center_norm': x_center_norm,
            'y_center_norm': y_center_norm,
            'width_norm': width_norm,
            'height_norm': height_norm
        })

    # Draw bounding boxes and class names
    for ann in annotations:
        class_id = ann['class_id']
        x_center_norm = ann['x_center_norm']
        y_center_norm = ann['y_center_norm']
        width_norm = ann['width_norm']
        height_norm = ann['height_norm']

        # Convert normalized coordinates to pixel coordinates
        x_center = int(x_center_norm * img_width)
        y_center = int(y_center_norm * img_height)
        width = int(width_norm * img_width)
        height = int(height_norm * img_height)

        # Calculate top-left and bottom-right corner for rectangle
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw rectangle
        color = (0, 255, 0)  # Green color for bounding box (BGR format)
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Superimpose class name
        label = class_names[class_id] if class_id < len(class_names) else f"Unknown_Class_{class_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Position text just above the bounding box
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10 # Adjust if too close to top edge

        cv2.putText(img, label, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

    # Save the processed image
    cv2.imwrite(output_image_name, img)
    print(f"Processed image saved as '{output_image_name}'")

# --- Main Execution ---
if __name__ == "__main__":
    # Load classes from classes.txt
    classes_file_path = r"D:\Projects\OMR\new_abhigyan\Phase1\Options\classes.txt"                   # CLASSES.TXT
    try:
        with open(classes_file_path, 'r') as f:
            classes_content = f.read()
    except FileNotFoundError:
        print(f"Error: {classes_file_path} not found.  Please check the path.")
        exit()

    # Load annotations from annotations.txt
    annotations_file_path = r"D:\Projects\OMR\new_abhigyan\Phase1\Options\BE23-01-01003.txt"        # ANNOTATIONS.TXT
    try:
        with open(annotations_file_path, 'r') as f:
            annotations_content = f.read()
    except FileNotFoundError:
        print(f"Error: {annotations_file_path} not found. Please check the path.")
        exit()

    # --- IMPORTANT: Replace this with the actual path to YOUR image ---
    # Example: image_to_visualize_path = r"D:\Projects\OMR\new_abhigyan\Phase1\Data\images\100418.jpg"
    image_to_visualize_path = r"D:\Projects\OMR\new_abhigyan\Phase1\Options\BE23-01-01003.jpg"              # IMAGE

    visualize_yolo_annotations(image_to_visualize_path, annotations_content, classes_content)