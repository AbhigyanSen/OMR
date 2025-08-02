import os
import cv2
import json

def draw_bboxes_from_json(json_path, images_folder, output_folder):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    for img_name, details in data.items():
        image_path = os.path.join(images_folder, img_name)
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Could not read: {image_path}")
            continue

        mapped_fields = details.get("mapped_fields", {})

        # Draw all bounding boxes
        for field_name, field_data in mapped_fields.items():
            bbox = field_data.get("bbox", None)
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                color = (0, 0, 0)   # Green color
                thickness = 2
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                # Optional: put label
                cv2.putText(img, field_name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save annotated image
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, img)
        print(f"✅ Saved annotated image: {output_path}")

    print(f"All images processed. Results saved in: {output_folder}")
    
    
# MAIN
json_path = r"D:\Projects\OMR\new_abhigyan\Restructure\Images\HSOMR\31072025\Output\BATCH018\annotate_BATCH018\field_mappings.json"
images_folder = r"D:\Projects\OMR\new_abhigyan\Restructure\Images\HSOMR\31072025\Output\BATCH018\annotate_BATCH018\mapped_images"
output_folder = r"D:\Projects\OMR\new_abhigyan\Restructure\KeyFields\Results"

draw_bboxes_from_json(json_path, images_folder, output_folder)
