import cv2
import numpy as np
import math
import os
import json
import csv
import math
import glob
import sys
from datetime import datetime

class OMRProcessor:
    def __init__(self, image_path, annotations_path, classes_path, target_width, target_height):
        self.image_path = image_path
        self.annotations_path = annotations_path
        self.classes_path = classes_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        self.image = cv2.resize(self.image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        self.original_image = self.image.copy()

        self.original_width = target_width
        self.original_height = target_height
        self.classes = self._load_classes()                 # Load classes.txt first
        self.annotations = self._load_annotations()         # Annotations need original image dimensions
        
        print(f"Original Image dimensions: {self.original_width}x{self.original_height}")
        # print(f"Loaded annotations: {self.annotations}")
        # print(f"Loaded classes: {self.classes}")

        # Store the transformation matrix from original to deskewed image
        self.M_transform = None 
        # Store the deskewed image dimensions
        self.deskewed_width = self.original_width
        self.deskewed_height = self.original_height

    def _load_annotations(self):
        annotations = {}
        try:
            with open(self.annotations_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0]) 
                        # Convert normalized coordinates to pixel coordinates based on original image
                        x_center = float(parts[1]) * self.original_width
                        y_center = float(parts[2]) * self.original_height
                        norm_width = float(parts[3]) * self.original_width
                        norm_height = float(parts[4]) * self.original_height

                        # Calculate top-left and bottom-right corners
                        x1 = int(x_center - norm_width / 2)
                        y1 = int(y_center - norm_height / 2)
                        x2 = int(x_center + norm_width / 2)
                        y2 = int(y_center + norm_height / 2)

                        # Store as (x1, y1, x2, y2)
                        annotations.setdefault(class_id, []).append((x1, y1, x2, y2))
        except FileNotFoundError:
            print(f"Annotation file not found at {self.annotations_path}. Proceeding without it.")
        return annotations

    def _load_classes(self):
        classes = []
        try:
            with open(self.classes_path, 'r') as f:
                for line in f:
                    # Remove any common newline characters like \r and \n
                    classes.append(line.strip().replace('\r', '')) 
        except FileNotFoundError:
            print(f"Classes file not found at {self.classes_path}. Proceeding without it.")
        return classes

    def _get_class_id(self, class_name):
        try:
            return self.classes.index(class_name)
        except ValueError:
            return -1

    def detect_anchor_points(self):
        detected_anchors = []

        # Define the anchor class IDs. These map to the *index* in your classes.txt
        anchor_class_names = ['anchor_1', 'anchor_2', 'anchor_3', 'anchor_4']

        # Work on the original image
        self.image = self.original_image.copy()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        for class_name_str in anchor_class_names:
            class_id = self._get_class_id(class_name_str)

            if class_id != -1 and class_id in self.annotations:
                for bbox in self.annotations[class_id]:
                    x1, y1, x2, y2 = bbox

                    # Define a dynamic search area around the current bbox
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    buffer_scale = 2.0
                    buffer_x = int(max(30, bbox_width * buffer_scale / 2))
                    buffer_y = int(max(30, bbox_height * buffer_scale / 2))

                    search_x1 = max(0, x1 - buffer_x)
                    search_y1 = max(0, y1 - buffer_y)
                    search_x2 = min(self.original_width, x2 + buffer_x)
                    search_y2 = min(self.original_height, y2 + buffer_y)

                    if search_x2 <= search_x1 or search_y2 <= search_y1:
                        print(f"Warning: Invalid search area for anchor {class_name_str}. Skipping.")
                        continue

                    roi = blurred[search_y1:search_y2, search_x1:search_x2]

                    if roi.shape[0] == 0 or roi.shape[1] == 0:
                        print(f"Warning: ROI is empty for anchor {class_name_str}. Skipping.")
                        continue

                    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    found_anchor_contour = None
                    min_area = 0.2 * bbox_width * bbox_height
                    max_area = 2.0 * bbox_width * bbox_height

                    for c in contours:
                        area = cv2.contourArea(c)
                        if area < min_area or area > max_area:
                            continue

                        (cx, cy, cw, ch) = cv2.boundingRect(c)
                        aspect_ratio = cw / float(ch)

                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

                        is_square = len(approx) == 4 and 0.8 <= aspect_ratio <= 1.2
                        is_circle_like = len(approx) > 6 and 0.8 <= aspect_ratio <= 1.2

                        if is_square or is_circle_like:
                            if found_anchor_contour is None or area > cv2.contourArea(found_anchor_contour):
                                found_anchor_contour = c

                    if found_anchor_contour is not None:
                        (fx, fy, fw, fh) = cv2.boundingRect(found_anchor_contour)
                        center_x = search_x1 + fx + fw // 2
                        center_y = search_y1 + fy + fh // 2
                        det_x1 = search_x1 + fx
                        det_y1 = search_y1 + fy
                        det_x2 = search_x1 + fx + fw
                        det_y2 = search_y1 + fy + fh

                        detected_anchors.append({
                            'class_name': class_name_str,
                            'bbox': (det_x1, det_y1, det_x2, det_y2),
                            'center': (center_x, center_y),
                            'area': cv2.contourArea(found_anchor_contour),
                        })

                        print(f"Detected {class_name_str}: Center=({center_x}, {center_y}), BBox=({det_x1},{det_y1},{det_x2},{det_y2}), Area={cv2.contourArea(found_anchor_contour):.2f}")
                    else:
                        print(f"No suitable anchor contour detected for {class_name_str}.")
            else:
                print(f"Warning: Class {class_name_str} (ID: {class_id}) not found in annotations.")

        return detected_anchors, self.image, None

   
    def visualize_results(self, detected_anchors, output_filename):
        display_image = self.image.copy()            # Use the deskewed image
        anchor_data_for_json = {}                   
    
        for anchor in detected_anchors:
            x1, y1, x2, y2 = anchor['bbox']
            center_x, center_y = int(anchor['center'][0]), int(anchor['center'][1])
            class_name = anchor['class_name']
    
            # ⬇️ Store both bbox and center
            anchor_data_for_json[class_name] = {
                "center": [center_x, center_y],
                "bbox": [x1, y1, x2, y2]
            }

            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(display_image, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)              # Changed text color for visibility
    
        cv2.imwrite(output_filename, display_image)
        print(f"Results saved to {output_filename}")
        return anchor_data_for_json                                                 # Return both bbox and center now

def get_annotation_files(annotations_dir):
    # --- Validate base folder ---
    if not os.path.isdir(annotations_dir):
        print(f"❌ Annotations folder not found: {annotations_dir}")
        sys.exit(1)

    labels_dir = os.path.join(annotations_dir, "labels")
    images_dir = os.path.join(annotations_dir, "images")
    classes_file = os.path.join(annotations_dir, "classes.txt")

    # --- Validate required subfolders and classes file ---
    if not os.path.isdir(labels_dir):
        print(f"❌ Labels folder not found: {labels_dir}")
        sys.exit(1)
    if not os.path.isdir(images_dir):
        print(f"❌ Images folder not found: {images_dir}")
        sys.exit(1)
    if not os.path.isfile(classes_file):
        print(f"❌ classes.txt file missing: {classes_file}")
        sys.exit(1)

    # --- Get label file ---
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if len(label_files) != 1:
        print(f"❌ Expected exactly 1 label file in {labels_dir}, found {len(label_files)}")
        sys.exit(1)
    annotations_file = label_files[0]

    # --- Get image file ---
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    if len(image_files) != 1:
        print(f"❌ Expected exactly 1 image file in {images_dir}, found {len(image_files)}")
        sys.exit(1)
    annotated_image_path = image_files[0]

    # --- Check filename match (excluding extension) ---
    label_name = os.path.splitext(os.path.basename(annotations_file))[0]
    image_name = os.path.splitext(os.path.basename(annotated_image_path))[0]
    if label_name != image_name:
        print(f"❌ Label file '{label_name}' does not match image file '{image_name}'")
        sys.exit(1)

    print(f"✅ Using annotation file: {annotations_file}")
    print(f"✅ Using annotated image: {annotated_image_path}")
    print(f"✅ Using classes file: {classes_file}")

    return annotated_image_path, annotations_file, classes_file

def compute_skew_angle(anchor1_center, anchor2_center):
    x1, y1 = anchor1_center
    x2, y2 = anchor2_center

    base = abs(x2 - x1)
    height = y2 - y1  # This retains sign for clockwise/counter-clockwise

    if base == 0:
        return 90.0 if height > 0 else -90.0

    angle_rad = math.atan2(height, base)
    angle_deg = math.degrees(angle_rad)

    return round(angle_deg, 4)

def save_rescaled_images(image_folder_path, output_process_image_path, ref_width, ref_height):
    folder_name = os.path.basename(folder_path.rstrip("\\/"))
    print(f"BATCH NAME: {folder_name}")
    processed_dir = os.path.join(output_process_image_path, f"processed_{folder_name}")
    os.makedirs(processed_dir, exist_ok=True)
    print(f"Processing images in: {processed_dir}")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not read image: {image_path}")
                continue

            # Resize image to match reference dimensions
            resized = cv2.resize(image, (ref_width, ref_height), interpolation=cv2.INTER_LINEAR)

            # Save the resized image
            save_path = os.path.join(processed_dir, filename)
            cv2.imwrite(save_path, resized)
            # print(f"✅ Saved: {save_path}")


# Generate a generalized JSON file for the batch ---------------------------------------------------------
def generate_generalized_json(base_folder, omr_template_name, date, folder_path, all_image_anchor_data, warning_dir):
    import os
    import json
    from datetime import datetime

    template_name = omr_template_name                     
    batch_name = os.path.basename(folder_path)                      
    process_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    # Count total images (jpg, jpeg, png)
    total_images = len([f for f in os.listdir(folder_path) 
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    output_json = {
        "TEMPLATE": template_name,
        "BATCHNAME": batch_name,
        "PROCESSDT": process_dt,
        "COUNT": total_images,      
        "IMAGES": []
    }

    # Sequence counter starting at 1
    seq_counter = 1

    for image_name, data in all_image_anchor_data.items():
        image_abs_path = os.path.abspath(os.path.join(folder_path, image_name))
        warning_image_path = os.path.join(warning_dir, image_name)
        skewed = "Y" if os.path.exists(warning_image_path) else "N"

        image_entry = {
            "IMAGENAME": image_abs_path.replace("/", "\\"),
            "SEQ": seq_counter,  # Incremental sequence starting from 1
            "SKEWED": skewed,
            "ERROR": "N" if data.get("valid_for_option_mapping", False) else "Y",
            "FIELDS": []
        }

        output_json["IMAGES"].append(image_entry)
        seq_counter += 1  # Increment sequence

    # Create final directory structure: <base_folder>/Images/<TEMPLATE>/<date>/Output/<BATCHNAME>
    final_output_dir = os.path.join(base_folder, "Images", template_name, date, "Output", batch_name)
    os.makedirs(final_output_dir, exist_ok=True)

    generalized_json_path = os.path.join(final_output_dir, f"{batch_name}.json")
    with open(generalized_json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    print(f"\n✅ Generalized batch JSON saved to: {generalized_json_path}") 
    

# Main execution
if __name__ == "__main__":

    # Define paths
    base_folder = r"D:\Projects\OMR\new_abhigyan\Restructure"
    
    # omr_template_name = "HSOMR"
    # date = "23072025"
    # batch_name = "Batch003"   
    # Expect arguments: omr_template_name, date, batch_name
    
    # # Inputs from Command Line
    if len(sys.argv) != 4:
        print("Usage: python AnchorDetection.py <omr_template_name> <date> <batch_name>")
        sys.exit(1)

    omr_template_name, date, batch_name = sys.argv[1:4]
    
    folder_path = os.path.join(base_folder, "Images", omr_template_name, date, "Input", batch_name)
    annotations_dir = os.path.join(base_folder, "Annotations", omr_template_name)
    annotated_image_path, annotations_file, classes_file = get_annotation_files(annotations_dir)
    
    ref_img = cv2.imread(annotated_image_path)
    if ref_img is None:
        raise FileNotFoundError(f"Annotated image not found at {annotated_image_path}")
    ref_height, ref_width = ref_img.shape[:2]
    print(f"📏 Reference dimensions from annotation image: {ref_width}x{ref_height}")
    
    # Save all images in rescaled (reference) dimensions to a clean folder
    output_folder_path = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name)
    save_rescaled_images(
        image_folder_path=folder_path,
        output_process_image_path=output_folder_path,
        ref_width=ref_width,
        ref_height=ref_height
    )
    
    # Create output directory based on folder name
    folder_name = os.path.basename(folder_path.rstrip("\\/"))
    output_dir = os.path.join(output_folder_path, "anchor_" + folder_name)
    
    warning_dir = os.path.join(base_folder, "Images", omr_template_name, date, "warnings")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(warning_dir, exist_ok=True)

    # CSV Path
    csv_output_path = os.path.join(output_dir, "anchor_centers.csv")
    
    # Initialize a dictionary to hold anchor data for all images
    anchor_json_path = os.path.join(output_dir, "anchor_centers.json") # New JSON file name
    all_image_anchor_data = {}

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            print(f"\nProcessing {image_path}...")

            try:
                # processor = OMRProcessor(image_path, annotations_file, classes_file)
                processor = OMRProcessor(image_path, annotations_file, classes_file, ref_width, ref_height)
                detected_anchors, deskewed_img_result, M_transform_result = processor.detect_anchor_points()

                # Dynamically determine expected anchors from classes.txt
                expected_anchors = [cls for cls in processor.classes if cls.startswith("anchor_")]
                expected_anchor_count = len(expected_anchors)

                if len(detected_anchors) != expected_anchor_count:
                    print(f"⚠️ Detected {len(detected_anchors)} anchors, but expected {expected_anchor_count}. Moving to warnings folder.")
                    warning_path = os.path.join(warning_dir, filename)
                    cv2.imwrite(warning_path, deskewed_img_result if deskewed_img_result is not None else processor.original_image)
                    all_image_anchor_data[filename] = {
                        "anchors": {anchor['class_name']: anchor['center'] for anchor in detected_anchors},
                        "skew_angle": None,
                        "valid_for_option_mapping": False
                    }

                else:
                    output_image_path = os.path.join(output_dir, filename)
                    anchor_full_data = processor.visualize_results(detected_anchors, output_image_path)

        
                    print("\n--- Detected Anchor Details ---")
                    for anchor in detected_anchors:
                        print(f"Class: {anchor['class_name']}")
                        print(f"  Bounding Box (x1, y1, x2, y2): {anchor['bbox']}")
                        print(f"  Center (x, y): {anchor['center']}")
                        print(f"  Area: {anchor['area']:.2f}")
                        print("-" * 30)
                        
                    # Compute skew angle if both anchor_1 and anchor_2 exist
                    if "anchor_1" in anchor_full_data and "anchor_2" in anchor_full_data:
                        angle = compute_skew_angle(
                            anchor_full_data["anchor_1"]["center"],
                            anchor_full_data["anchor_2"]["center"]
                        )
                    else:
                        angle = None
                    all_image_anchor_data[filename] = {
                        "anchors": anchor_full_data,
                        "skew_angle": angle,
                        "valid_for_option_mapping": True
                    }

                    print(f"Successfully processed {filename}. Anchor data stored.")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                try:
                    error_img = cv2.imread(image_path)
                    if error_img is not None:
                        warning_path = os.path.join(warning_dir, filename)
                        cv2.imwrite(warning_path, error_img)
                    all_image_anchor_data[filename] = {
                        "anchors": {},
                        "M_transform": None,
                        "deskewed_width": None,
                        "deskewed_height": None,
                        "valid_for_option_mapping": False,
                        "error": str(e)
                    }
                except Exception as img_err:
                    print(f"⚠️ Could not save error image: {img_err}")


    # Save all collected anchor data to a single JSON file
    with open(anchor_json_path, 'w') as f:
        json.dump(all_image_anchor_data, f, indent=2)
    print(f"\nAll anchor centers and transformation data saved to {anchor_json_path}")
    csv_headers = ["image_name", "anchor_1", "anchor_2", "anchor_3", "anchor_4"]

    with open(csv_output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_name", "anchor_1", "anchor_2", "anchor_3", "anchor_4", "skew_angle", "Warnings"])

        for img_name, data in all_image_anchor_data.items():
            anchors = data.get("anchors", {})
            row = [img_name]
            for key in ["anchor_1", "anchor_2", "anchor_3", "anchor_4"]:
                if key in anchors:
                    if isinstance(anchors[key], dict) and "center" in anchors[key]:
                        row.append(str(anchors[key]["center"]))
                    else:
                        row.append(str(anchors[key]))
                else:
                    row.append("")

            row.append(data.get("skew_angle", ""))
            row.append("" if data.get("valid_for_option_mapping", True) else "Error")

            writer.writerow(row)

    print(f"Anchor centers also saved to CSV: {csv_output_path}")
    
    # Call this function after everything is processed
    generate_generalized_json(base_folder, omr_template_name, date, folder_path, all_image_anchor_data, warning_dir)