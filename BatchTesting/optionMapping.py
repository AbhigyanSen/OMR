import cv2
import numpy as np
import os
import json

class OptionMapper:
    def __init__(self, image_path, annotations_path, classes_path, anchor_data):
        self.image_path = image_path
        self.annotations_path = annotations_path
        self.classes_path = classes_path
        self.anchor_data = anchor_data # This contains the FINAL, DESKEWED anchor info

        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        self.original_height, self.original_width = self.original_image.shape[:2]
        self.classes = self._load_classes()
        self.annotations = self._load_annotations() # These are original (normalized) annotations from the TXT

        # Retrieve transformation matrix and deskewed dimensions from anchor_data
        self.M_transform = np.array(self.anchor_data.get("M_transform")) if self.anchor_data.get("M_transform") is not None else None
        self.deskewed_width = self.anchor_data.get("deskewed_width", self.original_width)
        self.deskewed_height = self.anchor_data.get("deskewed_height", self.original_height)

        # Apply perspective transform to the original image to get the deskewed image
        if self.M_transform is not None:
            self.deskewed_width = int(self.deskewed_width)
            self.deskewed_height = int(self.deskewed_height)
            self.image = cv2.warpPerspective(self.original_image, self.M_transform, (self.deskewed_width, self.deskewed_height))
        else:
            self.image = self.original_image.copy() # Use original if no transform

        # Initialize mapped_annotations with the *exact* anchor data from anchor_centers.json.
        # This is the crucial part to ensure consistency.
        self.mapped_annotations = {}
        for anchor_name, anchor_info in self.anchor_data.get("anchors", {}).items():
            self.mapped_annotations[anchor_name] = {
                "center": list(anchor_info["center"]), # Ensure it's a list for JSON
                "bbox": list(anchor_info["bbox"]),     # Ensure it's a list for JSON
                "delta_from_Anch1": [None, None] # Placeholder, calculated in map_and_draw
            }


    def _load_classes(self):
        """
        Loads class names from the classes.txt file.
        """
        classes = []
        try:
            with open(self.classes_path, 'r') as f:
                for line in f:
                    classes.append(line.strip())
        except FileNotFoundError:
            print(f"Classes file not found at {self.classes_path}. Cannot load class names.")
            raise # Re-raise to stop if critical file is missing
        return classes

    def _load_annotations(self):
        """
        Loads annotations from the Label Studio .txt file.
        The format is: class_id x_center y_center width height (normalized)
        Returns:
            list: A list of tuples (class_name, norm_x_center, norm_y_center, norm_width, norm_height).
        """
        annotations = []
        try:
            with open(self.annotations_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(self.classes):
                            class_name = self.classes[class_id]
                            # Only load non-anchor annotations from the raw annotation file
                            # Anchors will come directly from anchor_data
                            if "anchor" not in class_name.lower():
                                norm_x_center = float(parts[1])
                                norm_y_center = float(parts[2])
                                norm_width = float(parts[3])
                                norm_height = float(parts[4])
                                annotations.append((class_name, norm_x_center, norm_y_center, norm_width, norm_height))
                        else:
                            print(f"Warning: Class ID {class_id} not found in classes.txt for line: {line.strip()}")
        except FileNotFoundError:
            print(f"Annotation file not found at {self.annotations_path}. No original annotations will be processed.")
        return annotations

    def map_and_draw(self, output_dir):
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]

        # Get anchor_1 center from the *already loaded and stored* mapped_annotations
        # This is the exact value from anchor_centers.json
        anch1_center = self.mapped_annotations.get("anchor_1", {}).get("center", [None, None])
        anch1_x, anch1_y = anch1_center

        if anch1_x is None or anch1_y is None:
            print(f"Error: anchor_1 center not found in provided anchor data for {base_name}. Cannot calculate deltas accurately.")
            # For now, we'll continue, but deltas will be None.
            
        # Ensure 'OMR' and subfolders exist for saving crops
        reg_path = os.path.join(output_dir, "OMR", "reg_no")
        roll_path = os.path.join(output_dir, "OMR", "roll_no")
        book_path = os.path.join(output_dir, "OMR", "booklet_no")
        os.makedirs(reg_path, exist_ok=True)
        os.makedirs(roll_path, exist_ok=True)
        os.makedirs(book_path, exist_ok=True)

        # --- Process and draw ALL annotations (anchors first, then others) ---

        # 1. Process and draw anchors (their data is already loaded and fixed)
        # Calculate their delta_from_Anch1 here
        for anchor_name, anchor_info in self.mapped_annotations.items():
            cx, cy = anchor_info["center"]
            x1, y1, x2, y2 = anchor_info["bbox"]

            # Calculate delta_from_Anch1 for all anchors based on their stored center
            delta_x = cx - anch1_x if anch1_x is not None else None
            delta_y = cy - anch1_y if anch1_y is not None else None
            self.mapped_annotations[anchor_name]["delta_from_Anch1"] = [delta_x, delta_y]

            # Draw bounding box and center point for anchors
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue for anchors
            cv2.circle(self.image, (cx, cy), 5, (0, 0, 255), -1) # Red center
            cv2.putText(self.image, anchor_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 2. Process other annotations (reg_no, roll_no, booklet_no etc.)
        # These will be transformed from their original annotation file coordinates
        for class_name, norm_x_center, norm_y_center, norm_width, norm_height in self.annotations:
            # We already filtered out anchors in _load_annotations, so no need to skip here
            
            # Convert normalized coordinates from original image to pixel coordinates
            original_x_center = norm_x_center * self.original_width
            original_y_center = norm_y_center * self.original_height
            original_width_px = norm_width * self.original_width
            original_height_px = norm_height * self.original_height

            x1_orig = int(original_x_center - original_width_px / 2)
            y1_orig = int(original_y_center - original_height_px / 2)
            x2_orig = int(original_x_center + original_width_px / 2)
            y2_orig = int(original_y_center + original_height_px / 2)

            # Apply perspective transform to get coordinates in the deskewed image
            if self.M_transform is not None:
                pts = np.float32([[x1_orig, y1_orig], [x2_orig, y1_orig], [x2_orig, y2_orig], [x1_orig, y2_orig]]).reshape(-1, 1, 2)
                transformed_pts = cv2.perspectiveTransform(pts, self.M_transform).reshape(-1, 2)
                
                x1 = max(0, int(np.min(transformed_pts[:, 0])))
                y1 = max(0, int(np.min(transformed_pts[:, 1])))
                x2 = min(self.image.shape[1], int(np.max(transformed_pts[:, 0])))
                y2 = min(self.image.shape[0], int(np.max(transformed_pts[:, 1])))
            else:
                x1, y1, x2, y2 = x1_orig, y1_orig, x2_orig, y2_orig

            # Calculate center point for the transformed bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Calculate delta from anchor_1
            delta_x = center_x - anch1_x if anch1_x is not None else None
            delta_y = center_y - anch1_y if anch1_y is not None else None

            # Store mapping result for non-anchor annotations
            self.mapped_annotations[class_name] = {
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y],
                "delta_from_Anch1": [delta_x, delta_y]
            }

            # Draw bounding box and text on the deskewed image for non-anchors
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for non-anchors
            cv2.putText(self.image, class_name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Crop and save specific regions
            crop = self.image[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] <= 0 or crop.shape[1] <= 0:
                print(f"Warning: Empty or invalid crop for {class_name} at bbox [{x1},{y1},{x2},{y2}]. Skipping save.")
                continue

            if class_name == "reg_no":
                cv2.imwrite(os.path.join(reg_path, f"{base_name}.jpg"), crop)
            elif class_name == "roll_no":
                cv2.imwrite(os.path.join(roll_path, f"{base_name}.jpg"), crop)
            elif class_name == "booklet_no":
                cv2.imwrite(os.path.join(book_path, f"{base_name}.jpg"), crop)

        return self.image, self.mapped_annotations


def process_folder(folder_path, annotations_file, classes_file, anchor_data_json_path):
    folder_name = os.path.basename(folder_path.rstrip("\\/"))
    output_dir = f"annotate_{folder_name}"
    warning_dir = os.path.join(output_dir, "warnings")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(warning_dir, exist_ok=True)

    try:
        with open(anchor_data_json_path, "r") as f:
            all_anchor_data = json.load(f)
    except Exception as e:
        print(f"Error loading anchor data from {anchor_data_json_path}: {e}")
        return

    all_mapped_annotations_data = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_specific_anchor_data = all_anchor_data.get(filename)
            
            # Check if anchor data exists and is valid for option mapping
            if (image_specific_anchor_data is None or
                not image_specific_anchor_data.get("valid_for_option_mapping", False)):
                print(f"⛔ Skipping {filename}: invalid or missing anchor data or not marked valid for option mapping.")
                # Save original image to warnings if skipped due to anchor data issue
                try:
                    original_image_for_warning = cv2.imread(os.path.join(folder_path, filename))
                    if original_image_for_warning is not None:
                        cv2.imwrite(os.path.join(warning_dir, filename), original_image_for_warning)
                except Exception as w_err:
                    print(f"⚠️ Failed to write skipped image to warnings: {w_err}")
                
                # Add a record to the output JSON even for skipped images
                all_mapped_annotations_data[filename] = {
                    "error": "Skipped due to invalid/missing anchor data",
                    "valid_for_marked_option": False
                }
                continue

            image_path = os.path.join(folder_path, filename)
            print(f"\nProcessing {image_path}...")

            try:
                mapper = OptionMapper(image_path, annotations_file, classes_file, image_specific_anchor_data)
                mapped_image, mapped_annotations = mapper.map_and_draw(output_dir)
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, mapped_image)
                mapped_annotations["valid_for_marked_option"] = True
                all_mapped_annotations_data[filename] = mapped_annotations
                print(f"✅ Saved mapped image: {save_path}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                try:
                    # Save the original image to warnings in case of an error during processing
                    cv2.imwrite(os.path.join(warning_dir, filename), cv2.imread(image_path))
                except Exception as w:
                    print(f"⚠️ Failed to write warning image for {filename}: {w}")
                all_mapped_annotations_data[filename] = {
                    "error": str(e),
                    "valid_for_marked_option": False
                }

    mapped_annotations_json_path = os.path.join(output_dir, "mapped_annotations.json")
    with open(mapped_annotations_json_path, 'w') as f:
        json.dump(all_mapped_annotations_data, f, indent=2)
    print(f"\n📝 All mapped annotations saved to {mapped_annotations_json_path}")


if __name__ == "__main__":
    base_folder = r"D:\Projects\OMR\new_abhigyan\BatchTesting"
    folder_path = os.path.join(base_folder, "TestData", "BE24-05-06")
    annotations_file = os.path.join(base_folder, "Annotations", "labels", "BE24-05-01001.txt")
    classes_file = os.path.join(base_folder, "Annotations", "classes.txt")
    
    # Corrected path to anchor_centers.json (from anchorDetection.py's output)
    anchor_output_folder_name = "anchor_" + os.path.basename(folder_path.rstrip("\\/"))
    anchor_data_json_path = os.path.join(base_folder, anchor_output_folder_name, "anchor_centers.json")

    process_folder(folder_path, annotations_file, classes_file, anchor_data_json_path)
    
    


# Generate a generalized JSON file for the batch ---------------------------------------------------------
def append_fields_to_batch_json(generalized_json_path, mapped_annotations_path):
    import os
    import json

    with open(generalized_json_path, 'r') as f:
        generalized_data = json.load(f)

    with open(mapped_annotations_path, 'r') as f:
        mapped_annotations = json.load(f)

    # Allow reg/roll/booklet and questions 1–50
    valid_fields = {"roll_no", "booklet_no"}
    valid_fields.update({f"question_{i}" for i in range(1, 51)})

    for image_entry in generalized_data.get("IMAGES", []):
        image_path = image_entry["IMAGENAME"]
        image_filename = os.path.basename(image_path)
        mapped_data = mapped_annotations.get(image_filename, {})

        if not mapped_data.get("valid_for_marked_option", False):
            continue

        image_entry["FIELDS"] = []

        for field_name, field_info in mapped_data.items():
            if field_name in ("valid_for_marked_option", "error"):
                continue
            if field_name.startswith("anchor_"):
                continue
            if field_name not in valid_fields:
                continue

            bbox = field_info.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox

            # Set FIELD name: e.g. "question_5" → "Q5", else keep uppercase
            if field_name.startswith("question_"):
                q_num = field_name.split("_")[1]
                field_key = f"Q{q_num}"
            else:
                field_key = field_name.upper()

            field_metadata = {
                "FIELD": field_key,
                # "FIELDDATA": "",
                # "SUCCESS": "",
                # "CONFIDENCE": "",
                "XCORD": str(x1),
                "YCORD": str(y1),
                "WIDTH": str(x2 - x1),
                "HEIGHT": str(y2 - y1)
            }

            image_entry["FIELDS"].append(field_metadata)

    with open(generalized_json_path, 'w') as f:
        json.dump(generalized_data, f, indent=2)

    print(f"\n✅ Updated {os.path.basename(generalized_json_path)}: only top-level fields, Q1–Q50 naming applied.")

# Paths
# mapped_annotations_path = os.path.join(output_dir, "mapped_annotations.json")
mapped_annotations_path = r"D:\Projects\OMR\new_abhigyan\BatchTesting\annotate_BE24-05-06\mapped_annotations.json"
# generalized_json_path = os.path.join(base_folder, template_name, batch_name, f"{batch_name}.json")
generalized_json_path = r"D:\Projects\OMR\new_abhigyan\BatchTesting\TestData\BE24-05-06\BE24-05-06.json"

# Call the function
append_fields_to_batch_json(generalized_json_path, mapped_annotations_path)

