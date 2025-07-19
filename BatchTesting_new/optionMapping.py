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
            self.image = cv2.warpPerspective(self.original_image, self.M_transform, (self.original_width, self.original_height))
            self.deskewed_height, self.deskewed_width = self.image.shape[:2]  # Recalculate from warped image
            print(f"✅ Applied transform: new size = {self.deskewed_width}x{self.deskewed_height}")
        else:
            print(f"⚠️ No M_transform found for {os.path.basename(self.image_path)} — using original image.")
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
        anch1_center = self.mapped_annotations.get("anchor_1", {}).get("center", [None, None])
        anch1_x, anch1_y = anch1_center

        reg_path = os.path.join(output_dir, "OMR", "reg_no")
        roll_path = os.path.join(output_dir, "OMR", "roll_no")
        book_path = os.path.join(output_dir, "OMR", "booklet_no")
        os.makedirs(reg_path, exist_ok=True)
        os.makedirs(roll_path, exist_ok=True)
        os.makedirs(book_path, exist_ok=True)

        # 1. Draw Anchors
        for anchor_name, anchor_info in self.mapped_annotations.items():
            cx, cy = anchor_info["center"]
            x1, y1, x2, y2 = anchor_info["bbox"]

            # Transform center and bbox using M_transform
            if self.M_transform is not None:
                center_pt = np.array([[[cx, cy]]], dtype=np.float32)
                cx, cy = cv2.perspectiveTransform(center_pt, self.M_transform)[0][0].astype(int)

                bbox_pts = np.float32([[x1, y1], [x2, y2]]).reshape(-1, 1, 2)
                transformed_bbox = cv2.perspectiveTransform(bbox_pts, self.M_transform).reshape(-1, 2)
                x1, y1 = transformed_bbox[0].astype(int)
                x2, y2 = transformed_bbox[1].astype(int)

            # Now draw
            delta_x = cx - anch1_x if anch1_x is not None else None
            delta_y = cy - anch1_y if anch1_y is not None else None
            self.mapped_annotations[anchor_name]["delta_from_Anch1"] = [delta_x, delta_y]

            cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(self.image, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(self.image, anchor_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(self.image, f"({cx},{cy})", (cx + 5, cy + 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # 2. Draw Other Fields (reg_no, roll_no, etc.)
        for class_name, norm_x_center, norm_y_center, norm_width, norm_height in self.annotations:
            original_x_center = norm_x_center * self.original_width
            original_y_center = norm_y_center * self.original_height
            original_width_px = norm_width * self.original_width
            original_height_px = norm_height * self.original_height

            x1_orig = int(original_x_center - original_width_px / 2)
            y1_orig = int(original_y_center - original_height_px / 2)
            x2_orig = int(original_x_center + original_width_px / 2)
            y2_orig = int(original_y_center + original_height_px / 2)

            if self.M_transform is not None:
                pts = np.float32([[x1_orig, y1_orig], [x2_orig, y1_orig], [x2_orig, y2_orig], [x1_orig, y2_orig]]).reshape(-1, 1, 2)
                transformed_pts = cv2.perspectiveTransform(pts, self.M_transform).reshape(-1, 2)
                x1 = max(0, int(np.min(transformed_pts[:, 0])))
                y1 = max(0, int(np.min(transformed_pts[:, 1])))
                x2 = min(self.image.shape[1], int(np.max(transformed_pts[:, 0])))
                y2 = min(self.image.shape[0], int(np.max(transformed_pts[:, 1])))
            else:
                x1, y1, x2, y2 = x1_orig, y1_orig, x2_orig, y2_orig

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            delta_x = center_x - anch1_x if anch1_x is not None else None
            delta_y = center_y - anch1_y if anch1_y is not None else None

            self.mapped_annotations[class_name] = {
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y],
                "delta_from_Anch1": [delta_x, delta_y]
            }
            
            # DEBUG VISUALIZATION for bounding boxes
            for pt in transformed_pts:
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(self.image, (px, py), 3, (255, 0, 0), -1)  # yellow corners


            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
            cv2.putText(self.image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            crop = self.image[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"⚠️ Invalid crop for {class_name} at [{x1},{y1},{x2},{y2}]")
                continue

            if class_name == "reg_no":
                cv2.imwrite(os.path.join(reg_path, f"{base_name}.jpg"), crop)
            elif class_name == "roll_no":
                cv2.imwrite(os.path.join(roll_path, f"{base_name}.jpg"), crop)
            elif class_name == "booklet_no":
                cv2.imwrite(os.path.join(book_path, f"{base_name}.jpg"), crop)

        return self.image, self.mapped_annotations
def convert_numpy_to_python(obj):
        """
        Recursively convert numpy data types to native Python types (int, float).
        """
        if isinstance(obj, dict):
            return {k: convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_python(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
        
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
        json.dump(convert_numpy_to_python(all_mapped_annotations_data), f, indent=2)
    print(f"\n📝 All mapped annotations saved to {mapped_annotations_json_path}")


if __name__ == "__main__":
    base_folder = r"D:\Projects\OMR\new_abhigyan\BatchTesting"
    folder_path = os.path.join(base_folder, "TestData", "BE24-05-05")
    annotations_file = os.path.join(base_folder, "Annotations", "labels", "BE24-05-01001.txt")
    classes_file = os.path.join(base_folder, "Annotations", "classes.txt")
    
    # Corrected path to anchor_centers.json (from anchorDetection.py's output)
    anchor_output_folder_name = "anchor_" + os.path.basename(folder_path.rstrip("\\/"))
    anchor_data_json_path = os.path.join(base_folder, anchor_output_folder_name, "anchor_centers.json")
    print(f"PAth to Anchor: {anchor_data_json_path}")

    process_folder(folder_path, annotations_file, classes_file, anchor_data_json_path)