import cv2
import numpy as np
import math
import os
import json
from collections import defaultdict

class OMRFieldMapper:
    def __init__(self, reference_image_path, reference_annotations_path, classes_path, anchor_centers_json_path, target_width, target_height):
        self.reference_image_path = reference_image_path
        self.reference_annotations_path = reference_annotations_path
        self.classes_path = classes_path
        self.anchor_centers_json_path = anchor_centers_json_path
        self.target_width = target_width
        self.target_height = target_height

        self.classes = self._load_classes()
        self.reference_annotations = self._load_annotations(self.reference_annotations_path, self.target_width, self.target_height)
        self.all_image_anchor_data = self._load_anchor_centers()
        self.relative_offsets = self._calculate_relative_offsets()

        if not self.relative_offsets:
            print("❌ Critical Error: Could not calculate relative offsets from reference image. Check annotations and anchor_1 presence.")

    def _load_classes(self):
        classes = []
        try:
            with open(self.classes_path, 'r') as f:
                for line in f:
                    classes.append(line.strip().replace('\r', ''))
        except FileNotFoundError:
            print(f"❌ Classes file not found at {self.classes_path}. Cannot proceed.")
            exit() # Exit if classes are not found, as it's critical
        return classes

    def _load_annotations(self, annotations_path, width, height):
        annotations = defaultdict(list)
        try:
            with open(annotations_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        norm_width = float(parts[3]) * width
                        norm_height = float(parts[4]) * height
                        
                        x1 = int(x_center - norm_width / 2)
                        y1 = int(y_center - norm_height / 2)
                        x2 = int(x_center + norm_width / 2)
                        y2 = int(y_center + norm_height / 2)
                        
                        annotations[class_id].append((x1, y1, x2, y2))
        except FileNotFoundError:
            print(f"❌ Annotation file not found at {annotations_path}.")
        return annotations

    def _load_anchor_centers(self):
        try:
            with open(self.anchor_centers_json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Anchor centers JSON not found at {self.anchor_centers_json_path}. Cannot proceed.")
            exit() # Exit if anchor data is not found, as it's critical
        except json.JSONDecodeError:
            print(f"❌ Error decoding JSON from {self.anchor_centers_json_path}. Check file format.")
            exit()
        return {}

    def _get_class_id(self, class_name):
        try:
            return self.classes.index(class_name)
        except ValueError:
            return -1

    def _calculate_relative_offsets(self):
        relative_offsets = {}
        anchor_1_class_id = self._get_class_id("anchor_1")

        if anchor_1_class_id == -1 or not self.reference_annotations.get(anchor_1_class_id):
            print("❌ Reference anchor_1 not found in annotations. Cannot calculate relative offsets.")
            return {}

        # Assuming there's only one anchor_1 on the reference sheet
        ref_anchor_1_bbox = self.reference_annotations[anchor_1_class_id][0]
        ref_anchor_1_x1, ref_anchor_1_y1, _, _ = ref_anchor_1_bbox

        for class_id, bboxes in self.reference_annotations.items():
            class_name = self.classes[class_id]
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1

                # Calculate offset from top-left of anchor_1
                dx = x1 - ref_anchor_1_x1
                dy = y1 - ref_anchor_1_y1

                unique_key = f"{class_name}_{i}" 
                relative_offsets[unique_key] = {
                    "dx": dx,
                    "dy": dy,
                    "width": width,
                    "height": height
                }
        print(f"✅ Calculated {len(relative_offsets)} relative offsets from reference anchor_1.")
        return relative_offsets

    def _order_points(self, points):
        rect = np.zeros((4, 2), dtype="float32")

        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]

        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        return rect

    def deskew_image(self, image, detected_anchors_data):
        anchor_coords = []
        # Extract center coordinates of the 4 anchors
        for i in range(1, 5):
            anchor_name = f'anchor_{i}'
            if anchor_name in detected_anchors_data and 'center' in detected_anchors_data[anchor_name]:
                anchor_coords.append(detected_anchors_data[anchor_name]['center'])
            else:
                print(f"⚠️ Missing {anchor_name} for deskewing. Cannot perform perspective transform.")
                return image, None, self.target_width, self.target_height # Return original image if anchors are missing

        if len(anchor_coords) != 4:
            print(f"⚠️ Only {len(anchor_coords)} anchors found. Need 4 for perspective deskew. Attempting rotational deskew if anchor_1 and anchor_2 exist.")
            
            # Fallback to rotational deskew if only 2 anchors are available
            anchor1 = detected_anchors_data.get('anchor_1')
            anchor2 = detected_anchors_data.get('anchor_2')
            if anchor1 and anchor2 and 'center' in anchor1 and 'center' in anchor2:
                angle = self._compute_skew_angle(anchor1['center'], anchor2['center'])
                if abs(angle) > 0.1: # Only rotate if angle is significant
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated_image = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    print(f"Image rotated by {angle:.2f} degrees.")
                    return rotated_image, M_rot, w, h
            return image, None, self.target_width, self.target_height # Return original if no suitable deskew

        src_points = np.array(anchor_coords, dtype="float32")
        ordered_src_points = self._order_points(src_points)

        ref_anchor_1_id = self._get_class_id("anchor_1")
        ref_anchor_2_id = self._get_class_id("anchor_2")
        ref_anchor_3_id = self._get_class_id("anchor_3")
        ref_anchor_4_id = self._get_class_id("anchor_4")

        if not (self.reference_annotations.get(ref_anchor_1_id) and 
                self.reference_annotations.get(ref_anchor_2_id) and
                self.reference_annotations.get(ref_anchor_3_id) and
                self.reference_annotations.get(ref_anchor_4_id)):
            print("❌ Reference annotations for all 4 anchors missing. Cannot define ideal destination points for deskew.")
            return image, None, self.target_width, self.target_height

        ref_anchors_bboxes = [
            self.reference_annotations[ref_anchor_1_id][0],
            self.reference_annotations[ref_anchor_2_id][0],
            self.reference_annotations[ref_anchor_3_id][0],
            self.reference_annotations[ref_anchor_4_id][0]
        ]
        
        # Use center points of reference anchors for ideal destination
        ref_centers = []
        for bbox in ref_anchors_bboxes:
            x1, y1, x2, y2 = bbox
            ref_centers.append([(x1 + x2) // 2, (y1 + y2) // 2])

        ordered_ref_points = self._order_points(np.array(ref_centers, dtype="float32"))

        # Calculate the dimensions of the deskewed image based on reference anchor positions
        (tl, tr, br, bl) = ordered_ref_points
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst_points = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        dst_points_ordered = np.array([
            [0, 0],                            # TL
            [maxWidth - 1, 0],                 # TR
            [maxWidth - 1, maxHeight - 1],     # BR
            [0, maxHeight - 1]                 # BL
        ], dtype="float32")


        M = cv2.getPerspectiveTransform(ordered_src_points, dst_points_ordered)
        deskewed_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        print(f"Image deskewed to {maxWidth}x{maxHeight} using perspective transform.")
        return deskewed_image, M, maxWidth, maxHeight

    def _compute_skew_angle(self, anchor1_center, anchor2_center):
        x1, y1 = anchor1_center
        x2, y2 = anchor2_center

        base = x2 - x1 # Use x2-x1 directly for base to maintain sign for angle
        height = y2 - y1 

        if base == 0:
            return 90.0 if height > 0 else -90.0

        angle_rad = math.atan2(height, base)
        angle_deg = math.degrees(angle_rad)

        return round(angle_deg, 4)

    def map_fields_and_visualize(self, image_path, output_dir, missing_fields_dir):
        filename = os.path.basename(image_path)
        image_data = self.all_image_anchor_data.get(filename)

        if not image_data:
            print(f"⚠️ No anchor data found for {filename}. Skipping field mapping.")
            return {
                "status": "skipped_no_anchor_data",
                "mapped_fields": {},
                "missing_fields": list(self.relative_offsets.keys()) # All fields are missing if no anchor data
            }

        if not image_data.get("valid_for_option_mapping", False):
            print(f"⚠️ Image {filename} flagged as invalid for option mapping. Skipping field mapping.")
            return {
                "status": "skipped_invalid_for_option_mapping",
                "mapped_fields": {},
                "missing_fields": list(self.relative_offsets.keys())
            }

        print(f"Processing {filename} for field mapping...")
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"❌ Could not read image: {image_path}. Skipping.")
            return {
                "status": "skipped_image_read_error",
                "mapped_fields": {},
                "missing_fields": list(self.relative_offsets.keys())
            }
        
        # Ensure image is at target resolution before deskewing
        if original_image.shape[1] != self.target_width or original_image.shape[0] != self.target_height:
            original_image = cv2.resize(original_image, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
            print(f"Resized {filename} to {self.target_width}x{self.target_height} for consistency.")

        detected_anchors_for_image = image_data.get("anchors", {})
        
        # Perform deskewing
        deskewed_image, M_transform, deskewed_width, deskewed_height = self.deskew_image(original_image.copy(), detected_anchors_for_image)
        
        mapped_anchor_1_x1, mapped_anchor_1_y1 = None, None
        anchor_1_class_id = self._get_class_id("anchor_1")
        if anchor_1_class_id != -1 and self.reference_annotations.get(anchor_1_class_id):
            ref_anchor_1_bbox = self.reference_annotations[anchor_1_class_id][0]
            ref_anchor_1_x1, ref_anchor_1_y1, _, _ = ref_anchor_1_bbox
            
            mapped_anchor_1_x1 = ref_anchor_1_x1
            mapped_anchor_1_y1 = ref_anchor_1_y1
        
        if mapped_anchor_1_x1 is None:
            print(f"❌ Could not determine reference anchor_1 position for {filename}. Skipping field mapping.")
            return {
                "status": "skipped_ref_anchor_1_missing",
                "mapped_fields": {},
                "missing_fields": list(self.relative_offsets.keys())
            }

        mapped_fields_data = {}
        missing_fields = []
        display_image = deskewed_image.copy() # Draw on the deskewed image

        for unique_key, offset_data in self.relative_offsets.items():
            class_name = "_".join(unique_key.split('_')[:-1]) # Reconstruct class name (e.g., 'question_1', '1A')
            
            x1_mapped = mapped_anchor_1_x1 + offset_data["dx"]
            y1_mapped = mapped_anchor_1_y1 + offset_data["dy"]
            x2_mapped = x1_mapped + offset_data["width"]
            y2_mapped = y1_mapped + offset_data["height"]

            x1_mapped = max(0, x1_mapped)
            y1_mapped = max(0, y1_mapped)
            x2_mapped = min(deskewed_width, x2_mapped)
            y2_mapped = min(deskewed_height, y2_mapped)

            if x1_mapped >= x2_mapped or y1_mapped >= y2_mapped:
                print(f"⚠️ Mapped bbox for {unique_key} is invalid [{x1_mapped},{y1_mapped},{x2_mapped},{y2_mapped}]. Marking as missing.")
                missing_fields.append(unique_key)
                continue

            mapped_fields_data[unique_key] = {
                "bbox": [int(x1_mapped), int(y1_mapped), int(x2_mapped), int(y2_mapped)],
                "width": int(x2_mapped - x1_mapped),
                "height": int(y2_mapped - y1_mapped)
            }
            
            # Visualize
            color = (0, 255, 0) # Green for mapped fields
            cv2.rectangle(display_image, (int(x1_mapped), int(y1_mapped)), (int(x2_mapped), int(y2_mapped)), color, 2)
            cv2.putText(display_image, class_name, (int(x1_mapped), int(y1_mapped) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save visualized image
        output_image_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_mapped.jpg")
        cv2.imwrite(output_image_path, display_image)
        print(f"✅ Mapped image saved to {output_image_path}")

        # Save missing fields to a text file
        if missing_fields:
            missing_file_path = os.path.join(missing_fields_dir, f"{os.path.splitext(filename)[0]}_missing_fields.txt")
            with open(missing_file_path, 'w') as f:
                for field in missing_fields:
                    f.write(f"{field}\n")
            print(f"⚠️ Missing fields for {filename} saved to {missing_file_path}")

        return {
            "status": "processed",
            "mapped_fields": mapped_fields_data,
            "missing_fields": missing_fields,
            "deskewed_dimensions": [deskewed_width, deskewed_height] if M_transform is not None else [self.target_width, self.target_height]
        }


# Main execution
if __name__ == "__main__":
    # Define paths - Adjust these to your actual directory structure
    base_folder = r"D:\Projects\OMR\new_abhigyan\Declutter" # Your base project folder
        
    # Paths from previous phase's output and original annotations
    # The 'folder_path' used in the previous script to process images
    batch_name = "BE24-05-02" 
    processed_images_folder = os.path.join(base_folder, "TestData", batch_name, f"processed_{batch_name}") # Images processed by previous script
    
    # Original annotation reference files
    reference_image_path = os.path.join(base_folder, "Annotations", "images", "BE24-05-01001.jpg")
    reference_annotations_path = os.path.join(base_folder, "Annotations", "labels", "BE24-05-01001.txt")
    classes_file = os.path.join(base_folder, "Annotations", "classes.txt")

    # Output from the previous anchor detection script
    anchor_output_folder_name = "anchor_" + batch_name
    anchor_centers_json_path = os.path.join(anchor_output_folder_name, "anchor_centers.json")

    # Determine reference dimensions from the reference image
    ref_img = cv2.imread(reference_image_path)
    if ref_img is None:
        raise FileNotFoundError(f"Reference annotated image not found at {reference_image_path}. This is critical.")
    ref_height, ref_width = ref_img.shape[:2]
    print(f"📏 Reference dimensions from annotation image: {ref_width}x{ref_height}")

    # Create output directories for this phase
    mapping_output_folder_name = "mapping_" + batch_name
    output_images_dir = os.path.join(mapping_output_folder_name, "mapped_images")
    missing_fields_log_dir = os.path.join(mapping_output_folder_name, "missing_fields_logs")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(missing_fields_log_dir, exist_ok=True)

    # Initialize the mapper
    mapper = OMRFieldMapper(
        reference_image_path=reference_image_path,
        reference_annotations_path=reference_annotations_path,
        classes_path=classes_file,
        anchor_centers_json_path=anchor_centers_json_path,
        target_width=ref_width,
        target_height=ref_height
    )

    all_image_field_data = {}

    # Process each image in the processed test data folder
    for filename in os.listdir(processed_images_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(processed_images_folder, filename)
            print(f"\n--- Starting field mapping for {filename} ---")
            
            field_data = mapper.map_fields_and_visualize(
                image_path=image_path,
                output_dir=output_images_dir,
                missing_fields_dir=missing_fields_log_dir
            )
            all_image_field_data[filename] = field_data
            print(f"--- Finished field mapping for {filename} ---")

    # Save all collected field data to a single JSON file
    json_output_path = os.path.join(mapping_output_folder_name, "field_mappings.json")
    with open(json_output_path, 'w') as f:
        json.dump(all_image_field_data, f, indent=2)
    print(f"\nAll field mappings saved to {json_output_path}")