import cv2
import numpy as np
import math
import os
import json
from collections import defaultdict

class OMRFieldMapper:
    def __init__(self, reference_image_path, reference_annotations_path, classes_path, 
                 anchor_centers_json_path, target_width, target_height):
        """
        Initializes the OMRFieldMapper.

        Args:
            reference_image_path (str): Path to the reference annotated image (e.g., BE24-05-01001.jpg).
            reference_annotations_path (str): Path to the reference YOLO annotation file (e.g., BE24-05-01001.txt).
            classes_path (str): Path to the classes.txt file.
            anchor_centers_json_path (str): Path to the JSON file containing detected anchor centers for all images.
            target_width (int): The target width to which all images were resized.
            target_height (int): The target height to which all images were resized.
        """
        self.reference_image_path = reference_image_path
        self.reference_annotations_path = reference_annotations_path
        self.classes_path = classes_path
        self.anchor_centers_json_path = anchor_centers_json_path
        self.target_width = target_width
        self.target_height = target_height

        self.classes = self._load_classes()
        self.reference_annotations = self._load_annotations(self.reference_annotations_path, 
                                                            self.target_width, self.target_height)
        self.all_image_anchor_data = self._load_anchor_centers()
        # Relative offsets are now calculated based on normalized distances
        self.relative_offsets = self._calculate_relative_offsets()

        if not self.relative_offsets:
            print("❌ Critical Error: Could not calculate relative offsets from reference image. Check annotations and anchor_1 presence.")
            exit() # Exit if critical data is missing

    def _load_classes(self):
        """Loads class names from classes.txt."""
        classes = []
        try:
            with open(self.classes_path, 'r') as f:
                for line in f:
                    classes.append(line.strip().replace('\r', ''))
        except FileNotFoundError:
            print(f"❌ Classes file not found at {self.classes_path}. Cannot proceed.")
            exit() 
        return classes

    def _load_annotations(self, annotations_path, width, height):
        """
        Loads YOLO annotations and converts them to (x1, y1, x2, y2) pixel coordinates.
        
        Args:
            annotations_path (str): Path to the YOLO annotation file.
            width (int): Original width of the image for denormalization.
            height (int): Original height of the image for denormalization.
        
        Returns:
            dict: A dictionary where keys are class IDs and values are lists of (x1, y1, x2, y2) bounding boxes.
        """
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
        """Loads anchor center data from the JSON file."""
        try:
            with open(self.anchor_centers_json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Anchor centers JSON not found at {self.anchor_centers_json_path}. Cannot proceed.")
            exit() 
        except json.JSONDecodeError:
            print(f"❌ Error decoding JSON from {self.anchor_centers_json_path}. Check file format.")
            exit()
        return {}

    def _get_class_id(self, class_name):
        """Returns the class ID for a given class name."""
        try:
            return self.classes.index(class_name)
        except ValueError:
            return -1

    def _calculate_relative_offsets(self):
        """
        Calculates the relative (dx, dy, dw, dh) offsets of all fields
        from anchor_1's center on the reference image, normalized by anchor distances.
        """
        relative_offsets = {}
        anchor_1_class_id = self._get_class_id("anchor_1")
        anchor_2_class_id = self._get_class_id("anchor_2")
        anchor_3_class_id = self._get_class_id("anchor_3")

        # Ensure reference anchors exist for calculating scale factors
        if not (self.reference_annotations.get(anchor_1_class_id) and 
                self.reference_annotations.get(anchor_2_class_id) and
                self.reference_annotations.get(anchor_3_class_id)):
            print("❌ Reference anchor_1, anchor_2, or anchor_3 not found in annotations. Cannot calculate relative offsets with scaling.")
            return {}

        # Get reference anchor centers
        ref_anchor_1_bbox = self.reference_annotations[anchor_1_class_id][0]
        ref_anchor_1_center_x = (ref_anchor_1_bbox[0] + ref_anchor_1_bbox[2]) // 2
        ref_anchor_1_center_y = (ref_anchor_1_bbox[1] + ref_anchor_1_bbox[3]) // 2

        ref_anchor_2_bbox = self.reference_annotations[anchor_2_class_id][0]
        ref_anchor_2_center_x = (ref_anchor_2_bbox[0] + ref_anchor_2_bbox[2]) // 2
        ref_anchor_2_center_y = (ref_anchor_2_bbox[1] + ref_anchor_2_bbox[3]) // 2

        ref_anchor_3_bbox = self.reference_annotations[anchor_3_class_id][0]
        ref_anchor_3_center_x = (ref_anchor_3_bbox[0] + ref_anchor_3_bbox[2]) // 2
        ref_anchor_3_center_y = (ref_anchor_3_bbox[1] + ref_anchor_3_bbox[3]) // 2

        # Calculate reference distances for normalization
        # Using horizontal distance between anchor_1 and anchor_2
        ref_horizontal_dist = math.sqrt((ref_anchor_2_center_x - ref_anchor_1_center_x)**2 + 
                                        (ref_anchor_2_center_y - ref_anchor_1_center_y)**2)
        # Using vertical distance between anchor_1 and anchor_3
        ref_vertical_dist = math.sqrt((ref_anchor_3_center_x - ref_anchor_1_center_x)**2 + 
                                      (ref_anchor_3_center_y - ref_anchor_1_center_y)**2)

        if ref_horizontal_dist == 0 or ref_vertical_dist == 0:
            print("❌ Reference anchor distances are zero. Check annotations. Cannot calculate normalized offsets.")
            return {}

        for class_id, bboxes in self.reference_annotations.items():
            class_name = self.classes[class_id]
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1

                # Calculate offset from center of reference anchor_1
                dx = x1 - ref_anchor_1_center_x
                dy = y1 - ref_anchor_1_center_y

                # Normalize offsets and dimensions by reference distances
                norm_dx = dx / ref_horizontal_dist
                norm_dy = dy / ref_vertical_dist
                norm_width = width / ref_horizontal_dist
                norm_height = height / ref_vertical_dist

                # Create a unique key for each annotation instance
                unique_key = f"{class_name}_{i}" 
                relative_offsets[unique_key] = {
                    "norm_dx": norm_dx,
                    "norm_dy": norm_dy,
                    "norm_width": norm_width,
                    "norm_height": norm_height
                }
        print(f"✅ Calculated {len(relative_offsets)} normalized relative offsets from reference anchor_1's center.")
        return relative_offsets

    def _order_points(self, points):
        """
        Orders a list of 4 points (x, y) as top-left, top-right, bottom-right, bottom-left.
        Assumes points are corners of a rectangle/quadrilateral.
        """
        rect = np.zeros((4, 2), dtype="float32")

        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)] # Top-left
        rect[2] = points[np.argmax(s)] # Bottom-right

        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)] # Top-right
        rect[3] = points[np.argmax(diff)] # Bottom-left

        return rect

    def deskew_image(self, image, detected_anchors_data):
        """
        Applies perspective transform deskewing to an image using 4 detected anchors.
        The image is transformed so its anchors align with the reference image's anchors.
        
        Args:
            image (np.array): The input image (already resized to target_width/height).
            detected_anchors_data (dict): Dictionary of detected anchors for the current image.
                                          Expected keys: 'anchor_1', 'anchor_2', 'anchor_3', 'anchor_4'
                                          with 'center' data.
        
        Returns:
            tuple: (deskewed_image, M_transform, deskewed_width, deskewed_height)
                   M_transform is None if deskewing couldn't be performed.
        """
        anchor_coords = []
        # Extract center coordinates of the 4 anchors from the current image
        for i in range(1, 5):
            anchor_name = f'anchor_{i}'
            if anchor_name in detected_anchors_data and 'center' in detected_anchors_data[anchor_name]:
                anchor_coords.append(detected_anchors_data[anchor_name]['center'])
            else:
                print(f"⚠️ Missing {anchor_name} for deskewing. Cannot perform perspective transform.")
                # Fallback to rotational deskew if not all 4 anchors are found
                return self._fallback_rotational_deskew(image, detected_anchors_data)

        if len(anchor_coords) != 4:
            # This case should ideally be caught by the loop above, but as a safeguard:
            print(f"⚠️ Only {len(anchor_coords)} anchors found. Need 4 for perspective deskew. Attempting rotational deskew.")
            return self._fallback_rotational_deskew(image, detected_anchors_data)

        src_points = np.array(anchor_coords, dtype="float32") # Current image's detected anchor centers
        ordered_src_points = self._order_points(src_points)

        # Get the reference image's anchor centers (these are our ideal destination points)
        ref_anchor_ids = [self._get_class_id(f"anchor_{i}") for i in range(1, 5)]
        
        # Ensure all reference anchors exist
        if not all(self.reference_annotations.get(aid) for aid in ref_anchor_ids):
            print("❌ Reference annotations for all 4 anchors missing. Cannot define ideal destination points for perspective deskew.")
            return self._fallback_rotational_deskew(image, detected_anchors_data)

        # Extract center points of reference anchors to form destination points
        ref_centers = []
        for aid in ref_anchor_ids:
            bbox = self.reference_annotations[aid][0] # Assuming one bbox per anchor class
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            ref_centers.append([center_x, center_y])
        
        dst_points = self._order_points(np.array(ref_centers, dtype="float32")) # These are the ideal positions on the target canvas

        # The dimensions of the deskewed image should be the same as the reference image
        # because we are mapping to the reference image's coordinate system.
        deskewed_width = self.target_width
        deskewed_height = self.target_height
        
        M = cv2.getPerspectiveTransform(ordered_src_points, dst_points)
        deskewed_image = cv2.warpPerspective(image, M, (deskewed_width, deskewed_height))
        
        print(f"✅ Image deskewed to {deskewed_width}x{deskewed_height} using perspective transform.")
        return deskewed_image, M, deskewed_width, deskewed_height

    def _fallback_rotational_deskew(self, image, detected_anchors_data):
        """
        Performs a simple rotational deskew if 4 anchors are not available for perspective transform.
        """
        anchor1 = detected_anchors_data.get('anchor_1')
        anchor2 = detected_anchors_data.get('anchor_2')
        if anchor1 and anchor2 and 'center' in anchor1 and 'center' in anchor2:
            angle = self._compute_skew_angle(anchor1['center'], anchor2['center'])
            if abs(angle) > 0.1: # Only rotate if angle is significant
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_image = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                print(f"⚠️ Image rotated by {angle:.2f} degrees (rotational deskew fallback).")
                return rotated_image, M_rot, w, h
        print("⚠️ No suitable deskewing performed. Returning original image.")
        return image, None, self.target_width, self.target_height # Return original if no suitable deskew

    def _compute_skew_angle(self, anchor1_center, anchor2_center):
        """Computes the skew angle between two anchor points."""
        x1, y1 = anchor1_center
        x2, y2 = anchor2_center

        base = x2 - x1 
        height = y2 - y1 

        if base == 0:
            return 90.0 if height > 0 else -90.0

        angle_rad = math.atan2(height, base)
        angle_deg = math.degrees(angle_rad)

        return round(angle_deg, 4)

    def map_fields_and_visualize(self, image_path, output_dir, missing_fields_dir):
        """
        Maps fields onto the given image, visualizes them, and stores data.
        
        Args:
            image_path (str): Path to the image to process.
            output_dir (str): Directory to save mapped images.
            missing_fields_dir (str): Directory to save missing fields log files.
        
        Returns:
            dict: Dictionary containing mapped field data for the image.
        """
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
        
        # Perform deskewing. The deskewed_image will now be aligned to the reference coordinate system.
        deskewed_image, M_transform, deskewed_width, deskewed_height = self.deskew_image(original_image.copy(), detected_anchors_for_image)
        
        # Get the center of anchor_1 from the REFERENCE annotations.
        # Since the image is now deskewed to align with the reference,
        # this is the correct base point for applying relative offsets.
        anchor_1_class_id = self._get_class_id("anchor_1")
        anchor_2_class_id = self._get_class_id("anchor_2")
        anchor_3_class_id = self._get_class_id("anchor_3")

        if not (self.reference_annotations.get(anchor_1_class_id) and 
                self.reference_annotations.get(anchor_2_class_id) and
                self.reference_annotations.get(anchor_3_class_id)):
            print(f"❌ Reference anchor_1, anchor_2, or anchor_3 not found in annotations. Cannot map fields for {filename}.")
            return {
                "status": "skipped_ref_anchor_data_missing",
                "mapped_fields": {},
                "missing_fields": list(self.relative_offsets.keys())
            }

        # Get reference anchor centers for applying offsets
        ref_anchor_1_bbox = self.reference_annotations[anchor_1_class_id][0]
        ref_anchor_1_center_x = (ref_anchor_1_bbox[0] + ref_anchor_1_bbox[2]) // 2
        ref_anchor_1_center_y = (ref_anchor_1_bbox[1] + ref_anchor_1_bbox[3]) // 2

        ref_anchor_2_bbox = self.reference_annotations[anchor_2_class_id][0]
        ref_anchor_2_center_x = (ref_anchor_2_bbox[0] + ref_anchor_2_bbox[2]) // 2
        ref_anchor_2_center_y = (ref_anchor_2_bbox[1] + ref_anchor_2_bbox[3]) // 2

        ref_anchor_3_bbox = self.reference_annotations[anchor_3_class_id][0]
        ref_anchor_3_center_x = (ref_anchor_3_bbox[0] + ref_anchor_3_bbox[2]) // 2
        ref_anchor_3_center_y = (ref_anchor_3_bbox[1] + ref_anchor_3_bbox[3]) // 2

        # Calculate current scale factors from the deskewed image's reference anchors
        # These will be the same as the reference image's distances because the image is deskewed to match.
        current_horizontal_dist = math.sqrt((ref_anchor_2_center_x - ref_anchor_1_center_x)**2 + 
                                            (ref_anchor_2_center_y - ref_anchor_1_center_y)**2)
        current_vertical_dist = math.sqrt((ref_anchor_3_center_x - ref_anchor_1_center_x)**2 + 
                                          (ref_anchor_3_center_y - ref_anchor_1_center_y)**2)

        if current_horizontal_dist == 0 or current_vertical_dist == 0:
            print(f"❌ Current anchor distances are zero for {filename}. Cannot map fields accurately.")
            return {
                "status": "skipped_zero_current_anchor_dist",
                "mapped_fields": {},
                "missing_fields": list(self.relative_offsets.keys())
            }

        mapped_fields_data = {}
        missing_fields = []
        display_image = deskewed_image.copy() # Draw on the deskewed image

        for unique_key, offset_data in self.relative_offsets.items():
            # Reconstruct class name (e.g., 'question_1', '1A') from unique_key (e.g., 'question_1_0')
            class_name_parts = unique_key.split('_')
            if len(class_name_parts) > 1 and class_name_parts[-1].isdigit():
                class_name = "_".join(class_name_parts[:-1])
            else:
                class_name = unique_key # Fallback if not in expected format

            # Calculate new bbox coordinates on the deskewed image
            # based on the reference anchor_1's *center* position and the scaled relative offsets.
            x1_mapped = ref_anchor_1_center_x + (offset_data["norm_dx"] * current_horizontal_dist)
            y1_mapped = ref_anchor_1_center_y + (offset_data["norm_dy"] * current_vertical_dist)
            width_mapped = offset_data["norm_width"] * current_horizontal_dist
            height_mapped = offset_data["norm_height"] * current_vertical_dist
            x2_mapped = x1_mapped + width_mapped
            y2_mapped = y1_mapped + height_mapped

            # Ensure coordinates are within image bounds
            x1_mapped = max(0, int(x1_mapped))
            y1_mapped = max(0, int(y1_mapped))
            x2_mapped = min(deskewed_width, int(x2_mapped))
            y2_mapped = min(deskewed_height, int(y2_mapped))

            if x1_mapped >= x2_mapped or y1_mapped >= y2_mapped:
                print(f"⚠️ Mapped bbox for {unique_key} is invalid [{x1_mapped},{y1_mapped},{x2_mapped},{y2_mapped}]. Marking as missing.")
                missing_fields.append(unique_key)
                continue

            mapped_fields_data[unique_key] = {
                "bbox": [x1_mapped, y1_mapped, x2_mapped, y2_mapped],
                "width": x2_mapped - x1_mapped,
                "height": y2_mapped - y1_mapped
            }
            
            # Visualize
            color = (34, 139, 34) # Green for mapped fields
            cv2.rectangle(display_image, (x1_mapped, y1_mapped), (x2_mapped, y2_mapped), color, 1)
            # cv2.putText(display_image, class_name, (x1_mapped, y1_mapped - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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
            "deskewed_dimensions": [deskewed_width, deskewed_height]
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
        raise FileNotFoundError(f"Reference annotated image not found at {reference_image_path}. This is critical for setting up dimensions.")
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
    