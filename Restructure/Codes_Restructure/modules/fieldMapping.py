import cv2
import numpy as np
import math
import os
import json
import re
import sys
import glob
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict
from PIL import Image
from datetime import datetime

def setup_logger(batch_name):
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Single log file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{batch_name}_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)

    logger = logging.getLogger("FieldMapping")
    logger.setLevel(logging.INFO)

    # File handler (no terminal logging)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s |%(levelname)s| %(message)s'))

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)

    return logger, log_path

# ---------------- CLASS OMRFieldMapper ----------------
class OMRFieldMapper:
    def __init__(self, annotated_image_path, annotations_file, classes_path, 
                 anchor_centers_json_path, target_width, target_height, logger):
        self.annotated_image_path = annotated_image_path
        self.annotations_file = annotations_file
        self.classes_path = classes_path
        self.anchor_centers_json_path = anchor_centers_json_path
        self.target_width = target_width
        self.target_height = target_height

        self.classes = self._load_classes(logger)
        self.reference_annotations = self._load_annotations(self.annotations_file, 
                                                            self.target_width, self.target_height, logger)
        self.all_image_anchor_data = self._load_anchor_centers(logger)
        self.relative_offsets = self._calculate_relative_offsets(logger)

        if not self.relative_offsets:
            logger.critical("Could not calculate relative offsets from reference image. "
                            "Check annotations and anchor_1 presence.")
            sys.exit(1)

    def _load_classes(self, logger):
        classes = []
        try:
            with open(self.classes_path, 'r') as f:
                for line in f:
                    classes.append(line.strip().replace('\r', ''))
            logger.info(f"Loaded {len(classes)} classes from {self.classes_path}")
        except FileNotFoundError:
            logger.error(f"Classes file not found at {self.classes_path}. Cannot proceed.")
            sys.exit(1)
        return classes

    def _load_annotations(self, annotations_path, width, height, logger):
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
            logger.info(f"Loaded annotations from {annotations_path}")
        except FileNotFoundError:
            logger.error(f"Annotation file not found at {annotations_path}.")
        return annotations

    def _load_anchor_centers(self, logger):
        try:
            with open(self.anchor_centers_json_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Anchor centers loaded from {self.anchor_centers_json_path}")
                return data
        except FileNotFoundError:
            logger.error(f"Anchor centers JSON not found at {self.anchor_centers_json_path}. Cannot proceed.")
            sys.exit(1)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {self.anchor_centers_json_path}. Check file format.")
            sys.exit(1)

    def _get_class_id(self, class_name):
        try:
            return self.classes.index(class_name)
        except ValueError:
            return -1

    def _calculate_relative_offsets(self, logger):
        relative_offsets = {}
        anchor_1_class_id = self._get_class_id("anchor_1")
        anchor_2_class_id = self._get_class_id("anchor_2")
        anchor_3_class_id = self._get_class_id("anchor_3")

        if not (self.reference_annotations.get(anchor_1_class_id) and 
                self.reference_annotations.get(anchor_2_class_id) and
                self.reference_annotations.get(anchor_3_class_id)):
            logger.error("Reference anchor_1, anchor_2, or anchor_3 not found in annotations. "
                         "Cannot calculate relative offsets with scaling.")
            return {}

        # Reference anchor centers
        ref_anchor_1_bbox = self.reference_annotations[anchor_1_class_id][0]
        ref_anchor_1_center_x = (ref_anchor_1_bbox[0] + ref_anchor_1_bbox[2]) // 2
        ref_anchor_1_center_y = (ref_anchor_1_bbox[1] + ref_anchor_1_bbox[3]) // 2

        ref_anchor_2_bbox = self.reference_annotations[anchor_2_class_id][0]
        ref_anchor_2_center_x = (ref_anchor_2_bbox[0] + ref_anchor_2_bbox[2]) // 2
        ref_anchor_2_center_y = (ref_anchor_2_bbox[1] + ref_anchor_2_bbox[3]) // 2

        ref_anchor_3_bbox = self.reference_annotations[anchor_3_class_id][0]
        ref_anchor_3_center_x = (ref_anchor_3_bbox[0] + ref_anchor_3_bbox[2]) // 2
        ref_anchor_3_center_y = (ref_anchor_3_bbox[1] + ref_anchor_3_bbox[3]) // 2

        # Calculate reference distances
        ref_horizontal_dist = math.sqrt((ref_anchor_2_center_x - ref_anchor_1_center_x)**2 + 
                                        (ref_anchor_2_center_y - ref_anchor_1_center_y)**2)
        ref_vertical_dist = math.sqrt((ref_anchor_3_center_x - ref_anchor_1_center_x)**2 + 
                                      (ref_anchor_3_center_y - ref_anchor_1_center_y)**2)

        if ref_horizontal_dist == 0 or ref_vertical_dist == 0:
            logger.error("Reference anchor distances are zero. Cannot calculate normalized offsets.")
            return {}

        for class_id, bboxes in self.reference_annotations.items():
            class_name = self.classes[class_id]
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1

                dx = x1 - ref_anchor_1_center_x
                dy = y1 - ref_anchor_1_center_y

                norm_dx = dx / ref_horizontal_dist
                norm_dy = dy / ref_vertical_dist
                norm_width = width / ref_horizontal_dist
                norm_height = height / ref_vertical_dist

                unique_key = class_name if len(bboxes) == 1 else f"{class_name}_{i}"

                relative_offsets[unique_key] = {
                    "norm_dx": norm_dx,
                    "norm_dy": norm_dy,
                    "norm_width": norm_width,
                    "norm_height": norm_height
                }
                
        logger.info(f"Calculated {len(relative_offsets)} normalized relative offsets "
                    f"from reference anchor_1's center.")
        return relative_offsets

    def _order_points(self, points):
        rect = np.zeros((4, 2), dtype="float32")

        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)] # Top-left
        rect[2] = points[np.argmax(s)] # Bottom-right

        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)] # Top-right
        rect[3] = points[np.argmax(diff)] # Bottom-left

        return rect

    def deskew_image(self, image, detected_anchors_data, logger):
        anchor_coords = []
        for i in range(1, 5):
            anchor_name = f'anchor_{i}'
            if anchor_name in detected_anchors_data and 'center' in detected_anchors_data[anchor_name]:
                anchor_coords.append(detected_anchors_data[anchor_name]['center'])
            else:
                logger.warning(f"Missing {anchor_name} for deskewing. "
                               "Falling back to rotational deskew.")
                return self._fallback_rotational_deskew(image, detected_anchors_data)

        if len(anchor_coords) != 4:
            logger.warning(f"Only {len(anchor_coords)} anchors found. "
                           "Need 4 for perspective deskew. Using fallback.")
            return self._fallback_rotational_deskew(image, detected_anchors_data)

        src_points = np.array(anchor_coords, dtype="float32")
        ordered_src_points = self._order_points(src_points)

        ref_anchor_ids = [self._get_class_id(f"anchor_{i}") for i in range(1, 5)]
        if not all(self.reference_annotations.get(aid) for aid in ref_anchor_ids):
            logger.error("Reference annotations for all 4 anchors missing. "
                         "Cannot define ideal destination points. Using fallback.")
            return self._fallback_rotational_deskew(image, detected_anchors_data)

        ref_centers = []
        for aid in ref_anchor_ids:
            bbox = self.reference_annotations[aid][0]
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            ref_centers.append([center_x, center_y])

        dst_points = self._order_points(np.array(ref_centers, dtype="float32"))
        deskewed_width = self.target_width
        deskewed_height = self.target_height

        M = cv2.getPerspectiveTransform(ordered_src_points, dst_points)
        deskewed_image = cv2.warpPerspective(image, M, (deskewed_width, deskewed_height))
        logger.info(f"Image deskewed to {deskewed_width}x{deskewed_height} using perspective transform.")
        return deskewed_image, M, deskewed_width, deskewed_height

    def _fallback_rotational_deskew(self, image, detected_anchors_data, logger):
        anchor1 = detected_anchors_data.get('anchor_1')
        anchor2 = detected_anchors_data.get('anchor_2')
        if anchor1 and anchor2 and 'center' in anchor1 and 'center' in anchor2:
            angle = self._compute_skew_angle(anchor1['center'], anchor2['center'])
            if abs(angle) > 0.1:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_image = cv2.warpAffine(image, M_rot, (w, h), 
                                               flags=cv2.INTER_CUBIC, 
                                               borderMode=cv2.BORDER_REPLICATE)
                logger.warning(f"Image rotated by {angle:.2f} degrees (rotational deskew fallback).")
                return rotated_image, M_rot, w, h
        logger.warning("No suitable deskewing performed. Returning original image.")
        return image, None, self.target_width, self.target_height

    def _compute_skew_angle(self, anchor1_center, anchor2_center):
        x1, y1 = anchor1_center
        x2, y2 = anchor2_center
        base = x2 - x1 
        height = y2 - y1 

        if base == 0:
            return 90.0 if height > 0 else -90.0

        angle_rad = math.atan2(height, base)
        return round(math.degrees(angle_rad), 4)

    def map_fields_and_visualize(self, image_path, output_dir, missing_fields_dir, raw_save_dir, org_image_dir, save_mapped_images, logger):
        os.makedirs(raw_save_dir, exist_ok=True)
        os.makedirs(org_image_dir, exist_ok=True)

        filename = os.path.basename(image_path)
        image_data = self.all_image_anchor_data.get(filename)

        if not image_data:
            logger.warning(f"No anchor data found for {filename}. Skipping field mapping.")
            return {"status": "skipped_no_anchor_data",
                    "mapped_fields": {},
                    "missing_fields": list(self.relative_offsets.keys())}

        if not image_data.get("valid_for_option_mapping", False):
            logger.warning(f"Image {filename} flagged as invalid for option mapping. Skipping.")
            return {"status": "skipped_invalid_for_option_mapping",
                    "mapped_fields": {},
                    "missing_fields": list(self.relative_offsets.keys())}

        logger.info(f"Processing {filename} for field mapping...")
        original_image = cv2.imread(image_path)
        if original_image is None:
            logger.error(f"Could not read image: {image_path}. Skipping.")
            return {"status": "skipped_image_read_error",
                    "mapped_fields": {},
                    "missing_fields": list(self.relative_offsets.keys())}

        # Resize to match template dimensions
        if original_image.shape[1] != self.target_width or original_image.shape[0] != self.target_height:
            original_image = cv2.resize(original_image, (self.target_width, self.target_height),
                                        interpolation=cv2.INTER_LINEAR)
            logger.info(f"Resized {filename} to {self.target_width}x{self.target_height}")

        detected_anchors_for_image = image_data.get("anchors", {})
        deskewed_image, M_transform, deskewed_width, deskewed_height = self.deskew_image(
            original_image.copy(), detected_anchors_for_image, logger=logger
        )

        # Transform anchors
        transformed_anchors = {}
        if M_transform is not None and len(M_transform) == 3:
            for name, anchor_data in detected_anchors_for_image.items():
                if "center" in anchor_data:
                    pt = np.array([[anchor_data["center"]]], dtype=np.float32)
                    transformed_pt = cv2.perspectiveTransform(pt, M_transform)[0][0]
                    transformed_anchors[name] = {
                        "center": (int(transformed_pt[0]), int(transformed_pt[1])),
                        "bbox": anchor_data["bbox"]
                    }
        else:
            transformed_anchors = detected_anchors_for_image

        try:
            anchor1_center = transformed_anchors["anchor_1"]["center"]
            anchor2_center = transformed_anchors["anchor_2"]["center"]
            anchor3_center = transformed_anchors["anchor_3"]["center"]
        except KeyError:
            logger.error(f"Required anchors missing after transformation for {filename}. Cannot map fields.")
            return {"status": "skipped_ref_anchor_data_missing",
                    "mapped_fields": {},
                    "missing_fields": list(self.relative_offsets.keys())}

        # Scale calculation
        current_horizontal_dist = math.sqrt((anchor2_center[0] - anchor1_center[0]) ** 2 +
                                            (anchor2_center[1] - anchor1_center[1]) ** 2)
        current_vertical_dist = math.sqrt((anchor3_center[0] - anchor1_center[0]) ** 2 +
                                        (anchor3_center[1] - anchor1_center[1]) ** 2)
        if current_horizontal_dist == 0 or current_vertical_dist == 0:
            logger.error(f"Current anchor distances are zero for {filename}. Cannot map fields accurately.")
            return {"status": "skipped_zero_current_anchor_dist",
                    "mapped_fields": {},
                    "missing_fields": list(self.relative_offsets.keys())}

        mapped_fields_data = {}
        missing_fields = []
        display_image = deskewed_image.copy()

        for unique_key, offset_data in self.relative_offsets.items():
            x1_mapped = anchor1_center[0] + (offset_data["norm_dx"] * current_horizontal_dist)
            y1_mapped = anchor1_center[1] + (offset_data["norm_dy"] * current_vertical_dist)
            width_mapped = offset_data["norm_width"] * current_horizontal_dist
            height_mapped = offset_data["norm_height"] * current_vertical_dist
            x2_mapped = x1_mapped + width_mapped
            y2_mapped = y1_mapped + height_mapped

            # Clip bounding box
            x1_mapped = max(0, int(x1_mapped))
            y1_mapped = max(0, int(y1_mapped))
            x2_mapped = min(deskewed_width, int(x2_mapped))
            y2_mapped = min(deskewed_height, int(y2_mapped))

            if x1_mapped >= x2_mapped or y1_mapped >= y2_mapped:
                logger.warning(f"Mapped bbox for {unique_key} invalid [{x1_mapped},{y1_mapped},{x2_mapped},{y2_mapped}]. Marked missing.")
                missing_fields.append(unique_key)
                continue

            mapped_fields_data[unique_key] = {
                "bbox": [x1_mapped, y1_mapped, x2_mapped, y2_mapped],
                "width": x2_mapped - x1_mapped,
                "height": y2_mapped - y1_mapped
            }
            cv2.rectangle(display_image, (x1_mapped, y1_mapped), (x2_mapped, y2_mapped), (34, 139, 34), 2)

        # Save deskewed image in raw_save_dir and org_image_dir
        raw_output_path = os.path.join(raw_save_dir, filename)
        org_output_path = os.path.join(org_image_dir, filename)
        cv2.imwrite(raw_output_path, deskewed_image)
        cv2.imwrite(org_output_path, deskewed_image)
        logger.info(f"Raw deskewed image saved to {raw_output_path} and {org_output_path}")

        # Save mapped image only if toggle is True
        if save_mapped_images:
            output_image_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_image_path, display_image)
            logger.info(f"Mapped image saved to {output_image_path}")

        # Missing fields log
        if missing_fields:
            missing_file_path = os.path.join(missing_fields_dir,
                                            f"{os.path.splitext(filename)[0]}_missing_fields.txt")
            with open(missing_file_path, 'w') as f:
                for field in missing_fields:
                    f.write(f"{field}\n")
            logger.warning(f"Missing fields for {filename} saved to {missing_file_path}")

        return {"status": "processed",
                "mapped_fields": mapped_fields_data,
                "missing_fields": missing_fields,
                "deskewed_dimensions": [deskewed_width, deskewed_height]}

    
    # def save_cropped_fields(self, deskewed_image_path, mapped_fields,
    #                         output_base_dir, image_filename,
    #                         target_field_names, key_field_mapping, logger):
    #     deskewed_image = cv2.imread(deskewed_image_path)
    #     if deskewed_image is None:
    #         logger.error(f"Could not read deskewed image: {deskewed_image_path}. Skipping crops.")
    #         return

    #     for key, data in mapped_fields.items():
    #         if key in target_field_names:
    #             x1, y1, x2, y2 = data["bbox"]
    #             h, w = deskewed_image.shape[:2]
    #             x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    #             if x2 <= x1 or y2 <= y1:
    #                 logger.warning(f"Invalid bbox for {key} in {image_filename}, skipping.")
    #                 continue

    #             cropped = deskewed_image[y1:y2, x1:x2]
    #             folder_name = key_field_mapping.get(key, key)
    #             field_folder = os.path.join(output_base_dir, folder_name)
    #             os.makedirs(field_folder, exist_ok=True)

    #             save_name = os.path.splitext(image_filename)[0] + ".jpg"
    #             save_path = os.path.join(field_folder, save_name)
    #             cv2.imwrite(save_path, cropped)
    #             logger.info(f"Cropped field '{folder_name}' saved to {save_path}")
    
    def save_cropped_fields(self, deskewed_image_path, mapped_fields,
                            output_base_dir, image_filename,
                            target_field_names, key_field_mapping, logger):
        deskewed_image = cv2.imread(deskewed_image_path)
        if deskewed_image is None:
            logger.error(f"Could not read deskewed image: {deskewed_image_path}. Skipping crops.")
            return

        # ---- Process normal keys ----
        for key, data in mapped_fields.items():
            if key in target_field_names:
                x1, y1, x2, y2 = data["bbox"]
                h, w = deskewed_image.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid bbox for {key} in {image_filename}, skipping.")
                    continue

                cropped = deskewed_image[y1:y2, x1:x2]
                folder_name = key_field_mapping.get(key, key)
                field_folder = os.path.join(output_base_dir, folder_name)
                os.makedirs(field_folder, exist_ok=True)

                save_name = os.path.splitext(image_filename)[0] + ".jpg"
                save_path = os.path.join(field_folder, save_name)
                cv2.imwrite(save_path, cropped)
                logger.info(f"Cropped field '{folder_name}' saved to {save_path}")

        # ---- Additional static crop: omr_sheet_no ----
        if "omr_sheet_no" in mapped_fields:
            x1, y1, x2, y2 = mapped_fields["omr_sheet_no"]["bbox"]
            h, w = deskewed_image.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                cropped = deskewed_image[y1:y2, x1:x2]
                field_folder = os.path.join(output_base_dir, "OMR SHEET NUMBER")
                os.makedirs(field_folder, exist_ok=True)

                save_name = os.path.splitext(image_filename)[0] + ".jpg"
                save_path = os.path.join(field_folder, save_name)
                cv2.imwrite(save_path, cropped)
                logger.info(f"Cropped field 'OMR SHEET NUMBER' saved to {save_path}")
            else:
                logger.warning(f"Invalid bbox for omr_sheet_no in {image_filename}, skipping.")


def get_annotation_files(annotations_dir, logger):
    # --- Validate base folder ---
    if not os.path.isdir(annotations_dir):
        logger.error(f"Annotations folder not found: {annotations_dir}")
        sys.exit(1)

    labels_dir = os.path.join(annotations_dir, "labels")
    images_dir = os.path.join(annotations_dir, "images")
    classes_file = os.path.join(annotations_dir, "classes.txt")

    if not os.path.isdir(labels_dir):
        logger.error(f"Labels folder not found: {labels_dir}")
        sys.exit(1)
    if not os.path.isdir(images_dir):
        logger.error(f"Images folder not found: {images_dir}")
        sys.exit(1)
    if not os.path.isfile(classes_file):
        logger.error(f"classes.txt file missing: {classes_file}")
        sys.exit(1)

    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if len(label_files) != 1:
        logger.error(f"Expected exactly 1 label file in {labels_dir}, found {len(label_files)}")
        sys.exit(1)
    annotations_file = label_files[0]

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    if len(image_files) != 1:
        logger.error(f"Expected exactly 1 image file in {images_dir}, found {len(image_files)}")
        sys.exit(1)
    annotated_image_path = image_files[0]

    label_name = os.path.splitext(os.path.basename(annotations_file))[0]
    image_name = os.path.splitext(os.path.basename(annotated_image_path))[0]
    if label_name != image_name:
        logger.error(f"Label file '{label_name}' does not match image file '{image_name}'")
        sys.exit(1)

    logger.info(f"Using annotation file: {annotations_file}")
    logger.info(f"Using annotated image: {annotated_image_path}")
    logger.info(f"Using classes file: {classes_file}")

    return annotated_image_path, annotations_file, classes_file

def generate_generalized_json(base_json_path, mapper_json_path, output_path, key_fields_json_path, logger):
    with open(key_fields_json_path, 'r') as f:
        key_fields_mapping = json.load(f)
    dynamic_key_patterns = [re.compile(f"^{re.escape(key)}_\\d+(_\\d+)*$", re.IGNORECASE) for key in key_fields_mapping.keys()]

    unwanted_patterns = [
        re.compile(r"^anchor", re.IGNORECASE),
        re.compile(r"^\d+[A-D]$", re.IGNORECASE)
    ] + dynamic_key_patterns

    def is_unwanted(field_name):
        return any(pattern.match(field_name) for pattern in unwanted_patterns)

    with open(base_json_path, 'r') as f:
        base_data = json.load(f)
    with open(mapper_json_path, 'r') as f:
        mapper_data = json.load(f)

    mapper_lookup = {fname: details.get("mapped_fields", {}) for fname, details in mapper_data.items()}

    missing_fields_log_dir = os.path.join(
        os.path.dirname(output_path),
        "annotate_" + os.path.splitext(os.path.basename(output_path))[0],
        "missing_fields_logs"
    )
    missing_field_files = set()
    if os.path.exists(missing_fields_log_dir):
        for file in os.listdir(missing_fields_log_dir):
            if file.endswith(".txt"):
                missing_field_files.add(file)

    for image_entry in base_data.get("IMAGES", []):
        filename = os.path.basename(image_entry.get("IMAGENAME", ""))
        mapped_fields = mapper_lookup.get(filename, {})

        fields_list = []
        for field_name, field_info in mapped_fields.items():
            if is_unwanted(field_name):
                continue
            fields_list.append({
                "FIELD": field_name,
                "XCORD": str(field_info["bbox"][0]),
                "YCORD": str(field_info["bbox"][1]),
                "WIDTH": str(field_info["width"]),
                "HEIGHT": str(field_info["height"])
            })
        image_entry["FIELDS"] = fields_list

        txt_filename = os.path.splitext(filename)[0] + ".txt"
        if txt_filename in missing_field_files:
            image_entry["ERROR"] = "Y"

    with open(output_path, 'w') as f:
        json.dump(base_data, f, indent=2)

    logger.info(f"Updated Generalized JSON saved to: {output_path}")

def convert_images_to_bw(raw_save_dir, logger, threshold=100):
    try:
        files = os.listdir(raw_save_dir)
    except Exception as e:
        logger.error(f"Error accessing folder {raw_save_dir}: {e}")
        return

    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(raw_save_dir, file)
            try:
                with Image.open(image_path) as img:
                    gray_img = img.convert("L")
                    bw_img = gray_img.point(lambda x: 255 if x > threshold else 0, '1')
                    bw_img.save(image_path)
                    logger.info(f"Converted {file} to pure black & white.")
            except Exception as e:
                logger.error(f"Error converting {file}: {e}")

def process_field_mapping(base_folder, omr_template_name, date, batch_name, save_mapped_images=True):
    # ---- Setup Logger ----
    logger, log_path = setup_logger(batch_name)
    logger.info(f"Starting field mapping for batch {batch_name}")
    logger.info(f"Logs stored at: {log_path}")

    processed_images_folder = os.path.join(base_folder, "Images", omr_template_name, date,
                                           "Output", batch_name, f"processed_{batch_name}")
    raw_save_dir = os.path.join(base_folder, "Images", omr_template_name, date,
                                "Output", batch_name, f"raw_{batch_name}")
    original_img_dir = os.path.join(base_folder, "Images", omr_template_name, date,
                                    "Input", batch_name)
    annotations_dir = os.path.join(base_folder, "Annotations", omr_template_name)

    annotated_image_path, annotations_file, classes_file = get_annotation_files(annotations_dir, logger)
    anchor_output_folder_name = "anchor_" + batch_name
    anchor_centers_json_path = os.path.join(base_folder, "Images", omr_template_name, date,
                                            "Output", batch_name, anchor_output_folder_name,
                                            "anchor_centers.json")

    ref_img = cv2.imread(annotated_image_path)
    if ref_img is None:
        logger.error(f"Reference annotated image not found at {annotated_image_path}")
        raise FileNotFoundError(f"Reference annotated image not found at {annotated_image_path}")
    ref_height, ref_width = ref_img.shape[:2]
    logger.info(f"Reference dimensions: {ref_width}x{ref_height}")

    mapping_output_folder_name = "annotate_" + batch_name
    output_images_dir = os.path.join(base_folder, "Images", omr_template_name, date,
                                     "Output", batch_name, mapping_output_folder_name, "mapped_images")
    missing_fields_log_dir = os.path.join(base_folder, "Images", omr_template_name, date,
                                          "Output", batch_name, mapping_output_folder_name, "missing_fields_logs")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(missing_fields_log_dir, exist_ok=True)

    key_fields_json_path = os.path.join(base_folder, "Annotations", omr_template_name, "key_fields.json")
    with open(key_fields_json_path, 'r') as f:
        key_fields_mapping = json.load(f)
    target_field_names = list(key_fields_mapping.keys())

    mapper = OMRFieldMapper(
        annotated_image_path=annotated_image_path,
        annotations_file=annotations_file,
        classes_path=classes_file,
        anchor_centers_json_path=anchor_centers_json_path,
        target_width=ref_width,
        target_height=ref_height,
        logger=logger
    )

    all_image_field_data = {}
    for filename in os.listdir(processed_images_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(processed_images_folder, filename)
            logger.info(f"Starting field mapping for {filename}")
            field_data = mapper.map_fields_and_visualize(
                image_path=image_path,
                output_dir=output_images_dir,
                missing_fields_dir=missing_fields_log_dir,
                raw_save_dir=raw_save_dir,
                org_image_dir=original_img_dir,
                save_mapped_images=save_mapped_images,
                logger=logger
            )
            all_image_field_data[filename] = field_data

            if field_data["status"] == "processed":
                icr_output_dir = os.path.join(base_folder, "Images", omr_template_name, date,
                                              "Output", batch_name, mapping_output_folder_name, "ICR")
                deskewed_image_path = os.path.join(raw_save_dir, filename)
                mapper.save_cropped_fields(
                    deskewed_image_path=deskewed_image_path,
                    mapped_fields=field_data["mapped_fields"],
                    output_base_dir=icr_output_dir,
                    image_filename=filename,
                    target_field_names=target_field_names,
                    key_field_mapping=key_fields_mapping,
                    logger=logger
                )
            logger.info(f"Finished field mapping for {filename}")

    json_output_path = os.path.join(base_folder, "Images", omr_template_name, date,
                                    "Output", batch_name, mapping_output_folder_name, "field_mappings.json")
    with open(json_output_path, 'w') as f:
        json.dump(all_image_field_data, f, indent=2)
    logger.info(f"All field mappings saved to {json_output_path}")

    final_generalized_json_path = os.path.join(base_folder, "Images", omr_template_name, date,
                                               "Output", batch_name, f"{batch_name}.json")

    generate_generalized_json(final_generalized_json_path, json_output_path,
                              final_generalized_json_path, key_fields_json_path, logger)
    
    convert_images_to_bw(raw_save_dir, logger=logger)
    
    logger.info(f"Field mapping completed. Log saved to {log_path}")

    return all_image_field_data