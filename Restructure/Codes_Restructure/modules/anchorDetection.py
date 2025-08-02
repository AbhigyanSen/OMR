import cv2
import numpy as np
import math
import os
import json
import csv
import glob
import logging
from datetime import datetime

def setup_logger(batch_name):
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Single log file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{batch_name}_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)

    logger = logging.getLogger("AnchorProcessor")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s |%(levelname)s| %(message)s'))

    # Remove any existing handlers before adding (prevents duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)

    return logger, log_path

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
        self.classes = self._load_classes()
        self.annotations = self._load_annotations()

        self.M_transform = None
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
                        x_center = float(parts[1]) * self.original_width
                        y_center = float(parts[2]) * self.original_height
                        norm_width = float(parts[3]) * self.original_width
                        norm_height = float(parts[4]) * self.original_height

                        x1 = int(x_center - norm_width / 2)
                        y1 = int(y_center - norm_height / 2)
                        x2 = int(x_center + norm_width / 2)
                        y2 = int(y_center + norm_height / 2)

                        annotations.setdefault(class_id, []).append((x1, y1, x2, y2))
        except FileNotFoundError:
            print(f"Annotation file not found at {self.annotations_path}. Proceeding without it.")
        return annotations

    def _load_classes(self):
        classes = []
        try:
            with open(self.classes_path, 'r') as f:
                for line in f:
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
        anchor_class_names = ['anchor_1', 'anchor_2', 'anchor_3', 'anchor_4']

        self.image = self.original_image.copy()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        for class_name_str in anchor_class_names:
            class_id = self._get_class_id(class_name_str)

            if class_id != -1 and class_id in self.annotations:
                for bbox in self.annotations[class_id]:
                    x1, y1, x2, y2 = bbox

                    buffer_scale = 2.0
                    buffer_x = int(max(30, (x2 - x1) * buffer_scale / 2))
                    buffer_y = int(max(30, (y2 - y1) * buffer_scale / 2))

                    search_x1 = max(0, x1 - buffer_x)
                    search_y1 = max(0, y1 - buffer_y)
                    search_x2 = min(self.original_width, x2 + buffer_x)
                    search_y2 = min(self.original_height, y2 + buffer_y)

                    roi = blurred[search_y1:search_y2, search_x1:search_x2]
                    if roi.shape[0] == 0 or roi.shape[1] == 0:
                        continue

                    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    found_anchor_contour = None
                    min_area = 0.2 * (x2 - x1) * (y2 - y1)
                    max_area = 2.0 * (x2 - x1) * (y2 - y1)

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
                        detected_anchors.append({
                            'class_name': class_name_str,
                            'bbox': (search_x1 + fx, search_y1 + fy, search_x1 + fx + fw, search_y1 + fy + fh),
                            'center': (center_x, center_y),
                            'area': cv2.contourArea(found_anchor_contour),
                        })

        return detected_anchors, self.image, None

    def visualize_results(self, detected_anchors, output_filename):
        display_image = self.image.copy()
        anchor_data_for_json = {}
        for anchor in detected_anchors:
            x1, y1, x2, y2 = anchor['bbox']
            center_x, center_y = int(anchor['center'][0]), int(anchor['center'][1])
            class_name = anchor['class_name']

            anchor_data_for_json[class_name] = {"center": [center_x, center_y], "bbox": [x1, y1, x2, y2]}
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(display_image, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imwrite(output_filename, display_image)
        return anchor_data_for_json

def get_annotation_files(annotations_dir):
    labels_dir = os.path.join(annotations_dir, "labels")
    images_dir = os.path.join(annotations_dir, "images")
    classes_file = os.path.join(annotations_dir, "classes.txt")

    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    if len(label_files) != 1 or len(image_files) != 1:
        raise FileNotFoundError("Invalid number of label or image files.")
    annotations_file = label_files[0]
    annotated_image_path = image_files[0]

    label_name = os.path.splitext(os.path.basename(annotations_file))[0]
    image_name = os.path.splitext(os.path.basename(annotated_image_path))[0]
    if label_name != image_name:
        raise ValueError("Label and image filenames do not match.")

    return annotated_image_path, annotations_file, classes_file

def compute_skew_angle(anchor1_center, anchor2_center):
    x1, y1 = anchor1_center
    x2, y2 = anchor2_center
    if abs(x2 - x1) == 0:
        return 90.0 if y2 - y1 > 0 else -90.0
    return round(math.degrees(math.atan2(y2 - y1, abs(x2 - x1))), 4)

def save_rescaled_images(image_folder_path, output_process_image_path, ref_width, ref_height):
    folder_name = os.path.basename(image_folder_path.rstrip("\\/"))
    processed_dir = os.path.join(output_process_image_path, f"processed_{folder_name}")
    os.makedirs(processed_dir, exist_ok=True)
    for filename in os.listdir(image_folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image = cv2.imread(os.path.join(image_folder_path, filename))
            if image is not None:
                resized = cv2.resize(image, (ref_width, ref_height), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(processed_dir, filename), resized)

def generate_generalized_json(base_folder, omr_template_name, date, folder_path, all_image_anchor_data, warning_dir):
    batch_name = os.path.basename(folder_path)
    process_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_images = len([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    output_json = {"TEMPLATE": omr_template_name, "BATCHNAME": batch_name, "PROCESSDT": process_dt,
                   "COUNT": total_images, "IMAGES": []}
    seq_counter = 1
    for image_name, data in all_image_anchor_data.items():
        warning_image_path = os.path.join(warning_dir, image_name)
        skewed = "Y" if os.path.exists(warning_image_path) else "N"
        image_entry = {"IMAGENAME": os.path.abspath(os.path.join(folder_path, image_name)).replace("/", "\\"),
                       "SEQ": seq_counter, "SKEWED": skewed,
                       "ERROR": "N" if data.get("valid_for_option_mapping", False) else "Y", "FIELDS": []}
        output_json["IMAGES"].append(image_entry)
        seq_counter += 1
    final_output_dir = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name)
    os.makedirs(final_output_dir, exist_ok=True)
    with open(os.path.join(final_output_dir, f"{batch_name}.json"), 'w') as f:
        json.dump(output_json, f, indent=2)

def save_anchor_visualizations(processor, detected_anchors, output_path, logger):
    """Save anchor visualization image."""
    processor.visualize_results(detected_anchors, output_path)
    logger.info(f"Anchor visualization saved: {output_path}")

def process_batch(base_folder, omr_template_name, date, batch_name, save_anchor_images=True):
    logger, log_path = setup_logger(batch_name)
    logger.info(f"Processing batch: {batch_name}, Template: {omr_template_name}, Date: {date}")
    logger.info(f"Anchor image saving enabled: {save_anchor_images}")

    folder_path = os.path.join(base_folder, "Images", omr_template_name, date, "Input", batch_name)
    annotations_dir = os.path.join(base_folder, "Annotations", omr_template_name)
    annotated_image_path, annotations_file, classes_file = get_annotation_files(annotations_dir)

    ref_img = cv2.imread(annotated_image_path)
    if ref_img is None:
        logger.error(f"Annotated image not found at {annotated_image_path}")
        raise FileNotFoundError(f"Annotated image not found at {annotated_image_path}")
    ref_height, ref_width = ref_img.shape[:2]

    # Save processed images
    output_folder_path = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name)
    save_rescaled_images(folder_path, output_folder_path, ref_width, ref_height)
    logger.info(f"Processed images saved to {output_folder_path}")

    # Anchors folder (always created for JSON, CSV, and optionally images)
    anchor_output_dir = os.path.join(output_folder_path, f"anchor_{batch_name}")
    os.makedirs(anchor_output_dir, exist_ok=True)
    logger.info(f"Anchor output directory: {anchor_output_dir}")

    warning_dir = os.path.join(base_folder, "Images", omr_template_name, date, "warnings")
    os.makedirs(warning_dir, exist_ok=True)

    all_image_anchor_data = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            logger.info(f"Processing {filename} ...")
            try:
                processor = OMRProcessor(image_path, annotations_file, classes_file, ref_width, ref_height)
                detected_anchors, deskewed_img_result, _ = processor.detect_anchor_points()

                expected_anchors = [cls for cls in processor.classes if cls.startswith("anchor_")]
                if len(detected_anchors) != len(expected_anchors):
                    warning_path = os.path.join(warning_dir, filename)
                    cv2.imwrite(warning_path, deskewed_img_result if deskewed_img_result is not None else processor.original_image)
                    logger.error(f"Anchor count mismatch for {filename}, saved to warnings folder.")
                    all_image_anchor_data[filename] = {
                        "anchors": {a['class_name']: a['center'] for a in detected_anchors},
                        "skew_angle": None,
                        "valid_for_option_mapping": False
                    }
                else:
                    # Save anchor images only if enabled
                    if save_anchor_images:
                        output_image_path = os.path.join(anchor_output_dir, filename)
                        save_anchor_visualizations(processor, detected_anchors, output_image_path, logger)
                    else:
                        logger.info(f"Anchor visualization skipped for {filename} (save_anchor_images=False)")

                    anchor_data = {a['class_name']: {"center": list(a['center']), "bbox": list(a['bbox'])}
                                   for a in detected_anchors}
                    angle = compute_skew_angle(anchor_data["anchor_1"]["center"], anchor_data["anchor_2"]["center"]) \
                        if "anchor_1" in anchor_data and "anchor_2" in anchor_data else None
                    all_image_anchor_data[filename] = {
                        "anchors": anchor_data,
                        "skew_angle": angle,
                        "valid_for_option_mapping": True
                    }

            except Exception as e:
                logger.error(f"Failed processing {filename}: {e}")
                try:
                    error_img = cv2.imread(image_path)
                    if error_img is not None:
                        warning_path = os.path.join(warning_dir, filename)
                        cv2.imwrite(warning_path, error_img)
                        logger.info(f"Original image saved to warnings folder: {warning_path}")
                except Exception as img_err:
                    logger.error(f"Could not save error image for {filename}: {img_err}")

                all_image_anchor_data[filename] = {
                    "anchors": {}, "M_transform": None, "deskewed_width": None, "deskewed_height": None,
                    "valid_for_option_mapping": False, "error": str(e)
                }

    # Save anchor center data (always saved)
    anchor_json_path = os.path.join(anchor_output_dir, "anchor_centers.json")
    with open(anchor_json_path, 'w') as f:
        json.dump(all_image_anchor_data, f, indent=2)
    logger.info(f"Anchor centers JSON saved to {anchor_json_path}")

    csv_output_path = os.path.join(anchor_output_dir, "anchor_centers.csv")
    csv_headers = ["image_name", "anchor_1", "anchor_2", "anchor_3", "anchor_4", "skew_angle", "Warnings"]
    with open(csv_output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_headers)
        for img_name, data in all_image_anchor_data.items():
            anchors = data.get("anchors", {})
            row = [img_name]
            for key in ["anchor_1", "anchor_2", "anchor_3", "anchor_4"]:
                row.append(str(anchors[key]["center"]) if key in anchors else "")
            row.append(data.get("skew_angle", ""))
            row.append("" if data.get("valid_for_option_mapping", True) else "Error")
            writer.writerow(row)
    logger.info(f"Anchor centers CSV saved to {csv_output_path}")

    generate_generalized_json(base_folder, omr_template_name, date, folder_path, all_image_anchor_data, warning_dir)
    logger.info("Generalized batch JSON generated.")

    logger.info(f"Batch completed successfully. Logs saved to {log_path}")
    return all_image_anchor_data