import os
import cv2
import json
import numpy as np
import csv
import re
import pandas as pd
import sys
import logging
from datetime import datetime
from collections import defaultdict

def setup_logger(batch_name):
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Single log file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{batch_name}_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)

    logger = logging.getLogger("MarkedOptions")
    logger.setLevel(logging.INFO)

    # File handler (no terminal logging)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s |%(levelname)s| %(message)s'))

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)

    return logger, log_path

def get_digits_count(classes_file):
    digits_count = {}
    with open(classes_file, 'r') as f:
        for line in f:
            label = line.strip()
            match = re.match(r'^(key\d+)_(\d+)$', label)
            if match:
                key_name = match.group(1)
                idx = int(match.group(2))
                digits_count[key_name] = max(digits_count.get(key_name, -1), idx)
    # Add +1 because index starts from 0
    return {k: v + 1 for k, v in digits_count.items()}

def is_filled_option(image, bbox, threshold=100):
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 255
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    return mean_intensity < threshold, mean_intensity

def extract_question_id(label):
    match = re.match(r"(\d{1,2})([a-d])", label.lower())
    return f"Q{match.group(1)}" if match else None

def extract_option_code(label):
    match = re.match(r"\d{1,2}([a-d])", label.lower())
    return match.group(1).upper() if match else None

def is_digit_option(label, prefix):
    return bool(re.match(rf"{prefix}_\d+_\d", label))

def extract_digit_group(label, prefix):
    match = re.match(rf"{prefix}_(\d+)_\d", label)
    return f"{prefix}_{match.group(1)}" if match else None

def extract_digit_value(label):
    match = re.match(r".+_(\d)$", label)
    return match.group(1) if match else None

def extract_final_number(group_dict, prefix, total_digits, image, option_score_map):
    """
    For each digit position, choose the darkest bubble (lowest mean intensity).
    """
    number_digits = []
    for i in range(total_digits):
        group_key = f"{prefix}_{i}"
        options = group_dict.get(group_key, [])
        best_score = float("inf")
        marked_digit = ""

        for label, bbox in options:
            _, score = is_filled_option(image, bbox)
            option_score_map[label] = round(score, 2)
            if score < best_score:
                best_score = score
                marked_digit = extract_digit_value(label)

        number_digits.append(marked_digit or "")
    return "".join(number_digits)
    
def detect_marked_options(mapped_json_path, processed_images_folder,
                          key_fields_json, classes_file, logger):
    """
    Optimized detection of marked options:
    Returns only option intensity scores required for verification and edge detection.
    """

    logger.info("Starting optimized marked options detection...")
    logger.info(f"Mapped JSON path: {mapped_json_path}")
    logger.info(f"Processed images folder: {processed_images_folder}")

    # Load dynamic key fields and digit counts
    with open(key_fields_json, 'r') as f:
        key_fields = json.load(f)
    digits_count = get_digits_count(classes_file)
    logger.info(f"Loaded key fields: {list(key_fields.keys())}")
    logger.info(f"Digit counts: {digits_count}")

    with open(mapped_json_path, 'r') as f:
        mapped_data = json.load(f)

    global_option_score_map = {}

    for img_name, annotations in mapped_data.items():
        if not isinstance(annotations, dict) or annotations.get("status", "").lower() != "processed":
            logger.warning(f"Skipping {img_name}: not marked as 'processed'.")
            continue

        image_path = os.path.join(processed_images_folder, img_name)
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            continue

        question_groups = defaultdict(list)
        digit_groups = {k: defaultdict(list) for k in key_fields.keys()}
        option_score_map = {}

        fields = annotations.get("mapped_fields", {})
        for label, data in fields.items():
            if not isinstance(data, dict) or "bbox" not in data:
                continue
            bbox = data["bbox"]

            # Regular question options like "12A", "5D"
            if re.match(r"\d{1,2}[a-dA-D]$", label):
                qid = extract_question_id(label)
                if qid:
                    question_groups[qid].append((label, bbox))
            else:
                # Dynamic digit fields (roll numbers, keys)
                for key_name in key_fields.keys():
                    if is_digit_option(label, key_name):
                        group_key = extract_digit_group(label, key_name)
                        digit_groups[key_name][group_key].append((label, bbox))

        # Process question options (store only intensity values)
        for qid in question_groups.keys():
            for label, bbox in question_groups[qid]:
                _, score = is_filled_option(image, bbox)
                option_score_map[label] = round(score, 2)

        # Process dynamic numeric keys
        for key_name in key_fields.keys():
            for group_key, options in digit_groups[key_name].items():
                for label, bbox in options:
                    _, score = is_filled_option(image, bbox)
                    option_score_map[label] = round(score, 2)

        global_option_score_map[img_name] = option_score_map
        logger.info(f"Processed image: {img_name}")

    logger.info("Optimized marked options detection completed.")
    return global_option_score_map

def clean_and_export_summary(marked_options_path, summary_json_path, summary_csv_path,
                             key_fields_json, classes_file, logger):
    logger.info("Cleaning and exporting summary...")
    logger.info(f"Marked options path: {marked_options_path}")

    with open(marked_options_path, 'r') as f:
        flat_data = json.load(f)
    with open(key_fields_json, 'r') as f:
        key_fields = json.load(f)

    digit_counts = get_digits_count(classes_file)
    logger.info(f"Digit counts: {digit_counts}")

    summary_dict = defaultdict(dict)
    for full_key, value in flat_data.items():
        if '_' not in full_key or not value:
            continue
        img_name, label = full_key.split("_", 1)
        if re.match(r"^Q\d+$", label):
            summary_dict[img_name][label] = value
        else:
            for key_name in key_fields.keys():
                if label.startswith(key_name) and value.isdigit():
                    summary_dict[img_name][label] = value

    for img_name in list(summary_dict.keys()):
        for key_name in key_fields.keys():
            digit_count = digit_counts.get(key_name, 0)
            final_value = summary_dict[img_name].get(key_name, "")
            if not final_value:
                final_value = "".join(
                    summary_dict[img_name].get(f"{key_name}_{i}", "")
                    for i in range(digit_count)
                )
            summary_dict[img_name][key_name] = final_value
            logger.debug(f"{img_name}: {key_name} -> {final_value}")

    with open(summary_json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    logger.info(f"Cleaned summary saved to: {summary_json_path}")

    all_questions = sorted(
        {q for q_data in summary_dict.values() for q in q_data.keys() if q.startswith("Q")},
        key=lambda x: int(x[1:])
    )
    csv_header = ["Image Name"] + all_questions + list(key_fields.keys())

    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for img_name, q_answers in summary_dict.items():
            row = [img_name] + [q_answers.get(q, "") for q in all_questions]
            for key_name in key_fields.keys():
                row.append(q_answers.get(key_name, ""))
            writer.writerow(row)
    logger.info(f"Summary CSV saved to: {summary_csv_path}")

def export_verification_csv(option_score_map, verification_csv_path, key_fields_json, classes_file):
    # ---- Load keys ----
    with open(key_fields_json, "r") as f:
        key_fields = json.load(f)
    with open(classes_file, "r") as f:
        class_labels = f.read().splitlines()

    # ---- Determine digit labels per key ----
    key_digit_labels = defaultdict(lambda: defaultdict(list))  # {key: {digit_idx: [labels]}}
    for label in class_labels:
        m = re.match(r"^(key\d+)_(\d+)_(\d+)$", label)
        if m:
            key, digit_idx, val = m.groups()
            key_digit_labels[key][int(digit_idx)].append(label)

    # ---- Patterns ----
    question_pattern = re.compile(r"^(\d{1,2})([A-Da-d])$")

    all_labels = set()
    for score_map in option_score_map.values():
        all_labels.update(score_map.keys())

    question_map = defaultdict(list)
    for label in all_labels:
        qmatch = question_pattern.match(label)
        if qmatch:
            qnum, opt = qmatch.groups()
            question_map[qnum].append(f"{qnum}{opt.upper()}")

    sorted_questions = sorted(question_map.keys(), key=lambda x: int(x))

    # ---- CSV Header ----
    header = ["Image Name"]
    # Question columns
    for q in sorted_questions:
        for opt in ["A", "B", "C", "D"]:
            label = f"{q}{opt}"
            header.append(label)
            header.append(f"Result {label}")

    # Dynamic key columns (only those that exist in classes.txt)
    for key, digits in key_digit_labels.items():
        for digit_idx, labels in sorted(digits.items()):
            for label in sorted(labels):
                header.append(label)
                header.append(f"Result {label}")

    # ---- Write CSV ----
    with open(verification_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for img_name in sorted(option_score_map.keys()):
            row = [img_name]
            score_map = option_score_map[img_name]

            # Questions
            for q in sorted_questions:
                scores = {}
                for opt in ["A", "B", "C", "D"]:
                    label = f"{q}{opt}"
                    try:
                        scores[opt] = float(score_map.get(label, ""))
                    except:
                        pass
                max_score = max(scores.values()) if scores else 1.0

                for opt in ["A", "B", "C", "D"]:
                    label = f"{q}{opt}"
                    raw = score_map.get(label, "")
                    row.append(raw)
                    try:
                        val = float(raw)
                        pct = round((val / max_score) * 100, 2) if max_score else ""
                        row.append(f"{pct}")
                    except:
                        row.append("")

            # Dynamic keys (use only labels that exist in classes.txt)
            for key, digits in key_digit_labels.items():
                for digit_idx, labels in sorted(digits.items()):
                    # Determine max score for this digit group
                    scores = {}
                    for label in labels:
                        try:
                            scores[label] = float(score_map.get(label, ""))
                        except:
                            pass
                    max_score = max(scores.values()) if scores else 1.0

                    for label in sorted(labels):
                        raw = score_map.get(label, "")
                        row.append(raw)
                        try:
                            v = float(raw)
                            pct = round((v / max_score) * 100, 2) if max_score else ""
                            row.append(f"{pct}")
                        except:
                            row.append("")
            writer.writerow(row)

    print(f"✅ Final verification.csv written to: {verification_csv_path}")

# def evaluate_edge_cases(verification_csv_path, edge_json_path, edge_csv_path,
#                         key_fields_json, classes_file, logger):

#     logger.info("Evaluating edge cases...")
#     logger.info(f"Verification CSV path: {verification_csv_path}")

#     # Load key field metadata
#     with open(key_fields_json, "r") as f:
#         key_fields = json.load(f)
#     with open(classes_file, "r") as f:
#         class_labels = f.read().splitlines()

#     # Count digits per key from classes.txt
#     key_digit_counts = defaultdict(int)
#     for label in class_labels:
#         m = re.match(r"^(key\d+)_(\d+)$", label)
#         if m:
#             key, digit_idx = m.groups()
#             digit_idx = int(digit_idx)
#             key_digit_counts[key] = max(key_digit_counts[key], digit_idx + 1)
#     logger.debug(f"Key digit counts: {key_digit_counts}")

#     # Load verification CSV
#     df = pd.read_csv(verification_csv_path)
#     result_data = {}

#     # Identify question result columns
#     result_cols = [col for col in df.columns if re.match(r"^Result \d{1,2}[A-Da-d]$", col)]
#     question_ids = sorted(set(col.split()[1][:-1] for col in result_cols), key=lambda x: int(x))
#     logger.info(f"Detected question IDs: {question_ids}")

#     # --- Updated helper for keys ---
#     def extract_number_from_result(row, key_name, digit_count):
#         digit_outputs = []
#         for d in range(digit_count):
#             fully_marked, partially_marked = [], []
#             for val in range(10):
#                 col = f"{key_name}_{d}_{val}"   # <-- no "Result" prefix now
#                 if col in row:
#                     try:
#                         pct = float(row[col])
#                         if pct < 185:  # fully marked bubble
#                             fully_marked.append(f"{key_name}_{d}_{val}")
#                         elif 185 <= pct <= 242:  # partial marking
#                             partially_marked.append(f"{key_name}_{d}_{val}")
#                     except:
#                         continue

#             # Decision for each digit
#             if len(fully_marked) == 1 and not partially_marked:
#                 digit_outputs.append(fully_marked[0])
#             elif len(fully_marked) > 1 or (fully_marked and partially_marked):
#                 digit_outputs.append("|".join(fully_marked + partially_marked))
#             elif not fully_marked and len(partially_marked) == 1:
#                 digit_outputs.append(partially_marked[0])
#             elif not fully_marked and len(partially_marked) > 1:
#                 digit_outputs.append("|".join(partially_marked))
#             else:
#                 digit_outputs.append(" ")
#         return "||".join(digit_outputs)


#     for idx, row in df.iterrows():
#         image_name = row["Image Name"]
#         result_data[image_name] = {}

#         # ------------------------
#         # Block 1: Handle Questions
#         # ------------------------
#         for qid in question_ids:
#             fully_marked, partially_marked = [], []
#             for opt in ['A', 'B', 'C', 'D']:
#                 mean_col = f"{qid}{opt}"
#                 if mean_col in row:
#                     try:
#                         mean_val = float(row[mean_col])
#                         if mean_val < 185:
#                             fully_marked.append(opt)
#                         elif 185 <= mean_val <= 242:
#                             partially_marked.append(opt)
#                     except:
#                         continue

#             if len(fully_marked) == 1 and len(partially_marked) == 0:
#                 result_data[image_name][f"Q{qid}"] = fully_marked[0]
#             elif len(fully_marked) > 1 or (fully_marked and partially_marked):
#                 result_data[image_name][f"Q{qid}"] = "*"
#             elif len(fully_marked) == 0 and len(partially_marked) == 1:
#                 result_data[image_name][f"Q{qid}"] = "?"
#             elif len(fully_marked) == 0 and len(partially_marked) > 1:
#                 result_data[image_name][f"Q{qid}"] = "*"
#             else:
#                 result_data[image_name][f"Q{qid}"] = ""

#         # ------------------------
#         # Block 2: Handle Key Fields
#         # ------------------------
#         for key_name, human_name in key_fields.items():
#             digit_count = key_digit_counts.get(key_name, 0)
#             if digit_count > 0:
#                 value = extract_number_from_result(row, key_name, digit_count)
#                 result_data[image_name][key_name] = value

#     # Save JSON
#     with open(edge_json_path, "w") as f:
#         json.dump(result_data, f, indent=2)
#     logger.info(f"Edge results saved to JSON: {edge_json_path}")

#     # CSV column headers
#     all_qs = sorted({qid for qmap in result_data.values() for qid in qmap.keys() if qid.startswith("Q")},
#                     key=lambda x: int(x[1:]))
#     key_columns = list(key_fields.keys())

#     # Save CSV
#     with open(edge_csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Image Name"] + all_qs + key_columns)
#         for image_name, qmap in result_data.items():
#             row = [image_name] + [qmap.get(q, "") for q in all_qs] + [qmap.get(k, "") for k in key_columns]
#             writer.writerow(row)
#     logger.info(f"Edge results saved to CSV: {edge_csv_path}")

#     # Human-readable output
#     human_result_data = {}
#     for img, data in result_data.items():
#         human_result_data[img] = {}
#         for k, v in data.items():
#             human_result_data[img][key_fields.get(k, k)] = v

#     human_json_path = os.path.splitext(edge_json_path)[0] + "_human.json"
#     with open(human_json_path, "w") as f:
#         json.dump(human_result_data, f, indent=2)
#     logger.info(f"Human-readable JSON saved to: {human_json_path}")

#     human_csv_path = os.path.splitext(edge_csv_path)[0] + "_human.csv"
#     with open(human_csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Image Name"] + all_qs + [key_fields[k] for k in key_columns])
#         for img_name, qmap in human_result_data.items():
#             row = [img_name] + [qmap.get(q, "") for q in all_qs] + [qmap.get(key_fields[k], "") for k in key_columns]
#             writer.writerow(row)
#     logger.info(f"Human-readable CSV saved to: {human_csv_path}")

def evaluate_edge_cases(verification_csv_path, edge_json_path, edge_csv_path,
                        key_fields_json, classes_file, logger, omr_template_name):

    logger.info("Evaluating edge cases...")
    logger.info(f"Verification CSV path: {verification_csv_path}")

    # Load key field metadata
    with open(key_fields_json, "r") as f:
        key_fields = json.load(f)
    with open(classes_file, "r") as f:
        class_labels = f.read().splitlines()

    # Count digits per key from classes.txt
    key_digit_counts = defaultdict(int)
    for label in class_labels:
        m = re.match(r"^(key\d+)_(\d+)$", label)
        if m:
            key, digit_idx = m.groups()
            digit_idx = int(digit_idx)
            key_digit_counts[key] = max(key_digit_counts[key], digit_idx + 1)
    logger.debug(f"Key digit counts: {key_digit_counts}")

    # Load verification CSV
    df = pd.read_csv(verification_csv_path)
    result_data = {}

    # Identify question result columns
    result_cols = [col for col in df.columns if re.match(r"^Result \d{1,2}[A-Da-d]$", col)]
    question_ids = sorted(set(col.split()[1][:-1] for col in result_cols), key=lambda x: int(x))
    logger.info(f"Detected question IDs: {question_ids}")

    # --- Updated helper for keys ---
    def extract_number_from_result(row, key_name, digit_count):
        digit_outputs = []
        for d in range(digit_count):
            fully_marked, partially_marked = [], []
            for val in range(10):
                col = f"{key_name}_{d}_{val}"   # <-- no "Result" prefix now
                if col in row:
                    try:
                        pct = float(row[col])
                        if pct < 185:  # fully marked bubble
                            fully_marked.append(f"{key_name}_{d}_{val}")
                        elif 185 <= pct <= 242:  # partial marking
                            partially_marked.append(f"{key_name}_{d}_{val}")
                    except:
                        continue

            # Decision for each digit
            if len(fully_marked) == 1 and not partially_marked:
                digit_outputs.append(fully_marked[0])
            elif len(fully_marked) > 1 or (fully_marked and partially_marked):
                digit_outputs.append("|".join(fully_marked + partially_marked))
            elif not fully_marked and len(partially_marked) == 1:
                digit_outputs.append(partially_marked[0])
            elif not fully_marked and len(partially_marked) > 1:
                digit_outputs.append("|".join(partially_marked))
            else:
                digit_outputs.append(" ")
        return "||".join(digit_outputs)

    for idx, row in df.iterrows():
        image_name = row["Image Name"]
        result_data[image_name] = {}

        # ------------------------
        # Block 1: Handle Questions
        # ------------------------
        for qid in question_ids:
            fully_marked, partially_marked = [], []
            for opt in ['A', 'B', 'C', 'D']:
                mean_col = f"{qid}{opt}"
                if mean_col in row:
                    try:
                        mean_val = float(row[mean_col])
                        if mean_val < 185:
                            fully_marked.append(opt)
                        elif 185 <= mean_val <= 242:
                            partially_marked.append(opt)
                    except:
                        continue

            if len(fully_marked) == 1 and len(partially_marked) == 0:
                result_data[image_name][f"Q{qid}"] = fully_marked[0]
            elif len(fully_marked) > 1 or (fully_marked and partially_marked):
                result_data[image_name][f"Q{qid}"] = "*"
            elif len(fully_marked) == 0 and len(partially_marked) == 1:
                result_data[image_name][f"Q{qid}"] = "?"
            elif len(fully_marked) == 0 and len(partially_marked) > 1:
                result_data[image_name][f"Q{qid}"] = "*"
            else:
                result_data[image_name][f"Q{qid}"] = ""

        # ------------------------
        # Block 2: Handle Key Fields
        # ------------------------
        for key_name, human_name in key_fields.items():
            digit_count = key_digit_counts.get(key_name, 0)
            if digit_count > 0:
                value = extract_number_from_result(row, key_name, digit_count)
                result_data[image_name][key_name] = value

    # Save JSON
    with open(edge_json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    logger.info(f"Edge results saved to JSON: {edge_json_path}")

    # CSV column headers
    all_qs = sorted({qid for qmap in result_data.values() for qid in qmap.keys() if qid.startswith("Q")},
                    key=lambda x: int(x[1:]))
    key_columns = list(key_fields.keys())

    # Save CSV
    with open(edge_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name"] + all_qs + key_columns)
        for image_name, qmap in result_data.items():
            row = [image_name] + [qmap.get(q, "") for q in all_qs] + [qmap.get(k, "") for k in key_columns]
            writer.writerow(row)
    logger.info(f"Edge results saved to CSV: {edge_csv_path}")

    # Human-readable output
    human_result_data = {}
    for img, data in result_data.items():
        human_result_data[img] = {}
        for k, v in data.items():
            human_result_data[img][key_fields.get(k, k)] = v

    human_json_path = os.path.splitext(edge_json_path)[0] + "_human.json"
    with open(human_json_path, "w") as f:
        json.dump(human_result_data, f, indent=2)
    logger.info(f"Human-readable JSON saved to: {human_json_path}")

    human_csv_path = os.path.splitext(edge_csv_path)[0] + "_human.csv"
    with open(human_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name"] + all_qs + [key_fields[k] for k in key_columns])
        for img_name, qmap in human_result_data.items():
            row = [img_name] + [qmap.get(q, "") for q in all_qs] + [qmap.get(key_fields[k], "") for k in key_columns]
            writer.writerow(row)
    logger.info(f"Human-readable CSV saved to: {human_csv_path}")

    # ------------------------------------------------------------
    # Internal Function for Key Conversion & Regeneration
    # ------------------------------------------------------------
    def convert_keys(json_path, omr_template_name):
        if omr_template_name != "HSOMR":
            logger.error(f"Batch name is not {omr_template_name}. Exiting conversion.")
            sys.exit(1)

        with open(json_path, "r") as f:
            data = json.load(f)

        for img, values in data.items():
            for key_name, val in values.items():
                if key_name.startswith("key"):
                    digits = []
                    for digit_str in val.split("||"):
                        if digit_str.strip() == "":
                            digits.append(" ")
                        elif "|" in digit_str:  # multiple marked
                            digits.append("*")
                        else:
                            digits.append(digit_str.split("_")[-1])
                    values[key_name] = "".join(digits)

        # overwrite same JSON
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Keys converted and overwritten: {json_path}")

        # regenerate CSV & human JSON/CSV
        all_qs = sorted({qid for qmap in data.values() for qid in qmap.keys() if qid.startswith("Q")},
                        key=lambda x: int(x[1:]))
        key_columns = list(key_fields.keys())

        # CSV
        with open(edge_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Name"] + all_qs + key_columns)
            for image_name, qmap in data.items():
                row = [image_name] + [qmap.get(q, "") for q in all_qs] + [qmap.get(k, "") for k in key_columns]
                writer.writerow(row)
        logger.info(f"Converted CSV regenerated: {edge_csv_path}")

        # human-readable JSON
        human_data = {}
        for img, values in data.items():
            human_data[img] = {}
            for k, v in values.items():
                human_data[img][key_fields.get(k, k)] = v

        human_json_path = os.path.splitext(edge_json_path)[0] + "_human.json"
        with open(human_json_path, "w") as f:
            json.dump(human_data, f, indent=2)
        logger.info(f"Converted human-readable JSON saved: {human_json_path}")

        human_csv_path = os.path.splitext(edge_csv_path)[0] + "_human.csv"
        with open(human_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Name"] + all_qs + [key_fields[k] for k in key_columns])
            for img_name, qmap in human_data.items():
                row = [img_name] + [qmap.get(q, "") for q in all_qs] + [qmap.get(key_fields[k], "") for k in key_columns]
                writer.writerow(row)
        logger.info(f"Converted human-readable CSV saved: {human_csv_path}")

    # ---- Call the conversion at correct place ----
    convert_keys(edge_json_path, omr_template_name)

def generate_generalized_json(base_json_path, ed_results_json, verification_csv_path,
                              generalized_json_path, key_fields_json, classes_file, logger):
    logger.info("Generating generalized JSON...")
    logger.info(f"Base JSON path: {base_json_path}")

    with open(base_json_path, 'r') as f:
        code2_data = json.load(f)
    with open(ed_results_json, 'r') as f:
        ed_results = json.load(f)
    df_ver = pd.read_csv(verification_csv_path)
    df_ver.set_index("Image Name", inplace=True)
    with open(key_fields_json, 'r') as f:
        key_fields = json.load(f)

    key_list = list(key_fields.keys())

    for image in code2_data["IMAGES"]:
        img_name = image["IMAGENAME"].split("\\")[-1]
        ed_data = ed_results.get(img_name, {})
        ver_data = df_ver.loc[img_name] if img_name in df_ver.index else None

        for field in image["FIELDS"]:
            field_name = field["FIELD"]
            value, confidence, success = "", "", "Y"

            if field_name in key_list:
                value = ed_data.get(field_name, "")
                confidence = "100" if value else ""
                success = "Y" if value else "N"
            elif "question" in field_name.lower():
                qnum = field_name.lower().replace("question_", "Q")
                value = ed_data.get(qnum, "")
                if not value or "*" in value:
                    success = "N"
                else:
                    if ver_data is not None:
                        col_name = f"Result {qnum[1:]}{value}"
                        confidence = str(ver_data[col_name]) if col_name in ver_data.index else ""
                        if confidence == "":
                            success = "N"
            else:
                success = "N"

            field["FIELDDATA"] = value
            field["CONFIDENCE"] = confidence
            field["SUCCESS"] = success
        logger.debug(f"Updated fields for {img_name}")

    with open(generalized_json_path, "w") as f:
        json.dump(code2_data, f, indent=4)
    logger.info(f"Generalized JSON saved at {generalized_json_path}")

def draw_marked_bboxes(processed_images_folder, verification_csv_path,
                       field_mappings, result_images, logger):
    logger.info("Drawing marked bounding boxes on images...")
    os.makedirs(result_images, exist_ok=True)

    with open(field_mappings, "r") as f:
        bbox_data = json.load(f)
    df = pd.read_csv(verification_csv_path)
    df.set_index("Image Name", inplace=True)

    option_pattern = re.compile(r"^(\d{1,2})([A-Da-d])$")
    key_pattern = re.compile(r"^key\d+_\d+_\d+$")

    for img_name, mapping in bbox_data.items():
        if mapping.get("status", "").lower() != "processed":
            continue

        img_path = os.path.join(processed_images_folder, img_name)
        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            logger.error(f"Unable to load image: {img_path}")
            continue

        # --- Block 1: Handle Question Options ---
        for label, field_data in mapping.get("mapped_fields", {}).items():
            if option_pattern.match(label):
                qnum = option_pattern.match(label).group(1)
                opt = option_pattern.match(label).group(2).upper()
                mean_col = f"{qnum}{opt}"

                intensity = None
                if img_name in df.index and mean_col in df.columns:
                    try:
                        intensity = float(df.loc[img_name, mean_col])
                    except:
                        intensity = None

                if intensity is not None:
                    color = (0, 255, 0) if intensity < 185 else (0, 255, 255) if 185 <= intensity <= 242 else None
                    if color:
                        x1, y1, x2, y2 = field_data["bbox"]
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # --- Block 2: Handle Key Fields ---
        for label, field_data in mapping.get("mapped_fields", {}).items():
            if key_pattern.match(label):
                intensity = None
                if img_name in df.index and label in df.columns:
                    try:
                        intensity = float(df.loc[img_name, label])
                    except:
                        intensity = None

                if intensity is not None:
                    color = (0, 255, 0) if intensity < 185 else (0, 255, 255) if 185 <= intensity <= 242 else None
                    if color:
                        x1, y1, x2, y2 = field_data["bbox"]
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        output_path = os.path.join(result_images, img_name)
        cv2.imwrite(output_path, image)
        logger.info(f"Annotated image saved: {output_path}")

    
def process_marked_options(base_folder, omr_template_name, date, batch_name, draw_bboxes=True):
    directory_name = os.path.join(
        base_folder, "Images", omr_template_name, date, "Output", batch_name, "options_" + batch_name
    )
    os.makedirs(directory_name, exist_ok=True)

    logger, log_path = setup_logger(batch_name)
    logger.info(f"Starting marked options processing for batch {batch_name}")

    mapped_json_path = os.path.join(
        base_folder, "Images", omr_template_name, date, "Output", batch_name,
        "annotate_" + batch_name, "field_mappings.json"
    )
    processed_images_folder = os.path.join(
        base_folder, "Images", omr_template_name, date, "Output", batch_name,
        f"raw_{batch_name}"
    )
    key_fields_json = os.path.join(base_folder, "Annotations", omr_template_name, "key_fields.json")
    classes_file = os.path.join(base_folder, "Annotations", omr_template_name, "classes.txt")

    verification_csv_path = os.path.join(directory_name, "verification.csv")

    logger.info("Detecting marked options...")
    option_score_map = detect_marked_options(
        mapped_json_path,
        processed_images_folder,
        key_fields_json,
        classes_file,
        logger
    )

    logger.info("Exporting verification CSV...")
    export_verification_csv(option_score_map, verification_csv_path, key_fields_json, classes_file)

    edge_json_path = os.path.join(directory_name, "ed_results.json")
    edge_csv_path = os.path.join(directory_name, "ed_results.csv")
    logger.info("Evaluating edge cases...")
    evaluate_edge_cases(verification_csv_path, edge_json_path, edge_csv_path,
                        key_fields_json, classes_file, logger, omr_template_name)

    base_json_path = os.path.join(
        base_folder, "Images", omr_template_name, date, "Output", batch_name, f"{batch_name}.json"
    )
    generalized_json_path = base_json_path
    logger.info("Generating generalized JSON...")
    generate_generalized_json(
        base_json_path, edge_json_path, verification_csv_path,
        generalized_json_path, key_fields_json, classes_file, logger
    )

    if draw_bboxes:   # <-- toggle check
        field_mappings = os.path.join(
            base_folder, "Images", omr_template_name, date, "Output", batch_name,
            "annotate_" + batch_name, "field_mappings.json"
        )
        result_images = os.path.join(directory_name, "Results")
        logger.info("Drawing marked bounding boxes...")
        draw_marked_bboxes(processed_images_folder, verification_csv_path, field_mappings, result_images, logger)
    else:
        logger.info("Skipping drawing of marked bounding boxes as per flag setting")

    processed_images_count = len(option_score_map)
    total_detected_fields = sum(len(v) for v in option_score_map.values())
    logger.info(f"Marked options completed: {processed_images_count} images, {total_detected_fields} fields")

    return {
        "processed_images": processed_images_count,
        "total_detected_fields": total_detected_fields
    }