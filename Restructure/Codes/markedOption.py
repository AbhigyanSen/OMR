import os
import cv2
import json
import numpy as np
import csv
import re
import pandas as pd
import sys
from collections import defaultdict

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
                          output_text_json_path, summary_json_path,
                          key_fields_json, classes_file):
    # Load dynamic key fields and digit counts
    with open(key_fields_json, 'r') as f:
        key_fields = json.load(f)
    digits_count = get_digits_count(classes_file)

    with open(mapped_json_path, 'r') as f:
        mapped_data = json.load(f)

    flat_results = {}
    summary_results = {}
    global_option_score_map = {}

    for img_name, annotations in mapped_data.items():
        if not isinstance(annotations, dict) or annotations.get("status", "").lower() != "processed":
            print(f"⛔ Skipping {img_name}: not marked as 'processed'.")
            continue

        image_path = os.path.join(processed_images_folder, img_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        question_groups = defaultdict(list)
        digit_groups = {k: defaultdict(list) for k in key_fields.keys()}
        option_score_map = {}
        summary_results[img_name] = {}
        global_option_score_map[img_name] = {}

        fields = annotations.get("mapped_fields", {})
        for label, data in fields.items():
            if not isinstance(data, dict) or "bbox" not in data:
                continue
            bbox = data["bbox"]

            # Question bubbles like 1A, 10D, etc.
            if re.match(r"\d{1,2}[a-dA-D]$", label):
                qid = extract_question_id(label)
                if qid:
                    question_groups[qid].append((label, bbox))
            else:
                # Dynamic key fields
                for key_name in key_fields.keys():
                    if is_digit_option(label, key_name):
                        group_key = extract_digit_group(label, key_name)
                        digit_groups[key_name][group_key].append((label, bbox))

        # Detect marked options per question
        for qid in sorted(question_groups.keys(), key=lambda x: int(x[1:])):
            options = question_groups[qid]
            best_score = float("inf")
            marked_label = None

            for label, bbox in options:
                _, score = is_filled_option(image, bbox)
                option_score_map[label] = round(score, 2)
                if score < best_score:
                    best_score = score
                    marked_label = label

            if marked_label:
                flat_results[f"{img_name}_{qid}"] = marked_label
                summary_results[img_name][qid] = extract_option_code(marked_label)

        # Extract each key field number dynamically
        for key_name, display_name in key_fields.items():
            total_digits = digits_count.get(key_name, 0)
            number_value = extract_final_number(digit_groups[key_name], key_name,
                                                total_digits, image, option_score_map)
            flat_results[f"{img_name}_{key_name}"] = number_value
            summary_results[img_name][display_name.replace(" ", "")] = number_value

        global_option_score_map[img_name] = option_score_map
        print(f"✅ Processed: {img_name}")

    # Save outputs
    with open(output_text_json_path, 'w') as f:
        json.dump(flat_results, f, indent=2)
    with open(summary_json_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n✅ Summary saved to: {summary_json_path}")
    print(f"✅ Flat mapping saved to: {output_text_json_path}")
    return global_option_score_map

def clean_and_export_summary(marked_options_path, summary_json_path, summary_csv_path,
                             key_fields_json, classes_file):
    import re
    from collections import defaultdict

    # Load marked options
    with open(marked_options_path, 'r') as f:
        flat_data = json.load(f)

    # Load dynamic key fields
    with open(key_fields_json, 'r') as f:
        key_fields = json.load(f)

    digit_counts = get_digits_count(classes_file)  # helper to count digit groups

    summary_dict = defaultdict(dict)

    for full_key, value in flat_data.items():
        if '_' not in full_key or not value:
            continue

        img_name, label = full_key.split("_", 1)

        # Handle questions (only Q-prefixed, safe)
        if re.match(r"^Q\d+$", label):
            summary_dict[img_name][label] = value
            continue

        # Handle dynamic keys (like key0_0)
        for key_name in key_fields.keys():
            if label.startswith(key_name) and value.isdigit():
                summary_dict[img_name][label] = value

    # Combine digits for each key (final values, but keep raw key name like key0)
    for img_name in list(summary_dict.keys()):
        for key_name in key_fields.keys():
            digit_count = digit_counts.get(key_name, 0)
            # print(f"DEBUG: {img_name} -> {key_name} digit_count={digit_count}")
            # If full key already exists, use it; otherwise, assemble from digits
            final_value = summary_dict[img_name].get(key_name, "")
            if not final_value:
                final_value = "".join(
                    summary_dict[img_name].get(f"{key_name}_{i}", "")
                    for i in range(digit_count)
                )
            summary_dict[img_name][key_name] = final_value
        
    # Save summary JSON
    with open(summary_json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"✅ Cleaned summary saved to: {summary_json_path}")

    # Collect question columns
    all_questions = sorted(
        {q for q_data in summary_dict.values() for q in q_data.keys() if q.startswith("Q")},
        key=lambda x: int(x[1:])
    )

    # CSV header = Image + Q columns + key0, key1...
    csv_header = ["Image Name"] + all_questions + list(key_fields.keys())

    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for img_name, q_answers in summary_dict.items():
            row = [img_name] + [q_answers.get(q, "") for q in all_questions]
            for key_name in key_fields.keys():
                row.append(q_answers.get(key_name, ""))
            writer.writerow(row)
    print(f"📄 Summary CSV saved to: {summary_csv_path}")

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

def evaluate_edge_cases(verification_csv_path, edge_json_path, edge_csv_path, key_fields_json, classes_file):
    
    # ---- Load keys ----
    with open(key_fields_json, "r") as f:
        key_fields = json.load(f)
    with open(classes_file, "r") as f:
        class_labels = f.read().splitlines()

    # ---- Determine number of digits per key ----
    key_digit_counts = defaultdict(int)
    for label in class_labels:
        m = re.match(r"^(key\d+)_(\d+)$", label)
        if m:
            key, digit_idx = m.groups()
            digit_idx = int(digit_idx)
            key_digit_counts[key] = max(key_digit_counts[key], digit_idx + 1)

    df = pd.read_csv(verification_csv_path)
    result_data = {}

    # ---- Detect question columns dynamically ----
    result_cols = [col for col in df.columns if re.match(r"^Result \d{1,2}[A-Da-d]$", col)]
    question_ids = sorted(set(col.split()[1][:-1] for col in result_cols), key=lambda x: int(x))

    def extract_number_from_result(row, key_name, digit_count):
        digits = []
        for d in range(digit_count):
            min_val = float('inf')
            selected_digit = ''
            for val in range(10):
                col = f"Result {key_name}_{d}_{val}"
                if col in row:
                    try:
                        pct = float(row[col])
                        if pct < min_val:
                            min_val = pct
                            selected_digit = str(val)
                    except:
                        continue
            digits.append(selected_digit)
        return ''.join(digits)

    # ---- Process each row ----
    for idx, row in df.iterrows():
        image_name = row["Image Name"]
        result_data[image_name] = {}

        # ---- Questions ----
        for qid in question_ids:
            marked_options = []
            for opt in ['A', 'B', 'C', 'D']:
                col = f"Result {qid}{opt}"
                if col in row:
                    try:
                        pct = float(row[col])
                        if pct < 85.0:
                            marked_options.append(opt)
                    except:
                        continue

            if len(marked_options) == 0:
                result_data[image_name][f"Q{qid}"] = ""
            elif len(marked_options) == 1:
                result_data[image_name][f"Q{qid}"] = marked_options[0]
            else:
                result_data[image_name][f"Q{qid}"] = "*".join(marked_options)

        # ---- Dynamic keys ----
        for key_name, human_name in key_fields.items():
            digit_count = key_digit_counts.get(key_name, 0)
            if digit_count > 0:
                result_data[image_name][key_name] = extract_number_from_result(row, key_name, digit_count)

    # ---- Save JSON ----
    with open(edge_json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"✅ Edge results saved to JSON: {edge_json_path}")

    # ---- Save CSV ----
    all_qs = sorted({qid for qmap in result_data.values() for qid in qmap.keys() if qid.startswith("Q")},
                    key=lambda x: int(x[1:]))

    key_columns = list(key_fields.keys())  # keep key0, key1 ... as columns

    with open(edge_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name"] + all_qs + key_columns)
        for image_name, qmap in result_data.items():
            row = [image_name] + [qmap.get(q, "") for q in all_qs] + [qmap.get(k, "") for k in key_columns]
            writer.writerow(row)
    print(f"📄 Edge results saved to CSV: {edge_csv_path}")
    
    # ---------- POST-PROCESS: Create human-readable versions ----------
    # JSON with human-readable keys
    human_result_data = {}
    for img, data in result_data.items():
        human_result_data[img] = {}
        for k, v in data.items():
            human_result_data[img][key_fields.get(k, k)] = v  # replace key0 -> Roll Number etc.

    human_json_path = os.path.splitext(edge_json_path)[0] + "_human.json"
    with open(human_json_path, "w") as f:
        json.dump(human_result_data, f, indent=2)
    print(f"📝 Human-readable JSON saved to: {human_json_path}")

    # CSV with human-readable column headers
    human_csv_path = os.path.splitext(edge_csv_path)[0] + "_human.csv"
    with open(human_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name"] + all_qs + [key_fields[k] for k in key_columns])
        for img_name, qmap in human_result_data.items():
            row = [img_name] + [qmap.get(q, "") for q in all_qs] + [qmap.get(key_fields[k], "") for k in key_columns]
            writer.writerow(row)
    print(f"📝 Human-readable CSV saved to: {human_csv_path}")
    
# Generate a generalized JSON file for the batch ---------------------------------------------------------
def generate_generalized_json(base_json_path, ed_results_json, verification_csv_path, generalized_json_path, key_fields_json, classes_file):

    # ---- Load inputs ----
    with open(base_json_path, 'r') as f:
        code2_data = json.load(f)
    with open(ed_results_json, 'r') as f:
        ed_results = json.load(f)
    df_ver = pd.read_csv(verification_csv_path)
    df_ver.set_index("Image Name", inplace=True)
    with open(key_fields_json, 'r') as f:
        key_fields = json.load(f)  # e.g. {"key0": "Roll Number", "key1": "Question Booklet Number"}

    key_list = list(key_fields.keys())  # ["key0", "key1", ...]

    for image in code2_data["IMAGES"]:
        img_name = image["IMAGENAME"].split("\\")[-1]
        ed_data = ed_results.get(img_name, {})
        ver_data = df_ver.loc[img_name] if img_name in df_ver.index else None

        for field in image["FIELDS"]:
            field_name = field["FIELD"]  # e.g. "key0", "question_12"
            value = ""
            confidence = ""
            success = "Y"

            # ---------- Direct key handling ----------
            if field_name in key_list:
                value = ed_data.get(field_name, "")
                confidence = "100" if value else ""
                success = "Y" if value else "N"

            # ---------- Question handling ----------
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

            # ---------- Unknown fields ----------
            else:
                success = "N"

            field["FIELDDATA"] = value
            field["CONFIDENCE"] = confidence
            field["SUCCESS"] = success

    # ---- Save final JSON ----
    with open(generalized_json_path, "w") as f:
        json.dump(code2_data, f, indent=4)

    print(f"✅ Generalized JSON saved at {generalized_json_path}")

    
# MAIN BLOCK   
if __name__ == "__main__":
    # Define paths
    base_folder = r"D:\Projects\OMR\new_abhigyan\Restructure"
    
    # omr_template_name = "HSOMR"
    # date = "23072025"
    # batch_name = "Batch003"   
    # Expect arguments: omr_template_name, date, batch_name
    
    # Inputs from Command Line
    if len(sys.argv) != 4:
        print("Usage: python AnchorDetection.py <omr_template_name> <date> <batch_name>")
        sys.exit(1)

    omr_template_name, date, batch_name = sys.argv[1:4] 
    
    mapped_json_path = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, "annotate_" + batch_name, "field_mappings.json")
    processed_images_folder = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, f"processed_{batch_name}")
    # NEW: Paths for dynamic config files
    key_fields_json = os.path.join(base_folder, "Annotations", omr_template_name, "key_fields.json")
    classes_file = os.path.join(base_folder, "Annotations", omr_template_name, "classes.txt")
    
    directory_name = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, "options_" + batch_name)
    os.makedirs(directory_name, exist_ok=True)
    
    output_text_json_path = os.path.join(directory_name, "marked_options.json")
    summary_json_path = os.path.join(directory_name, "summary.json")
    summary_csv_path = os.path.join(directory_name, "summary.csv")
    verification_csv_path = os.path.join(directory_name, "verification.csv")

    option_score_map = detect_marked_options(
        mapped_json_path,
        processed_images_folder,
        output_text_json_path,
        summary_json_path,
        key_fields_json,
        classes_file
    )

    clean_and_export_summary(output_text_json_path, summary_json_path, summary_csv_path, key_fields_json, classes_file)
    export_verification_csv(option_score_map, verification_csv_path, key_fields_json, classes_file)
    
    edge_json_path = os.path.join(directory_name, "ed_results.json")
    edge_csv_path = os.path.join(directory_name, "ed_results.csv")

    # evaluate_edge_cases(verification_csv_path, edge_json_path, edge_csv_path)
    evaluate_edge_cases(verification_csv_path, edge_json_path, edge_csv_path, key_fields_json, classes_file)
    
    base_json_path = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, f"{batch_name}.json")
    ed_results_json = edge_json_path
    verification_csv_path = verification_csv_path
    generalized_json_path = base_json_path
    generate_generalized_json(base_json_path, ed_results_json, verification_csv_path, generalized_json_path, key_fields_json, classes_file)