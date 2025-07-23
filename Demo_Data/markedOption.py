import os
import cv2
import json
import numpy as np
import csv
import re
import pandas as pd
from collections import defaultdict

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

def detect_marked_options(mapped_json_path, image_folder, output_text_json_path, summary_json_path):
    with open(mapped_json_path, 'r') as f:
        mapped_data = json.load(f)

    flat_results = {}
    summary_results = {}
    global_option_score_map = {}

    for img_name, annotations in mapped_data.items():
        if (
            "error" in annotations
            or not isinstance(annotations, dict)
            or not annotations.get("valid_for_marked_option", False)
        ):
            print(f"⛔ Skipping {img_name}: invalid or not marked for processing.")
            continue

        image_path = os.path.join(image_folder, img_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        question_groups = defaultdict(list)
        regno_groups = defaultdict(list)
        rollno_groups = defaultdict(list)
        booklet_groups = defaultdict(list)

        option_score_map = {}
        summary_results[img_name] = {}
        global_option_score_map[img_name] = {}

        for label, data in annotations.items():
            if not isinstance(data, dict) or "bbox" not in data:
                continue
            bbox = data["bbox"]

            if re.match(r"\d{1,2}[a-dA-D]$", label):  # <-- ✅ Match options like '1A', '10D', etc.
                qid = extract_question_id(label)
                if qid:
                    question_groups[qid].append((label, bbox))
            elif is_digit_option(label, "reg_no"):
                key = extract_digit_group(label, "reg_no")
                regno_groups[key].append((label, bbox))
            elif is_digit_option(label, "roll_no"):
                key = extract_digit_group(label, "roll_no")
                rollno_groups[key].append((label, bbox))
            elif is_digit_option(label, "booklet_no"):
                key = extract_digit_group(label, "booklet_no")
                booklet_groups[key].append((label, bbox))

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

        def extract_final_number(group_dict, prefix, total_digits):
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

                flat_results[f"{img_name}_{group_key}"] = marked_digit
                number_digits.append(marked_digit or "")
            return "".join(number_digits)

        reg_number = extract_final_number(regno_groups, "reg_no", 10)
        roll_number = extract_final_number(rollno_groups, "roll_no", 10)
        booklet_number = extract_final_number(booklet_groups, "booklet_no", 9)

        summary_results[img_name]["RegistrationNumber"] = reg_number
        summary_results[img_name]["RollNumber"] = roll_number
        summary_results[img_name]["BookletNumber"] = booklet_number
        global_option_score_map[img_name] = option_score_map

        print(f"✅ Processed: {img_name} | Reg: {reg_number}, Roll: {roll_number}, Booklet: {booklet_number}")

    with open(output_text_json_path, 'w') as f:
        json.dump(flat_results, f, indent=2)

    with open(summary_json_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n✅ Summary saved to: {summary_json_path}")
    print(f"✅ Flat mapping saved to: {output_text_json_path}")
    return global_option_score_map

def clean_and_export_summary(marked_options_path, summary_json_path, summary_csv_path):
    with open(marked_options_path, 'r') as f:
        flat_data = json.load(f)

    summary_dict = defaultdict(dict)

    for full_key, value in flat_data.items():
        if '_' not in full_key or not value:
            continue

        img_name, label = full_key.split("_", 1)
        if label.startswith("Q"):
            summary_dict[img_name][label] = value
        elif label.startswith("reg_no") and value.isdigit():
            summary_dict[img_name][label] = value
        elif label.startswith("roll_no") and value.isdigit():
            summary_dict[img_name][label] = value
        elif label.startswith("booklet_no") and value.isdigit():
            summary_dict[img_name][label] = value

    for img_name in summary_dict.keys():
        summary_dict[img_name]["RegistrationNumber"] = "".join(
            summary_dict[img_name].get(f"reg_no_{i}", "") for i in range(10)
        )
        summary_dict[img_name]["RollNumber"] = "".join(
            summary_dict[img_name].get(f"roll_no_{i}", "") for i in range(10)
        )
        summary_dict[img_name]["BookletNumber"] = "".join(
            summary_dict[img_name].get(f"booklet_no_{i}", "") for i in range(9)
        )

    with open(summary_json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"✅ Cleaned summary saved to: {summary_json_path}")

    all_questions = sorted(
        {q for q_data in summary_dict.values() for q in q_data.keys() if q.startswith("Q")},
        key=lambda x: int(x[1:])
    )

    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name"] + all_questions + ["RegistrationNumber", "RollNumber", "BookletNumber"])
        for img_name, q_answers in summary_dict.items():
            row = [img_name] + [q_answers.get(q, "") for q in all_questions] + [
                q_answers.get("RegistrationNumber", ""),
                q_answers.get("RollNumber", ""),
                q_answers.get("BookletNumber", "")
            ]
            writer.writerow(row)
    print(f"📄 Summary CSV saved to: {summary_csv_path}")

import csv
import re
from collections import defaultdict

def export_verification_csv(option_score_map, output_csv_path):
    question_pattern = re.compile(r"^(\d{1,2})([A-Da-d])$")
    digit_pattern = re.compile(r"^(reg_no|roll_no|booklet_no)_(\d)_(\d)$")

    all_labels = set()
    for score_map in option_score_map.values():
        all_labels.update(score_map.keys())

    question_map = defaultdict(list)
    digit_map = defaultdict(lambda: defaultdict(list))  # {prefix: {digit_idx: [label]}}

    for label in all_labels:
        qmatch = question_pattern.match(label)
        dmatch = digit_pattern.match(label)
        if qmatch:
            qnum, opt = qmatch.groups()
            question_map[qnum].append(f"{qnum}{opt.upper()}")
        elif dmatch:
            prefix, digit_idx, val = dmatch.groups()
            digit_map[prefix][int(digit_idx)].append(label)

    sorted_questions = sorted(question_map.keys(), key=lambda x: int(x))

    header = ["Image Name"]

    # Add question columns
    for q in sorted_questions:
        for opt in ["A", "B", "C", "D"]:
            label = f"{q}{opt}"
            header.append(label)
            header.append(f"Result {label}")

    # Add digit columns for reg_no, roll_no, booklet_no
    for prefix in ["reg_no", "roll_no", "booklet_no"]:
        max_digits = 10 if prefix in ["reg_no", "roll_no"] else 9
        for d in range(max_digits):
            for val in range(10):
                label = f"{prefix}_{d}_{val}"
                header.append(label)
                header.append(f"Result {label}")

    # Write the CSV
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for img_name in sorted(option_score_map.keys()):
            row = [img_name]
            score_map = option_score_map[img_name]

            # Process Questions
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

            # Process Digit Columns (reg_no, roll_no, booklet_no)
            for prefix in ["reg_no", "roll_no", "booklet_no"]:
                max_digits = 10 if prefix in ["reg_no", "roll_no"] else 9
                for d in range(max_digits):
                    scores = {}
                    for val in range(10):
                        label = f"{prefix}_{d}_{val}"
                        try:
                            scores[val] = float(score_map.get(label, ""))
                        except:
                            pass

                    max_score = max(scores.values()) if scores else 1.0

                    for val in range(10):
                        label = f"{prefix}_{d}_{val}"
                        raw = score_map.get(label, "")
                        row.append(raw)
                        try:
                            v = float(raw)
                            pct = round((v / max_score) * 100, 2) if max_score else ""
                            row.append(f"{pct}")
                        except:
                            row.append("")

            writer.writerow(row)

    print(f"✅ Final verification.csv written to: {output_csv_path}")

def evaluate_edge_cases(verification_csv_path, output_json_path, output_csv_path):
    df = pd.read_csv(verification_csv_path)
    result_data = {}

    # Identify all 'Result XX' columns like 'Result 1A', 'Result 2B', etc.
    result_cols = [col for col in df.columns if re.match(r"^Result \d{1,2}[A-Da-d]$", col)]
    question_ids = sorted(set(col.split()[1][:-1] for col in result_cols), key=lambda x: int(x))

    for idx, row in df.iterrows():
        image_name = row["Image Name"]
        result_data[image_name] = {}

        for qid in question_ids:
            marked_options = []
            for opt in ['A', 'B', 'C', 'D']:
                col = f"Result {qid}{opt}"
                if col in row:
                    try:
                        pct = float(row[col])
                        if pct < 90.0:
                            marked_options.append(opt)
                    except:
                        continue

            if len(marked_options) == 0:
                result_data[image_name][f"Q{qid}"] = ""
            elif len(marked_options) == 1:
                result_data[image_name][f"Q{qid}"] = marked_options[0]
            else:
                result_data[image_name][f"Q{qid}"] = "|".join(marked_options)

    # 🔸 Save to JSON
    with open(output_json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"✅ Edge results saved to JSON: {output_json_path}")

    # 🔸 Save to CSV
    all_qs = [f"Q{i}" for i in range(1, 41)]  # Fixed 40 questions
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name"] + all_qs)
        for image_name, qmap in result_data.items():
            row = [image_name] + [qmap.get(q, "") for q in all_qs]
            writer.writerow(row)
    print(f"📄 Edge results saved to CSV: {output_csv_path}")


if __name__ == "__main__":
    mapped_json_path = r"D:\Projects\OMR\new_abhigyan\Demo_Data\annotate_Test_Series\mapped_annotations.json"
    image_folder = r"D:\Projects\OMR\new_abhigyan\Demo_Data\TestData\Test_Series"

    output_text_json_path = os.path.join(image_folder, "marked_options.json")
    summary_json_path = os.path.join(image_folder, "summary.json")
    summary_csv_path = os.path.join(image_folder, "summary.csv")
    verification_csv_path = os.path.join(image_folder, "verification.csv")

    option_score_map = detect_marked_options(
        mapped_json_path,
        image_folder,
        output_text_json_path,
        summary_json_path
    )

    clean_and_export_summary(output_text_json_path, summary_json_path, summary_csv_path)
    export_verification_csv(option_score_map, verification_csv_path)
    
    edge_json_path = os.path.join(image_folder, "ed_results.json")
    edge_csv_path = os.path.join(image_folder, "ed_results.csv")

    evaluate_edge_cases(verification_csv_path, edge_json_path, edge_csv_path)
