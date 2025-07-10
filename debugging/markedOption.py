import os
import cv2
import json
import numpy as np
import csv
import re
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

def export_verification_csv(option_score_map, output_csv_path):
    import csv

    question_ids = [f"Q{i}" for i in range(1, 11)]
    options = ["A", "B", "C", "D"]

    header = ["Image Name"]
    for q in question_ids:
        qnum = q[1:]
        for opt in options:
            header.append(f"{qnum}{opt}")
            header.append(f"Result {qnum}{opt}")

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for img_name in sorted(option_score_map.keys()):
            row = [img_name]
            score_map = option_score_map[img_name]

            for q in question_ids:
                qnum = q[1:]
                intensities = []
                for opt in options:
                    label = f"{qnum}{opt}"
                    label_lower = f"{qnum.lower()}{opt.lower()}"
                    label_full = f"question_{qnum}_{opt.lower()}"
                    
                    # Try multiple label formats
                    score = (
                        score_map.get(f"{qnum}{opt}") or
                        score_map.get(label_lower) or
                        score_map.get(label_full) or
                        ""
                    )
                    intensities.append((opt, score))

                opt_scores = {
                    opt: score for opt, score in intensities if isinstance(score, (int, float))
                }

                sorted_opts = sorted(opt_scores.items(), key=lambda x: x[1])  # lowest score = darkest

                for opt in options:
                    score = opt_scores.get(opt, "")
                    row.append(score)

                    if not isinstance(score, (int, float)) or len(sorted_opts) < 2:
                        row.append("")
                        continue

                    current_idx = next((i for i, (o, _) in enumerate(sorted_opts) if o == opt), None)

                    if current_idx is None or current_idx + 1 >= len(sorted_opts):
                        row.append(f"100.0 ({opt})")
                    else:
                        _, next_score = sorted_opts[current_idx + 1]
                        if next_score == 0:
                            row.append(f"100.0 ({opt})")
                        else:
                            pct = round((score / next_score) * 100, 2)
                            row.append(f"{pct} ({sorted_opts[current_idx + 1][0]})")

            writer.writerow(row)

    print(f"✅ Fixed verification CSV saved to: {output_csv_path}")


if __name__ == "__main__":
    mapped_json_path = r"D:\Projects\OMR\new_abhigyan\debugging\annotate_Test_Series\mapped_annotations.json"
    image_folder = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series"

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