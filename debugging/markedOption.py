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
    match = re.match(r"q(\d+)", label.lower())
    return f"Q{match.group(1)}" if match else None


def extract_option_code(label):
    match = re.match(r"q(\d+)_([a-d])", label.lower())
    return match.group(2).upper() if match else None


def is_registration_digit_option(label):
    return bool(re.match(r"reg_no_d\d{1,2}_\d", label))


def extract_digit_group(label):
    match = re.match(r"reg_no_d(\d{1,2})_\d", label)
    return f"reg_no_d{match.group(1)}" if match else None


def extract_digit_value(label):
    match = re.match(r"reg_no_d\d{1,2}_(\d)", label)
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
        option_score_map = {}

        summary_results[img_name] = {}
        global_option_score_map[img_name] = {}

        # Grouping
        for label, data in annotations.items():
            if not isinstance(data, dict) or "bbox" not in data:
                continue
            if label.startswith("q") and "_" in label:
                qid = extract_question_id(label)
                if qid:
                    question_groups[qid].append((label, data["bbox"]))
            elif is_registration_digit_option(label):
                digit_group = extract_digit_group(label)
                if digit_group:
                    regno_groups[digit_group].append((label, data["bbox"]))

        # Process Questions
        for qid in sorted(question_groups.keys(), key=lambda x: int(x[1:])):
            options = question_groups[qid]
            marked_label = None
            best_score = float("inf")

            for label, bbox in options:
                _, score = is_filled_option(image, bbox)
                option_score_map[label] = round(score, 2)

                if score < best_score:
                    best_score = score
                    marked_label = label

            if marked_label:
                flat_results[f"{img_name}_{qid}"] = marked_label
                option = extract_option_code(marked_label)
                summary_results[img_name][qid] = option

        # Process Registration Number
        reg_digits = []
        for i in range(1, 11):
            group_key = f"reg_no_d{i}"
            options = regno_groups.get(group_key, [])
            best_score = float("inf")
            marked_digit = ""

            for label, bbox in options:
                _, score = is_filled_option(image, bbox)
                option_score_map[label] = round(score, 2)
                if score < best_score:
                    best_score = score
                    marked_digit = extract_digit_value(label)

            flat_results[f"{img_name}_{group_key}"] = marked_digit
            reg_digits.append(marked_digit or "")

        reg_number = "".join(reg_digits)
        summary_results[img_name]["RegistrationNumber"] = reg_number

        global_option_score_map[img_name] = option_score_map

        print(f"✅ Processed: {img_name} | Reg. No: {reg_number}")

    with open(output_text_json_path, 'w') as f:
        json.dump(flat_results, f, indent=2)

    with open(summary_json_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n✅ Summary saved to:\n{summary_json_path}")
    print(f"✅ Flat mapping saved to:\n{output_text_json_path}")
    return global_option_score_map

def clean_and_export_summary(marked_options_path, summary_json_path, summary_csv_path):
    from collections import defaultdict
    import json, csv

    with open(marked_options_path, 'r') as f:
        flat_data = json.load(f)

    summary_dict = defaultdict(dict)

    for full_key, value in flat_data.items():
        if '_' not in full_key or not value:
            continue

        img_name, label = full_key.split("_", 1)

        if label.startswith("Q") and "_" not in label:
            qnum = label[1:]
            if len(value) >= 2 and value.startswith(f"q{qnum}_"):
                summary_dict[img_name][f"Q{qnum}"] = value.split("_")[-1].upper()

        elif label.startswith("reg_no_d") and value.isdigit():
            summary_dict[img_name][label] = value

    # Build Registration Number
    for img_name in summary_dict.keys():
        digits = [summary_dict[img_name].get(f"reg_no_d{i}", "") for i in range(1, 11)]
        summary_dict[img_name]["RegistrationNumber"] = "".join(digits)

    with open(summary_json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"✅ Cleaned summary saved to: {summary_json_path}")

    all_questions = sorted({q for q_data in summary_dict.values() for q in q_data.keys()
                            if q.startswith("Q")}, key=lambda x: int(x[1:]))

    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name"] + all_questions + ["RegistrationNumber"])
        for img_name, q_answers in summary_dict.items():
            row = [img_name] + [q_answers.get(q, "") for q in all_questions] + [q_answers.get("RegistrationNumber", "")]
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
                    label = f"q{qnum}_{opt.lower()}"
                    score = score_map.get(label, "")
                    intensities.append((opt, score))

                opt_scores = {opt: score for opt, score in intensities if isinstance(score, (int, float))}

                # ✅ Sort in ascending order: darkest (lowest intensity) to brightest
                sorted_opts = sorted(opt_scores.items(), key=lambda x: x[1])

                for opt in options:
                    score = opt_scores.get(opt, "")
                    row.append(score)

                    if not isinstance(score, (int, float)) or len(sorted_opts) < 2:
                        row.append("")
                        continue

                    current_idx = next((i for i, (o, _) in enumerate(sorted_opts) if o == opt), None)

                    if current_idx is None or current_idx + 1 >= len(sorted_opts):
                        # This is the brightest → 100% of itself
                        row.append(f"100.0 ({opt})")
                    else:
                        next_opt, next_score = sorted_opts[current_idx + 1]
                        if next_score == 0:
                            row.append(f"100.0 ({opt})")
                        else:
                            pct = round((score / next_score) * 100, 2)
                            row.append(f"{pct} ({next_opt})")

            writer.writerow(row)

    print(f"✅ Fixed verification CSV saved to: {output_csv_path}")


if __name__ == "__main__":
    mapped_json_path = r"D:\Projects\OMR\new_abhigyan\debugging\annotate_Test_Series\mapped_annotations.json"
    image_folder = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series"

    output_text_json_path = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series\marked_options.json"
    summary_json_path = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series\summary.json"
    summary_csv_path = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series\summary.csv"
    verification_csv_path = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series\verification.csv"

    option_score_map = detect_marked_options(
        mapped_json_path,
        image_folder,
        output_text_json_path,
        summary_json_path
    )

    clean_and_export_summary(output_text_json_path, summary_json_path, summary_csv_path)
    # export_verification_csv_raw_scores_only(option_score_map, verification_csv_path)
    export_verification_csv(option_score_map, verification_csv_path)