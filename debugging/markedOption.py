import os
import cv2
import json
import numpy as np
from collections import defaultdict
import re
import csv

def is_filled_option(image, bbox, threshold=100):
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 255
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    return mean_intensity < threshold, mean_intensity

def extract_question_id(label):
    for i in range(len(label)):
        if label[i].isdigit():
            while i < len(label) and label[i].isdigit():
                i += 1
            return label[:i]
    return label

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
        option_score_map = {}
        for label, data in annotations.items():
            if not isinstance(data, dict) or "bbox" not in data:
                continue
            qid = extract_question_id(label)
            question_groups[qid].append((label, data["bbox"]))

        summary_results[img_name] = {}
        global_option_score_map[img_name] = {}

        print(f"\n[📄] Results for: {img_name}")
        for qid in sorted(question_groups.keys(), key=lambda x: int(x.replace('Q', '').replace('q', ''))):
            options = question_groups[qid]
            marked_label = None
            best_score = float("inf")

            for label, bbox in options:
                _, score = is_filled_option(image, bbox)
                option_score_map[label] = round(score, 2)

                if score < best_score:
                    best_score = score
                    marked_label = label
                    best_score_value = score

            if marked_label:
                flat_results[f"{img_name}_{qid}"] = marked_label
                summary_results[img_name][qid] = marked_label

        global_option_score_map[img_name] = option_score_map

    with open(output_text_json_path, 'w') as f:
        json.dump(flat_results, f, indent=2)

    with open(summary_json_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n✅ Summary saved to:\n{summary_json_path}")
    print(f"✅ Flat mapping saved to:\n{output_text_json_path}")
    return global_option_score_map

def clean_and_export_summary(marked_options_path, summary_json_path, summary_csv_path):
    with open(marked_options_path, 'r') as f:
        flat_data = json.load(f)

    summary_dict = defaultdict(dict)

    for full_key, value in flat_data.items():
        if '_' not in full_key or not value:
            continue

        img_name, q_key = full_key.split("_", 1)

        # Only process "_1", "_2", etc.
        if re.match(r'^\d+$', q_key):
            qnum = q_key
            if len(value) >= 2 and value[:-1] == qnum:
                selected_option = value[-1]
                question_label = f"Q{qnum}"
                summary_dict[img_name][question_label] = selected_option

    with open(summary_json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"✅ Cleaned summary saved to: {summary_json_path}")

    all_questions = sorted({q for q_data in summary_dict.values() for q in q_data.keys()},
                           key=lambda x: int(x[1:]))

    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name"] + all_questions)

        for img_name, q_answers in summary_dict.items():
            row = [img_name] + [q_answers.get(q, "") for q in all_questions]
            writer.writerow(row)

    print(f"📄 Summary CSV saved to: {summary_csv_path}")

def export_verification_csv(option_score_map, flat_results_path, output_csv_path):
    import csv
    import re
    from collections import defaultdict

    with open(flat_results_path, 'r') as f:
        flat_data = json.load(f)

    image_to_question_map = defaultdict(dict)
    for full_key, value in flat_data.items():
        if '_' not in full_key or not value:
            continue
        img_name, q_key = full_key.split("_", 1)
        if re.match(r'^\d+$', q_key):
            qnum = q_key
            if len(value) >= 2 and value[:-1] == qnum:
                question_label = f"Q{qnum}"
                image_to_question_map[img_name][question_label] = value[-1]

    all_question_ids = sorted(set(
        q for q_data in image_to_question_map.values() for q in q_data.keys()
    ), key=lambda x: int(x[1:]))

    all_columns = []
    for q in all_question_ids:
        qnum = q[1:]
        for opt in ["A", "B", "C", "D"]:
            all_columns.append(f"{qnum}{opt}")
            all_columns.append(f"Result {qnum}{opt}")

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name"] + all_columns)

        for img_name in image_to_question_map.keys():
            row = [img_name]
            scores = option_score_map.get(img_name, {})

            for q in all_question_ids:
                qnum = q[1:]
                option_scores = {}
                for opt in ["A", "B", "C", "D"]:
                    key = f"{qnum}{opt}"
                    option_scores[opt] = scores.get(key, "")

                sorted_opts = sorted(
                    [(opt, val) for opt, val in option_scores.items() if isinstance(val, (int, float))],
                    key=lambda x: x[1]
                )

                for opt in ["A", "B", "C", "D"]:
                    key = f"{qnum}{opt}"
                    score = option_scores.get(opt, "")
                    row.append(score)

                    if not isinstance(score, (int, float)):
                        row.append("")
                        continue

                    idx = next((i for i, (o, _) in enumerate(sorted_opts) if o == opt), -1)
                    if idx == -1:
                        row.append("")
                        continue

                    if idx + 1 < len(sorted_opts):
                        next_opt, next_score = sorted_opts[idx + 1]
                        percentage = round((score / next_score) * 100, 2) if next_score else 100.0
                        row.append(f"{percentage} ({next_opt})")
                    else:
                        row.append(f"100.0 ({opt})")  # highest score, compared with itself

            writer.writerow(row)

    print(f"🧾 Verification CSV saved to: {output_csv_path}")



if __name__ == "__main__":
    mapped_json_path = r"D:\Projects\OMR\new_abhigyan\debugging\annotate_Test_Series\mapped_annotations.json"
    image_folder = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series"

    output_text_json_path = r"D:\Projects\OMR\new_abhigyan\debugging\annotate_Test_Series\marked_options.json"
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
    export_verification_csv(option_score_map, output_text_json_path, verification_csv_path)