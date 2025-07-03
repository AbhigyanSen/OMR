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
        for label, data in annotations.items():
            if not isinstance(data, dict) or "bbox" not in data:
                continue
            qid = extract_question_id(label)
            question_groups[qid].append((label, data["bbox"]))

        summary_results[img_name] = {}

        print(f"\n[📄] Results for: {img_name}")
        for qid in sorted(question_groups.keys(), key=lambda x: int(x.replace('Q', '').replace('q', ''))):
            options = question_groups[qid]
            marked_label = None
            best_score = float("inf")

            for label, bbox in options:
                _, score = is_filled_option(image, bbox)
                if score < best_score:
                    best_score = score
                    marked_label = label
                    best_score_value = score

            if marked_label:
                print(f"{qid}: {qid} (score: {best_score_value:.2f})")
                print(f"{qid.replace('Q', '')}: {marked_label} (score: {best_score_value:.2f})")
                flat_results[f"{img_name}_{qid}"] = marked_label
                summary_results[img_name][qid] = marked_label

    with open(output_text_json_path, 'w') as f:
        json.dump(flat_results, f, indent=2)

    with open(summary_json_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n✅ Summary saved to:\n{summary_json_path}")
    print(f"✅ Flat mapping saved to:\n{output_text_json_path}")

def clean_and_export_summary(marked_options_path, summary_json_path, summary_csv_path):
    with open(marked_options_path, 'r') as f:
        flat_data = json.load(f)

    summary_dict = defaultdict(dict)

    for full_key, value in flat_data.items():
        if '_' not in full_key or not value:
            continue
        
        img_name, q_key = full_key.split("_", 1)

        # Only process keys like "_1", "_2", etc. (not "_Q1")
        if re.match(r'^\d+$', q_key):
            qnum = q_key  # "1"
            if len(value) >= 2 and value[:-1] == qnum:  # e.g., "1C"
                selected_option = value[-1]
                question_label = f"Q{qnum}"
                summary_dict[img_name][question_label] = selected_option

    # Save JSON
    with open(summary_json_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"✅ Cleaned summary saved to: {summary_json_path}")

    # Save CSV
    all_questions = sorted({q for q_data in summary_dict.values() for q in q_data.keys()},
                           key=lambda x: int(x[1:]))  # Q1, Q2, etc.

    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name"] + all_questions)

        for img_name, q_answers in summary_dict.items():
            row = [img_name] + [q_answers.get(q, "") for q in all_questions]
            writer.writerow(row)

    print(f"📄 Summary CSV saved to: {summary_csv_path}")



if __name__ == "__main__":
    mapped_json_path = r"D:\Projects\OMR\new_abhigyan\debugging\annotate_Test_Series\mapped_annotations.json"
    image_folder = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series"
    output_text_json_path = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series\marked_options.json"
    summary_json_path = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series\summary.json"
    summary_csv_path = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series\summary.csv"

    detect_marked_options(mapped_json_path, image_folder, output_text_json_path, summary_json_path)
    clean_and_export_summary(output_text_json_path, summary_json_path, summary_csv_path)