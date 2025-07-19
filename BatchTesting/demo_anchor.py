import cv2
import os
import json
import math
import numpy as np
from imutils import rotate_bound

# --- CONFIGURATION ---
YOLO_CLASSES_FILE = r'D:\Projects\OMR\new_abhigyan\BatchTesting\Annotations\classes.txt'
YOLO_LABEL_FILE = r'D:\Projects\OMR\new_abhigyan\BatchTesting\Annotations\labels\BE24-05-01001.txt'
YOLO_IMAGE_FILE = r'D:\Projects\OMR\new_abhigyan\BatchTesting\Annotations\images\BE24-05-01001.jpg'

ANCHOR_NAMES = ['anchor_1', 'anchor_2', 'anchor_3', 'anchor_4']
SKEW_THRESHOLD = 3  # degrees
SEARCH_MARGIN = 0.1  # 10% around anchor template

def load_classes(classes_file):
    with open(classes_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def load_yolo_boxes(label_file, classes, img_w, img_h):
    anchors = {}
    with open(label_file, 'r') as f:
        for line in f:
            class_id, x, y, w, h = map(float, line.strip().split())
            class_name = classes[int(class_id)]
            if class_name in ANCHOR_NAMES:
                abs_x = int(x * img_w)
                abs_y = int(y * img_h)
                abs_w = int(w * img_w)
                abs_h = int(h * img_h)
                x1 = abs_x - abs_w // 2
                y1 = abs_y - abs_h // 2
                x2 = abs_x + abs_w // 2
                y2 = abs_y + abs_h // 2
                anchors[class_name] = (x1, y1, x2, y2)
    return anchors

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def get_angle(pt1, pt2):
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

def rotate_image(image, angle, clockwise=True):
    return rotate_bound(image, -angle if clockwise else angle)

def extract_templates(img, anchor_boxes):
    return {
        name: img[y1:y2, x1:x2]
        for name, (x1, y1, x2, y2) in anchor_boxes.items()
    }

def match_template(img, template, ref_box):
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = ref_box
    pad_x = int((x2 - x1) * SEARCH_MARGIN * 2)
    pad_y = int((y2 - y1) * SEARCH_MARGIN * 2)

    sx1 = max(x1 - pad_x, 0)
    sy1 = max(y1 - pad_y, 0)
    sx2 = min(x2 + pad_x, w_img)
    sy2 = min(y2 + pad_y, h_img)

    search_region = img[sy1:sy2, sx1:sx2]

    # Convert to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

    # Edge detection for robustness
    template_edges = cv2.Canny(template_gray, 50, 150)
    search_edges = cv2.Canny(search_gray, 50, 150)

    method = cv2.TM_CCOEFF_NORMED
    result = cv2.matchTemplate(search_edges, template_edges, method)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    top_left = (sx1 + max_loc[0], sy1 + max_loc[1])
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    return (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

def draw_anchors(img, anchors):
    for name, box in anchors.items():
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def process_omr_batch(input_folder):
    foldername = os.path.basename(os.path.normpath(input_folder))
    output_dir = f'anchor_{foldername}'
    warnings_dir = os.path.join(output_dir, 'warnings')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(warnings_dir, exist_ok=True)

    # Load reference image and anchors
    ref_img = cv2.imread(YOLO_IMAGE_FILE)
    h_ref, w_ref = ref_img.shape[:2]
    classes = load_classes(YOLO_CLASSES_FILE)
    ref_anchor_boxes = load_yolo_boxes(YOLO_LABEL_FILE, classes, w_ref, h_ref)
    ref_templates = extract_templates(ref_img, ref_anchor_boxes)

    # Reference skew angle
    ref_angle = get_angle(get_center(ref_anchor_boxes['anchor_1']),
                          get_center(ref_anchor_boxes['anchor_2']))

    summary = []

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        if fname == os.path.basename(YOLO_IMAGE_FILE):
            continue

        img_path = os.path.join(input_folder, fname)
        img = cv2.imread(img_path)
        h_img, w_img = img.shape[:2]

        anchor_results = {}
        warning = False

        for name in ANCHOR_NAMES:
            try:
                ref_box = ref_anchor_boxes[name]
                template = ref_templates[name]
                matched_box = match_template(img, template, ref_box)
                anchor_results[name] = matched_box
            except Exception:
                warning = True
                break

        if len(anchor_results) != 4:
            warning = True

        result_entry = {
            "image_name": fname,
            "anchors": {},
            "skew_angle": None,
            "rotation": None,
            "warning": warning
        }

        for name, box in anchor_results.items():
            center = get_center(box)
            result_entry["anchors"][name] = {
                "bbox": list(box),
                "center": list(center)
            }

        if not warning:
            p1 = get_center(anchor_results['anchor_1'])
            p2 = get_center(anchor_results['anchor_2'])
            angle = get_angle(p1, p2)
            skew_diff = angle - ref_angle

            result_entry["skew_angle"] = round(angle, 2)
            result_entry["rotation"] = "clockwise" if skew_diff < 0 else "anticlockwise"

            if abs(skew_diff) > SKEW_THRESHOLD:
                warning = True
                result_entry["warning"] = True
                draw_anchors(img, anchor_results)
                cv2.imwrite(os.path.join(warnings_dir, fname), img)
            else:
                # Correct skew
                deskewed = rotate_image(img, abs(skew_diff), clockwise=(skew_diff < 0))
                draw_anchors(deskewed, anchor_results)
                cv2.imwrite(os.path.join(output_dir, fname), deskewed)
        else:
            draw_anchors(img, anchor_results)
            cv2.imwrite(os.path.join(warnings_dir, fname), img)

        summary.append(result_entry)

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n✅ Completed! Results saved to: {output_dir}")
    print(f"📄 Summary written to: {os.path.join(output_dir, 'summary.json')}")
    print(f"⚠️ Warnings in: {warnings_dir}")

# ------------------------------
# USAGE
# ------------------------------
if __name__ == '__main__':
    # Replace with your folder of OMR images
    input_folder_path = r'D:\Projects\OMR\new_abhigyan\BatchTesting\TestData\BE24-05-01_Test'
    process_omr_batch(input_folder_path)
