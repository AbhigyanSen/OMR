import os
import cv2
import json
import re

# --- INPUTS ---
template_image_path = r"D:\Projects\OMR\new_abhigyan\BatchTesting\Colab\Annotations\images\BE24-05-01001.jpg"
label_file_path = r"D:\Projects\OMR\new_abhigyan\BatchTesting\Colab\Annotations\labels\BE24-05-01001.txt"
class_file_path = r"D:\Projects\OMR\new_abhigyan\BatchTesting\Colab\Annotations\classes.txt"
output_json_path = r"BE24-05-01001_template.json"

# --- LOAD IMAGE & CLASSES ---
template_image = cv2.imread(template_image_path)
image_height, image_width = template_image.shape[:2]

with open(label_file_path, 'r') as f:
    labels = [line.strip().split() for line in f]

with open(class_file_path, 'r') as f:
    classes = [line.strip() for line in f]

# --- BUILD ANCHOR & OBJECT MAPS ---
object_centers = {}
object_boxes = {}

for label in labels:
    class_id, x_c, y_c, w, h = map(float, label)
    class_name = classes[int(class_id)]

    x_center = x_c * image_width
    y_center = y_c * image_height
    width = w * image_width
    height = h * image_height

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    object_centers[class_name] = (x_center, y_center)
    object_boxes[class_name] = (x1, y1, x2, y2)

anchor_name = "anchor_1"
anchor_center = object_centers[anchor_name]

# --- RELATIVE JSON FORMAT CREATION ---
def parse_class_name(name):
    if re.match(r'^question_\d+$', name):
        return "question", int(name.split('_')[1])
    elif re.match(r'^\d+[A-D]$', name):
        return "option", int(re.match(r'^(\d+)', name).group(1))
    elif name.startswith("reg_no") and name != "reg_no":
        return "reg_no_char", name
    elif name == "reg_no":
        return "reg_no_main", name
    elif name.startswith("roll_no") and name != "roll_no":
        return "roll_no_char", name
    elif name == "roll_no":
        return "roll_no_main", name
    elif name.startswith("booklet_no") and name != "booklet_no":
        return "booklet_no_char", name
    elif name == "booklet_no":
        return "booklet_no_main", name
    else:
        return None, None

json_data = {
    "questions": {},
    "reg_no": {},
    "roll_no": {},
    "booklet_no": {}
}

for name in object_centers:
    if name == anchor_name:
        continue

    kind, identifier = parse_class_name(name)
    if kind is None:
        continue

    cx, cy = object_centers[name]
    x1, y1, x2, y2 = object_boxes[name]

    rel = {
        "center": {
            "dx": cx - anchor_center[0],
            "dy": cy - anchor_center[1]
        },
        "bbox": {
            "x1": x1 - anchor_center[0],
            "y1": y1 - anchor_center[1],
            "x2": x2 - anchor_center[0],
            "y2": y2 - anchor_center[1]
        }
    }

    if kind == "question":
        qnum = identifier
        json_data["questions"].setdefault(qnum, {"question": {}, "options": {}})
        json_data["questions"][qnum]["question"] = rel
    elif kind == "option":
        qnum = identifier
        json_data["questions"].setdefault(qnum, {"question": {}, "options": {}})
        json_data["questions"][qnum]["options"][name] = rel
    elif kind == "reg_no_char" or kind == "reg_no_main":
        json_data["reg_no"][identifier] = rel
    elif kind == "roll_no_char" or kind == "roll_no_main":
        json_data["roll_no"][identifier] = rel
    elif kind == "booklet_no_char" or kind == "booklet_no_main":
        json_data["booklet_no"][identifier] = rel

# --- SAVE JSON ---
with open(output_json_path, "w") as f:
    json.dump(json_data, f, indent=2)

print(f"[INFO] Template JSON saved to: {output_json_path}")