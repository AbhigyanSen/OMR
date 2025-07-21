import os
import cv2
import json
import numpy as np

# --- CONFIG ---
test_images_folder = r"D:\Projects\OMR\new_abhigyan\BatchTesting\Colab\BE24-05-07"
template_image_path = r"D:\Projects\OMR\new_abhigyan\BatchTesting\Colab\Annotations\images\BE24-05-01001.jpg"
template_json_path = r"D:\Projects\OMR\new_abhigyan\BatchTesting\Colab\BE24-05-01001_template.json"
label_file_path = r"D:\Projects\OMR\new_abhigyan\BatchTesting\Colab\Annotations\labels\BE24-05-01001.txt"
class_file_path = r"D:\Projects\OMR\new_abhigyan\BatchTesting\Colab\Annotations\classes.txt"
output_folder = "output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Load template ---
template_image = cv2.imread(template_image_path)
image_height, image_width = template_image.shape[:2]

with open(template_json_path, "r") as f:
    template_data = json.load(f)

with open(class_file_path, "r") as f:
    classes = [line.strip() for line in f]

with open(label_file_path, "r") as f:
    labels = [line.strip().split() for line in f]

# --- Extract anchor_1 ---
anchor_name = "anchor_1"
object_centers = {}
object_boxes = {}

for label in labels:
    class_id, x_c, y_c, w, h = map(float, label)
    class_name = classes[int(class_id)]

    if class_name == anchor_name:
        x_center = x_c * image_width
        y_center = y_c * image_height
        width = w * image_width
        height = h * image_height

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        object_centers[anchor_name] = (x_center, y_center)
        object_boxes[anchor_name] = [x1, y1, x2, y2]
        break
else:
    raise Exception("[ERROR] anchor_1 not found")

# --- Homography + Mapping ---
def process_test_image(test_image_path):
    test_img = cv2.imread(test_image_path)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(template_image, None)
    kp2, des2 = orb.detectAndCompute(test_img, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        raise Exception("Not enough good matches")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    anchor_pt = np.array([[object_centers[anchor_name]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(anchor_pt, M)
    anchor_x, anchor_y = map(int, transformed[0][0])

    result = test_img.copy()
    
    def draw_label_group(group, color, prefix=""):
        for name, data in group.items():
            dx, dy = data["center"]["dx"], data["center"]["dy"]
            cx, cy = int(anchor_x + dx), int(anchor_y + dy)
            cv2.circle(result, (cx, cy), 4, color, -1)
            # cv2.putText(result, f"{prefix}{name}", (cx + 5, cy - 5),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            bbox = data.get("bbox")
            if bbox:
                x1 = int(anchor_x + bbox["x1"])
                y1 = int(anchor_y + bbox["y1"])
                x2 = int(anchor_x + bbox["x2"])
                y2 = int(anchor_y + bbox["y2"])
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 1)

    # Draw fields
    for qnum, qdata in template_data["questions"].items():
        if "question" in qdata:
            rel = qdata["question"]
            cx = int(anchor_x + rel["center"]["dx"])
            cy = int(anchor_y + rel["center"]["dy"])
            cv2.circle(result, (cx, cy), 4, (255, 0, 255), -1)
            if "bbox" in rel:
                x1 = int(anchor_x + rel["bbox"]["x1"])
                y1 = int(anchor_y + rel["bbox"]["y1"])
                x2 = int(anchor_x + rel["bbox"]["x2"])
                y2 = int(anchor_y + rel["bbox"]["y2"])
                cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 100), 2)

        for opt, rel in qdata.get("options", {}).items():
            cx = int(anchor_x + rel["center"]["dx"])
            cy = int(anchor_y + rel["center"]["dy"])
            cv2.circle(result, (cx, cy), 4, (0, 255, 255), -1)
            if "bbox" in rel:
                x1 = int(anchor_x + rel["bbox"]["x1"])
                y1 = int(anchor_y + rel["bbox"]["y1"])
                x2 = int(anchor_x + rel["bbox"]["x2"])
                y2 = int(anchor_y + rel["bbox"]["y2"])
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Draw other fields
    draw_label_group(template_data["reg_no"], (0, 0, 255), "REG_")
    draw_label_group(template_data["roll_no"], (0, 255, 0), "ROLL_")
    draw_label_group(template_data["booklet_no"], (255, 0, 0), "BOOK_")

    return result

# --- Run on folder ---
for filename in os.listdir(test_images_folder):
    if filename.lower().endswith((".jpg", ".png")):
        try:
            input_path = os.path.join(test_images_folder, filename)
            result_img = process_test_image(input_path)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_result.jpg")
            cv2.imwrite(output_path, result_img)
            print(f"[INFO] Saved: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed on {filename}: {e}")
