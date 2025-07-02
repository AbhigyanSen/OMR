# anchorDetection_modified.py
import cv2
import numpy as np
import math
import os
import json

class OMRProcessor:
    def __init__(self, image_path, annotations_path, classes_path):
        self.image_path = image_path
        self.annotations_path = annotations_path
        self.classes_path = classes_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        self.original_image = self.image.copy()
        self.original_height, self.original_width, _ = self.image.shape
        self.classes = self._load_classes()
        self.annotations = self._load_annotations()

        self.M_transform = None 
        self.deskewed_width = self.original_width
        self.deskewed_height = self.original_height

    def _load_annotations(self):
        annotations = {}
        try:
            with open(self.annotations_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0]) 
                        x_center = float(parts[1]) * self.original_width
                        y_center = float(parts[2]) * self.original_height
                        norm_width = float(parts[3]) * self.original_width
                        norm_height = float(parts[4]) * self.original_height

                        x1 = int(x_center - norm_width / 2)
                        y1 = int(y_center - norm_height / 2)
                        x2 = int(x_center + norm_width / 2)
                        y2 = int(y_center + norm_height / 2)

                        annotations.setdefault(class_id, []).append((x1, y1, x2, y2))
        except FileNotFoundError:
            print(f"Annotation file not found at {self.annotations_path}. Proceeding without it.")
        return annotations

    def _load_classes(self):
        classes = []
        try:
            with open(self.classes_path, 'r') as f:
                for line in f:
                    classes.append(line.strip().replace('\r', '')) 
        except FileNotFoundError:
            print(f"Classes file not found at {self.classes_path}. Proceeding without it.")
        return classes

    def _get_class_id(self, class_name):
        try:
            return self.classes.index(class_name)
        except ValueError:
            return -1

    def _find_omr_sheet_contour(self, img_to_process):
        gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                if cv2.contourArea(contour) > (img_to_process.shape[0] * img_to_process.shape[1] * 0.4):
                    return contour, approx
        return None, None

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _deskew_image(self, sheet_approx_poly):
        if sheet_approx_poly is None:
            self.M_transform = None
            self.deskewed_width = self.original_width
            self.deskewed_height = self.original_height
            return self.original_image 

        pts = sheet_approx_poly.reshape(4, 2)
        ordered_pts = self._order_points(pts)
        (tl, tr, br, bl) = ordered_pts
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
        maxWidth = max(1, maxWidth)
        maxHeight = max(1, maxHeight)

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        deskewed_image = cv2.warpPerspective(self.original_image, M, (maxWidth, maxHeight))
        self.M_transform = M
        self.deskewed_width = maxWidth
        self.deskewed_height = maxHeight
        return deskewed_image

    def detect_anchor_points(self):
        detected_anchors = []
        anchor_class_names = ['Anch1', 'Anch2', 'Anch3', 'Anch4']
        sheet_contour, sheet_approx_poly = self._find_omr_sheet_contour(self.original_image)
        self.image = self._deskew_image(sheet_approx_poly)                                      # self.image is now deskewed!
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        for class_name_str in anchor_class_names:
            class_id = self._get_class_id(class_name_str)
            if class_id != -1 and class_id in self.annotations:
                for bbox in self.annotations[class_id]:
                    x1, y1, x2, y2 = bbox
                    if self.M_transform is not None:
                        original_pts = np.float32([
                            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                        ]).reshape(-1, 1, 2)
                        transformed_pts = cv2.perspectiveTransform(original_pts, self.M_transform).reshape(-1, 2)
                        x_min = int(np.min(transformed_pts[:, 0]))
                        y_min = int(np.min(transformed_pts[:, 1]))
                        x_max = int(np.max(transformed_pts[:, 0]))
                        y_max = int(np.max(transformed_pts[:, 1]))
                        center_x = int((x_min + x_max) / 2)
                        center_y = int((y_min + y_max) / 2)
                    else:
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                    detected_anchors.append({
                        'class_name': class_name_str,
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y)
                    })
        return detected_anchors, self.image, self.M_transform

    def visualize_results(self, detected_anchors, output_filename):
        img = self.image.copy()
        anchor_centers = {}
        for anchor in detected_anchors:
            x1, y1, x2, y2 = anchor['bbox']
            cx, cy = anchor['center']
            class_name = anchor['class_name']
            anchor_centers[class_name] = [cx, cy]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imwrite(output_filename, img)
        return anchor_centers


# MAIN EXECUTION
if __name__ == "__main__":
    folder_path = r"D:\Projects\OMR\new_abhigyan\Phase1\testData\Option_Checking"
    annotations_file = r"D:\Projects\OMR\new_abhigyan\Phase1\Options\BE23-01-01003.txt"
    classes_file = r"D:\Projects\OMR\new_abhigyan\Phase1\Options\classes.txt"

    folder_name = os.path.basename(folder_path.rstrip("\\/"))
    output_dir = os.path.join("output_" + folder_name)
    warning_dir = os.path.join(output_dir, "warnings")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(warning_dir, exist_ok=True)

    anchor_json_path = os.path.join(output_dir, "anchor_centers.json")
    anchor_center_data = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            print(f"\nProcessing {image_path}...")
            try:
                processor = OMRProcessor(image_path, annotations_file, classes_file)
                anchors, deskewed_img, M_transform = processor.detect_anchor_points() # Get deskewed_img and M_transform
                
                if len(anchors) < 4:
                    print("⚠️ Not all anchors detected. Moving to warnings folder.")
                    warning_path = os.path.join(warning_dir, filename)
                    cv2.imwrite(warning_path, deskewed_img if deskewed_img is not None else processor.original_image) # Save deskewed or original
                else:
                    out_path = os.path.join(output_dir, filename)
                    anchor_centers = processor.visualize_results(anchors, out_path)
                    anchor_center_data[filename] = {
                        "anchors": anchor_centers,
                        "M_transform": M_transform.tolist() if M_transform is not None else None, # Convert to list for JSON
                        "deskewed_width": processor.deskewed_width,
                        "deskewed_height": processor.deskewed_height
                    }
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                try:
                    fallback_img = cv2.imread(image_path)
                    if fallback_img is not None:
                        warning_path = os.path.join(warning_dir, filename)
                        cv2.imwrite(warning_path, fallback_img)
                except Exception as save_error:
                    print(f"⚠️ Could not save fallback image: {save_error}")

    with open(anchor_json_path, 'w') as f:
        json.dump(anchor_center_data, f, indent=2)
    print(f"Anchor centers and transformation data saved to {anchor_json_path}")