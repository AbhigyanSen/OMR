import cv2
import numpy as np
import os
import math

# --- Configuration ---
WARNING_FOLDER = "warning"
SKEW_THRESHOLD_DEGREES = 3
ROI_PERCENTAGE = 0.10  # 10% area around the center of annotated anchor for search

# Global variable to store the detected anchor shape from the first processed image in a batch
# 0: Unknown, 1: Square/Rectangle, 2: Circle
detected_anchor_shape = 0
ANCHOR_AREA_RANGE_FACTOR = 0.3 # Allowable deviation from the average anchor area (e.g., 0.3 means +/- 30%)

def load_yolo_annotations(label_path, img_width, img_height):
    """
    Loads YOLO format annotations from a file and converts them to pixel coordinates.
    Returns a list of dictionaries, each containing 'class_id', 'x_center', 'y_center', 'width', 'height'.
    """
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            x_center_norm, y_center_norm, width_norm, height_norm = parts[1:]

            # Convert normalized coordinates to pixel coordinates
            x_center = int(x_center_norm * img_width)
            y_center = int(y_center_norm * img_height)
            width = int(width_norm * img_width)
            height = int(height_norm * img_height)

            annotations.append({
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
    return annotations

def get_anchor_shape(contour):
    """
    Analyzes a contour to determine if it's likely a square/rectangle or a circle.
    Returns 1 for square/rectangle, 2 for circle, 0 for unknown.
    """
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    area = cv2.contourArea(contour)

    if area == 0:
        return 0

    # Check for square/rectangle
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.8 <= aspect_ratio <= 1.2:  # Allowing for slight variations
            return 1  # Likely a square or rectangle
    
    # Check for circle
    # Calculate circularity: 4 * pi * Area / (Perimeter^2)
    # A perfect circle has a circularity of 1.
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if 0.7 <= circularity <= 1.1: # A range to account for imperfections
            return 2 # Likely a circle
            
    return 0 # Unknown shape

def detect_anchors(image_path, annotations=None):
    """
    Detects the four anchors in an OMR sheet.
    Determines anchor shape from the first image and uses it for subsequent images.
    Returns a list of detected anchor center points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    and their bounding box coordinates.
    """
    global detected_anchor_shape

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}. Please ensure the path is correct and the image exists.")
        return None, None

    img_height, img_width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    detected_anchor_coords = [None] * 4  # To store (x, y, w, h) of detected anchors
    detected_anchor_centers = [None] * 4 # To store (x_center, y_center) of detected anchors

    # Threshold the image
    # Use adaptive thresholding for better results across varying lighting conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    avg_anchor_area = None
    min_anchor_area = None
    max_anchor_area = None

    if annotations:
        # If annotations are provided (for the first image in a batch),
        # use them to determine the anchor shape and approximate area.
        initial_anchor_areas = []
        for ann in annotations:
            # Extract the region for the anchor from annotations
            x_min = max(0, ann['x_center'] - ann['width'] // 2)
            y_min = max(0, ann['y_center'] - ann['height'] // 2)
            x_max = min(img_width, ann['x_center'] + ann['width'] // 2)
            y_max = min(img_height, ann['y_center'] + ann['height'] // 2)

            # Ensure the ROI is valid
            if x_max <= x_min or y_max <= y_min:
                continue

            anchor_roi = thresh[y_min:y_max, x_min:x_max]
            
            contours_roi, _ = cv2.findContours(anchor_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours_roi:
                area = cv2.contourArea(cnt)
                if area > 10: # Filter small noise
                    initial_anchor_areas.append(area)
                    shape = get_anchor_shape(cnt)
                    if shape != 0:
                        # We only need to determine the shape once per batch
                        if detected_anchor_shape == 0:
                            detected_anchor_shape = shape
                        break # Found a potential anchor in this ROI, move to next annotation
            if detected_anchor_shape != 0: # If shape determined, no need to check other annotations for shape
                break # If the shape is determined, stop processing annotations for shape

        if not initial_anchor_areas:
            print("Warning: Could not determine initial anchor areas from annotations. Anchor area filtering might be less effective.")
        else:
            avg_anchor_area = np.mean(initial_anchor_areas)
            min_anchor_area = avg_anchor_area * (1 - ANCHOR_AREA_RANGE_FACTOR)
            max_anchor_area = avg_anchor_area * (1 + ANCHOR_AREA_RANGE_FACTOR)

        if detected_anchor_shape == 0:
            print("Warning: Could not determine anchor shape from the annotated image. Defaulting to assuming squares/rectangles.")
            detected_anchor_shape = 1 # Default to squares/rectangles if not determined
    
    # If this is a subsequent image in a batch (no annotations provided),
    # or if we are processing the first image and have determined the shape.
    # We need a reference for the expected positions of anchors.
    # For subsequent images, we assume their relative positions are similar to the first annotated image.
    # We will use hardcoded relative positions if annotations were not provided,
    # or infer from the first annotation's relative positions.
    
    # Since the problem statement implies we always start with an annotated image for a batch type,
    # 'annotations' will be available for the first call. For subsequent calls, they won't be.
    # If annotations were provided, use them as reference points.
    # If not, we cannot reliably find anchors without a template or learned positions.
    # Assuming 'annotations' will always be available for the first call of a batch type.
    
    # To handle subsequent images without re-providing annotations,
    # we need a way to store/infer the expected relative positions.
    # For this current setup, we'll assume the provided `annotations` for 100418.jpg
    # serve as the template for all images in this "batch".

    if annotations is None:
        # This block would be for subsequent images in a batch.
        # Without annotations, we need to infer where to look.
        # A robust solution would involve storing the relative positions (normalized)
        # from the first annotated image and applying them here.
        # For this specific problem, since the first image sets the "batch" type,
        # we will use a placeholder message indicating the need for a template.
        print("Error: No annotations provided for this image. Cannot detect anchors without a reference template. Please provide annotations for the first image in a batch.")
        return None, None

    # Now, search for anchors based on their expected positions and known shape
    for i, ann in enumerate(annotations):
        # Define ROI around the annotated center point
        # The ROI size is based on a percentage of the overall image dimensions
        roi_half_width = int(img_width * ROI_PERCENTAGE / 2)
        roi_half_height = int(img_height * ROI_PERCENTAGE / 2)

        roi_x_min = max(0, ann['x_center'] - roi_half_width)
        roi_y_min = max(0, ann['y_center'] - roi_half_height)
        roi_x_max = min(img_width, ann['x_center'] + roi_half_width)
        roi_y_max = min(img_height, ann['y_center'] + roi_half_height)
        
        # Ensure ROI is valid
        if roi_x_max <= roi_x_min or roi_y_max <= roi_y_min:
            print(f"Warning: Invalid ROI for anchor {i+1}. Skipping.")
            continue

        roi = thresh[roi_y_min:roi_y_max, roi_x_min:roi_x_max]

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_match = None
        min_dist = float('inf')

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filter by area if average anchor area was determined
            if avg_anchor_area is not None and not (min_anchor_area <= area <= max_anchor_area):
                continue # Skip contours outside the expected area range

            shape = get_anchor_shape(cnt)
            if shape == detected_anchor_shape:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Convert bounding box coordinates back to original image coordinates
                abs_x = x + roi_x_min
                abs_y = y + roi_y_min

                current_center_x = abs_x + w // 2
                current_center_y = abs_y + h // 2

                # Calculate distance from the annotated center
                dist = math.sqrt((current_center_x - ann['x_center'])**2 + (current_center_y - ann['y_center'])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_match = (abs_x, abs_y, w, h)
                    detected_anchor_centers[ann['class_id']] = (current_center_x, current_center_y)
                    detected_anchor_coords[ann['class_id']] = (abs_x, abs_y, w, h)

    # Ensure all 4 anchors are detected
    if None in detected_anchor_centers or None in detected_anchor_coords:
        print("Error: Not all 4 anchors could be detected.")
        return None, None

    # Print the detected anchor shape (this will be printed only once for the batch type)
    if detected_anchor_shape == 1:
        print(f"Detected Anchor Shape: Square/Rectangle")
    elif detected_anchor_shape == 2:
        print(f"Detected Anchor Shape: Circle")
    else:
        print(f"Detected Anchor Shape: Unknown (Defaulting to Square/Rectangle for future processing)")


    return detected_anchor_coords, detected_anchor_centers

def calculate_skew_angle(p1, p2):
    """
    Calculates the skew angle in degrees between two points and the horizontal axis.
    Points are (x, y).
    """
    x1, y1 = p1
    x2, y2 = p2
    
    if x2 == x1: # Vertical line
        return 90.0
    
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def process_omr_sheet(image_path, label_path=None, initial_annotations=None):
    """
    Main function to process an OMR sheet: detect anchors, calculate skew, and handle output.
    initial_annotations: This parameter will hold the annotations from the first image
                         in a batch, so subsequent images can use them as a reference
                         even if they don't have their own label_path.
    """
    print(f"\nProcessing: {os.path.basename(image_path)}")

    annotations_for_current_image = None
    if label_path and os.path.exists(label_path):
        # This is the first image in a batch, load its specific annotations
        temp_img = cv2.imread(image_path)
        if temp_img is not None:
            img_height, img_width = temp_img.shape[:2]
            annotations_for_current_image = load_yolo_annotations(label_path, img_width, img_height)
        else:
            print(f"Warning: Could not read image {image_path} for dimension inference.")
    elif initial_annotations:
        # This is a subsequent image, use the annotations from the first image of the batch
        annotations_for_current_image = initial_annotations
        # For subsequent images, we also need their dimensions to scale the stored normalized annotations
        temp_img = cv2.imread(image_path)
        if temp_img is not None:
            img_height, img_width = temp_img.shape[:2]
            # Re-normalize if needed, or convert annotations directly to current image's pixel coords
            # For simplicity, if we have initial_annotations (which are already in pixel form relative to their image),
            # we need to adjust them for the current image's size if the image sizes differ.
            # A more robust approach would be to store normalized annotations from the first image
            # and apply them to the current image's dimensions here.
            # For now, let's assume images in a batch are roughly the same resolution.
            # If resolution varies significantly, the ROI calculation needs to be smarter.
            pass # The detect_anchors function expects pixel coords, and will use the original img_width/height from the current image.
        else:
            print(f"Warning: Could not read image {image_path} for dimension inference for subsequent processing.")
            return

    detected_coords, detected_centers = detect_anchors(image_path, annotations_for_current_image)

    if detected_coords is None or detected_centers is None:
        print(f"Skipping {os.path.basename(image_path)} due to anchor detection failure.")
        return

    # Print detected anchor coordinates and center points
    for i, (coords, center) in enumerate(zip(detected_coords, detected_centers)):
        if coords and center:
            print(f"Anchor {i+1} (Class ID {i}): Coords=({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}), Center=({center[0]}, {center[1]})")

    # Calculate skew angle using anchors 1 and 2 (class_id 0 and 1)
    # As per problem statement: "do not pair anch1 with anch4 and anch3 with anch2 as my anchors are in the order 1 2 4 3, in clockwise order."
    # So we use anchors 1 (class_id 0) and 2 (class_id 1) for the top line.
    
    # Ensure points for skew calculation are available
    if detected_centers[0] is not None and detected_centers[1] is not None:
        skew_angle = calculate_skew_angle(detected_centers[0], detected_centers[1])
        print(f"Calculated Skew Angle (using Anchor 1 and Anchor 2): {skew_angle:.2f} degrees")
    else:
        print("Error: Could not calculate skew angle, anchor centers for 1 and 2 are missing.")
        return

    # Skew angle validation
    if abs(skew_angle) > SKEW_THRESHOLD_DEGREES:
        print(f"ERROR: SKEW - Angle {skew_angle:.2f} degrees (exceeds {SKEW_THRESHOLD_DEGREES} degrees threshold)")
        # Move image to warning folder
        os.makedirs(WARNING_FOLDER, exist_ok=True)
        # Construct destination path
        dest_path = os.path.join(WARNING_FOLDER, os.path.basename(image_path))
        try:
            # Need to copy instead of rename if the original image needs to stay in its place
            # Or make a copy, then delete original if required. For this, `shutil` would be better.
            # For simplicity, if we rename, it will move it.
            # If the source and destination are on different drives, os.rename will fail.
            # Assuming they are on the same drive.
            if os.path.exists(image_path): # Check if source file still exists before moving
                os.rename(image_path, dest_path)
                print(f"Moved {os.path.basename(image_path)} to {WARNING_FOLDER}/")
            else:
                print(f"Warning: Source file {image_path} not found for moving.")

        except OSError as e:
            print(f"Error moving file {image_path} to {dest_path}: {e}")
    else:
        print(f"Skew Angle: {skew_angle:.2f} degrees (within {SKEW_THRESHOLD_DEGREES} degrees threshold)")

# --- Main Execution ---
if __name__ == "__main__":
    # Define the provided file paths using raw strings (r"...") for Windows paths
    # or ensure forward slashes
    annotated_image_path = r"D:\Projects\OMR\new_abhigyan\Phase1\Data\images\100418.jpg"
    annotated_labels_path = r"D:\Projects\OMR\new_abhigyan\Phase1\Data\labels\100418.txt"
    test_image_path = r"D:\Projects\OMR\new_abhigyan\Phase1\testData\101660.jpg"
    classes_txt_path = r"D:\Projects\OMR\new_abhigyan\Phase1\Data\classes.txt" # Not directly used in current logic, but good to keep reference

    # Create the warning folder if it doesn't exist
    os.makedirs(WARNING_FOLDER, exist_ok=True)

    # Global variable to store initial annotations for reuse across a batch
    global initial_batch_annotations

    # Process the initial annotated image to determine anchor shape and get initial reference points
    print("--- Processing Initial Annotated Image (Batch 1 Reference) ---")
    
    # Load initial annotations only once for the batch reference
    temp_img_initial = cv2.imread(annotated_image_path)
    if temp_img_initial is not None:
        img_height_initial, img_width_initial = temp_img_initial.shape[:2]
        initial_batch_annotations = load_yolo_annotations(annotated_labels_path, img_width_initial, img_height_initial)
        process_omr_sheet(annotated_image_path, annotated_labels_path, initial_batch_annotations)
    else:
        print(f"CRITICAL ERROR: Could not load initial annotated image at {annotated_image_path}. Cannot proceed.")
        initial_batch_annotations = None # Ensure it's None if loading fails

    # Process the test image (assuming it's in the same batch type)
    # This will use the `detected_anchor_shape` determined above and the `initial_batch_annotations` as reference.
    if initial_batch_annotations:
        print("\n--- Processing Test Image (Same Batch) ---")
        process_omr_sheet(test_image_path, initial_annotations=initial_batch_annotations)
    else:
        print("\nSkipping test image processing as initial batch setup failed.")

    print("\nProcessing complete.")