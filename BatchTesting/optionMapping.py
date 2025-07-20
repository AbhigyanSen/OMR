import cv2
import numpy as np
import os
import json

class OptionMapper:
    def __init__(self, image_path, annotations_path, classes_path, anchor_data,
                 use_orb=False, template_image=None, template_annotations=None, template_json=None):
        self.image_path = image_path
        self.anchor_data = anchor_data
        self.use_orb = use_orb
        self.template_image = template_image
        self.template_boxes = template_annotations or {}
        self.template_centers = {k: ( (x1+x2)/2, (y1+y2)/2 )
                                 for k,(x1,y1,x2,y2) in self.template_boxes.items()}
        self.template_json = template_json or {}
        
        # Load and deskew original image
        orig = cv2.imread(image_path)
        if orig is None: raise FileNotFoundError(f"Missing: {image_path}")
        self.original_image = orig
        h,w = orig.shape[:2]
        self.original_width, self.original_height = w,h

        self.classes = self._load_classes(classes_path)
        self.annotations = self._load_annotations(annotations_path)

        M = np.array(anchor_data.get("M_transform")) if anchor_data.get("M_transform") else None
        dw = int(anchor_data.get("deskewed_width", w))
        dh = int(anchor_data.get("deskewed_height", h))
        self.image = cv2.warpPerspective(orig, M, (dw,dh)) if M is not None else orig.copy()

        self.mapped_annotations = {}

    def _load_classes(self, classes_path):
        with open(classes_path) as f:
            return [l.strip() for l in f]

    def _load_annotations(self, ann_path):
        ann = []
        with open(ann_path) as f:
            for line in f:
                cid, *vals = line.strip().split()
                cname = self.classes[int(cid)]
                if "anchor" in cname.lower(): continue
                x,y,w,h = map(float, vals)
                ann.append((cname, x,y,w,h))
        return ann

    def map_and_draw(self, output_dir):
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        reg_path = os.path.join(output_dir, "OMR", "reg_no")
        roll_path = os.path.join(output_dir, "OMR", "roll_no")
        book_path = os.path.join(output_dir, "OMR", "booklet_no")
        os.makedirs(reg_path, exist_ok=True)
        os.makedirs(roll_path, exist_ok=True)
        os.makedirs(book_path, exist_ok=True)

        if self.use_orb:
            # ORB + Homography branch
            orb = cv2.ORB_create(5000)
            kp1, d1 = orb.detectAndCompute(self.template_image, None)
            kp2, d2 = orb.detectAndCompute(self.image, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            good = [m for m,n in bf.knnMatch(d1, d2, k=2) if m.distance<0.75*n.distance]
            if len(good) < 10: raise RuntimeError("Not enough ORB matches")
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            M, _ = cv2.findHomography(src, dst, cv2.RANSAC,5.0)

            # Map anchor_1
            # Map anchor_1 from template to current image
            anchor1_pt = np.array([[self.template_centers["anchor_1"]]], dtype=np.float32)  # shape: (1,1,2)
            mapped_anchor1 = cv2.perspectiveTransform(anchor1_pt, M)[0][0]  # shape: (2,)
            tx, ty = int(mapped_anchor1[0]), int(mapped_anchor1[1])
            self.mapped_annotations["anchor_1"] = {
                "center": [tx, ty],
                "bbox": list(self.template_boxes["anchor_1"]),
                "delta_from_Anch1": [0, 0]
            }

            # Now map ALL other template fields using dx, dy from anchor_1
            for q, qdat in self.template_json.get("questions", {}).items():
                if "question" in qdat:
                    dx = qdat["question"]["center"]["dx"]
                    dy = qdat["question"]["center"]["dy"]
                    cx, cy = tx + dx, ty + dy
                    self.mapped_annotations[f"question_{q}"] = {
                        "center": [cx, cy],
                        "bbox": [],
                        "delta_from_Anch1": [dx, dy]
                    }

                for opt, od in qdat.get("options", {}).items():
                    dx = od["center"]["dx"]
                    dy = od["center"]["dy"]
                    cx, cy = tx + dx, ty + dy
                    self.mapped_annotations[opt] = {
                        "center": [cx, cy],
                        "bbox": [],
                        "delta_from_Anch1": [dx, dy]
                    }


            # No OMR crops here (because using template), so skip other logic

        else:
            # YOLO-based anchor + OMR branch
            anch1 = self.anchor_data["anchors"]["anchor_1"]["center"]
            a1x,a1y = anch1
            for nm,info in self.anchor_data["anchors"].items():
                cx,cy = info["center"]
                dx,dy = cx-a1x, cy-a1y
                self.mapped_annotations[nm] = {"center":[cx,cy],"bbox":info["bbox"],"delta_from_Anch1":[dx,dy]}

            # Map annotation fields + cropping
            for cname,x,y,w,h in self.annotations:
                ox,oy,ow,oh = x*self.original_width, y*self.original_height, w*self.original_width, h*self.original_height
                x1,y1,x2,y2 = int(ox-ow/2), int(oy-oh/2), int(ox+ow/2), int(oy+oh/2)
                
                x1 = max(0, min(x1, self.image.shape[1] - 1))
                x2 = max(0, min(x2, self.image.shape[1] - 1))
                y1 = max(0, min(y1, self.image.shape[0] - 1))
                y2 = max(0, min(y2, self.image.shape[0] - 1))
                
                cx2,cy2 = (x1+x2)//2,(y1+y2)//2
                dx,dy = cx2-a1x, cy2-a1y
                self.mapped_annotations[cname] = {"center":[cx2,cy2],"bbox":[x1,y1,x2,y2],"delta_from_Anch1":[dx,dy]}

                # Draw and crop
                cv2.rectangle(self.image,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(self.image,cname,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
                crop = self.image[y1:y2, x1:x2]
                if crop.size>0:
                    out = os.path.join(reg_path if cname=="reg_no" else roll_path if cname=="roll_no" else book_path, f"{base}.jpg")
                    cv2.imwrite(out, crop)

        # Draw all mapped annotations on self.image
        debug = True

        if debug:
            cv2.circle(self.image, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(self.image, nm, (int(cx)+4, int(cy)-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        for nm, info in self.mapped_annotations.items():
            cx,cy = info["center"]
            bbox = info.get("bbox",[])
            if bbox:
                x1,y1,x2,y2 = map(int,bbox)
                cv2.rectangle(self.image,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.circle(self.image,(int(cx),int(cy)),4,(0,0,255),-1)
            cv2.putText(self.image,nm,(int(cx)+5,int(cy)-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)

        return self.image, self.mapped_annotations

def process_folder(folder_path, annotations_file, classes_file, anchor_json_path,
                   use_orb=False, template_img_path=None, template_label_path=None, template_json_path=None):
    warning_dir = os.path.join(".", "warnings")
    os.makedirs(warning_dir, exist_ok=True)
    anchors = json.load(open(anchor_json_path))

    template_img, tb, tj = None, None, None
    if use_orb:
        template_img = cv2.imread(template_img_path)
        tb, _ = {}, {}
        for line in open(template_label_path):
            cid,xc,yc,w,h = line.strip().split()
            allc = open(classes_file).read().split(); name = allc[int(cid)]
            img = template_img; iw,ih = img.shape[1],img.shape[0]
            x1,y1 = float(xc)*iw - float(w)*iw/2, float(yc)*ih - float(h)*ih/2
            x2,y2 = x1 + float(w)*iw, y1 + float(h)*ih
            tb[name] = (x1,y1,x2,y2)
        tj = json.load(open(template_json_path))

    results = {}
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg",".png")): continue
        ad = anchors.get(fname, {})
        if not ad.get("valid_for_option_mapping", False):
            cv2.imwrite(os.path.join(warning_dir, fname), cv2.imread(os.path.join(folder_path, fname)))
            results[fname] = {"error": "invalid anchor", "valid_for_marked_option":False}
            continue
        mapper = OptionMapper(
            os.path.join(folder_path, fname),
            annotations_file, classes_file,
            ad, use_orb, template_img, tb, tj
        )
        img, ma = mapper.map_and_draw(folder_path)
        cv2.imwrite(os.path.join(".", fname), img)
        ma["valid_for_marked_option"] = True
        results[fname] = ma

    json.dump(results, open("mapped_annotations.json","w"), indent=2)
    print("✅ Done.")
    

if __name__ == "__main__":
    base_folder = r"D:\Projects\OMR\new_abhigyan\BatchTesting"
    # folder_path = os.path.join(base_folder, "TestData", "BE24-05-07")
    folder_path = r"D:\Projects\OMR\new_abhigyan\BatchTesting\anchor_BE24-05-07"
    annotations_file = os.path.join(base_folder, "Annotations", "labels", "BE24-05-01001.txt")
    classes_file = os.path.join(base_folder, "Annotations", "classes.txt")
    
    # Corrected path to anchor_centers.json (from anchorDetection.py's output)
    anchor_output_folder_name = "anchor_" + os.path.basename(folder_path.rstrip("\\/"))
    # anchor_data_json_path = os.path.join(base_folder, anchor_output_folder_name, "anchor_centers.json")
    anchor_data_json_path = r"D:\Projects\OMR\new_abhigyan\BatchTesting\anchor_BE24-05-07\anchor_centers.json"

    process_folder(folder_path, annotations_file, classes_file, anchor_data_json_path)