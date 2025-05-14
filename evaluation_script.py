import json
import sys
import os
import re
import ast

def compare_json_files(gt_file, msg_file):
    # load validation data
    with open(gt_file, 'r') as f:
        records = json.load(f)

    # build map image -> ground-truth info (classes + bboxes)
    gt_map = {}
    for rec in records:
        img = rec.get('image')
        # find the GPT reply in conversations
        for conv in rec.get('conversations', []):
            if conv.get('from') == 'gpt':
                val = conv.get('value', '')
                try:
                    data = ast.literal_eval(val)
                    # pull GT detections
                    dets = data.get('detections', [])
                    classes = {d['class'] for d in dets if 'class' in d}
                    bboxes = [
                        {'class': d['class'], **d['bbox']}
                        for d in dets
                        if 'class' in d and 'bbox' in d
                    ]
                    gt_map[img] = {'classes': classes, 'bboxes': bboxes}
                except:
                    gt_map[img] = {'classes': set(), 'bboxes': []}
                break

    total_score = 0
    count = 0
    total_mse = 0.0
    mse_count = 0
    # process message.txt
    with open(msg_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            img, content = parts
            # extract predicted classes via substring matching (works on malformed payloads)
            pred_classes = re.findall(r"'class'\s*:\s*'([^']+)'", content)
            valid_pred = len(pred_classes) > 0

            # get GT classes set (gt_map[img] is {'classes':…, 'bboxes':…})
            gt_info    = gt_map.get(img, {'classes': set(), 'bboxes': []})
            gt_classes = gt_info.get('classes', set())
            # score +1 for each predicted class in gt, -1 otherwise
            score = sum((1 if c in gt_classes else -1) for c in pred_classes)
            print(f"{img}: GT={sorted(gt_classes)}, Pred={pred_classes}, Score={score}")
            total_score += score
            count += 1

            # if there was any predicted class, try to compute bbox-MSE
            if valid_pred:
                # extract predicted bboxes via regex (works on truncated/malformed payloads)
                pred_bboxes = []
                # find all "'bbox': { ... }" substrings
                bbox_strs = re.findall(r"'bbox'\s*:\s*{[^}]*}", content)
                for bs in bbox_strs:
                    # extract numeric fields
                    nums = {}
                    for field in ('center_x','center_y','size_x','size_y'):
                        m = re.search(rf"'{field}'\s*:\s*([0-9.+\-eE,]+)", bs)
                        if m:
                            val = m.group(1).rstrip(',')     # strip trailing comma
                            try:
                                nums[field] = float(val)
                            except ValueError:
                                # skip malformed numbers
                                continue
                    if len(nums) == 4:
                        pred_bboxes.append(nums)
                pred_dets = pred_bboxes

                # display bbox info
                # print(f"    Predicted bboxes: {pred_dets}")
                # gt_info   = gt_map.get(img, {'bboxes': []})
                # gt_bboxes = gt_info.get('bboxes', [])
                # print(f"    Ground truth bboxes: {gt_bboxes}")

                # compute IoU for each predicted bbox vs GT of same class
                for cls, pb in zip(pred_classes, pred_dets):
                    # convert pred box to corners
                    px1 = pb['center_x'] - pb['size_x']/2
                    py1 = pb['center_y'] - pb['size_y']/2
                    px2 = pb['center_x'] + pb['size_x']/2
                    py2 = pb['center_y'] + pb['size_y']/2
                    # match GT bboxes by class
                    for gd in bboxes:
                        if gd.get('class') != cls:
                            continue
                        # convert GT box to corners
                        gx1 = gd['center_x'] - gd['size_x']/2
                        gy1 = gd['center_y'] - gd['size_y']/2
                        gx2 = gd['center_x'] + gd['size_x']/2
                        gy2 = gd['center_y'] + gd['size_y']/2
                        # intersection
                        ix1 = max(px1, gx1)
                        iy1 = max(py1, gy1)
                        ix2 = min(px2, gx2)
                        iy2 = min(py2, gy2)
                        inter_w  = max(0.0, ix2 - ix1)
                        inter_h  = max(0.0, iy2 - iy1)
                        inter_a  = inter_w * inter_h
                        # union
                        area_p = (px2 - px1) * (py2 - py1)
                        area_g = (gx2 - gx1) * (gy2 - gy1)
                        union = area_p + area_g - inter_a
                        iou = inter_a / union if union > 0 else 0.0
                        print(f"    IoU ({cls}): {iou:.4f}")
                        # bonus +5 if IoU > 0
                        if iou > 0.0:
                            score += 5
                            total_score += 5
                            print(f"    +5 bonus for IoU>0 → Updated Score: {score}")

    avg = total_score / count if count else 0.0
    print(f"Total Score: {total_score}, Average Score per image: {avg:.2f}")
    if mse_count:
        print(f"Average BBox MSE (over {mse_count} imgs): {total_mse/mse_count:.4f}")
    else:
        print("No BBox MSE computed (no valid preds or no GT bboxes).")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval.py <validation.json> <message.txt>")
        sys.exit(1)
    file1, file2 = sys.argv[1], sys.argv[2]
    if not os.path.isfile(file1) or not os.path.isfile(file2):
        print("Both arguments must be valid file paths.")
        sys.exit(1)
    compare_json_files(file1, file2)
