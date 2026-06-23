from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import tempfile, os
from PIL import Image

app = Flask(__name__)
ocr = None

# ── 合并同行拆分文本框 ──
def merge_adjacent(lines_data, image_path):
    """
    第2遍：按行分组 → 同组相邻框合并裁剪 → 重OCR。
    解决 PP-OCRv4 det 在低分辨率下把同行文字切成多块的问题。
    """
    if not ocr or len(lines_data) < 2:
        return lines_data

    # ── 按行分组 ──
    items = [(i, d) for i, d in enumerate(lines_data)]
    items.sort(key=lambda t: t[1]['cy'])

    rows = []  # [[(idx, data), ...], ...]
    for idx, d in items:
        placed = False
        for row in rows:
            rep = row[0][1]
            if abs(d['cy'] - rep['cy']) < (d['h'] + rep['h']) * 0.4:
                row.append((idx, d))
                placed = True
                break
        if not placed:
            rows.append([(idx, d)])

    # ── 每组内：相邻、靠近/重叠的框合并 ──
    img = Image.open(image_path)
    pw, ph = img.size
    merged = list(lines_data)
    to_drop = set()

    for row in rows:
        if len(row) < 2:
            continue
        row.sort(key=lambda t: t[1]['x'])
        i = 0
        while i < len(row) - 1:
            # 收集当前可合并的连续框
            group = [row[i]]
            j = i + 1
            while j < len(row):
                prev = group[-1][1]
                curr = row[j][1]
                gap = curr['x'] - (prev['x'] + prev['w'])
                # 只合并重叠或紧邻的框（绝对间距 < 10px）
                if gap < 10:
                    group.append(row[j])
                    j += 1
                else:
                    break
            if len(group) >= 2:
                # 裁剪合并区域
                min_x = min(d['x'] for _, d in group)
                max_xe = max(d['x'] + d['w'] for _, d in group)
                min_y = min(d['y'] for _, d in group)
                max_ye = max(d['y'] + d['h'] for _, d in group)
                x1, y1 = max(0, min_x - 3), max(0, min_y - 3)
                x2, y2 = min(pw, max_xe + 3), min(ph, max_ye + 3)

                crop = img.crop((x1, y1, x2, y2))
                fd, crop_path = tempfile.mkstemp(suffix='.jpg')
                os.close(fd)
                crop.save(crop_path)

                r = ocr.ocr(crop_path)
                if r and r[0]:
                    best = max(r[0], key=lambda it: it[1][1])
                    if best[1][1] > 0.5:
                        bbox, (text, conf) = best
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        keep_idx = group[0][0]
                        merged[keep_idx] = {
                            'text': text,
                            'confidence': round(conf, 2),
                            'x': x1 + round(min(xs)),
                            'y': y1 + round(min(ys)),
                            'w': round(max(xs) - min(xs)),
                            'h': round(max(ys) - min(ys)),
                            'cx': x1 + round(sum(xs) / 4, 1),
                            'cy': y1 + round(sum(ys) / 4, 1),
                        }
                        for gidx, _ in group[1:]:
                            to_drop.add(gidx)
                os.unlink(crop_path)
            i = j

    return [item for i, item in enumerate(merged) if i not in to_drop]


@app.route("/ocr", methods=["POST"])
def do_ocr():
    global ocr
    try:
        if ocr is None:
            ocr = PaddleOCR(
                lang="ch",
                use_angle_cls=True,
                det_db_thresh=0.2,
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.8,
            )
            print("PaddleOCR initialized (PP-OCRv4 mobile, tuned + merge)")

        file = request.files.get("image")
        if not file:
            return jsonify({"error": "no image"}), 400

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        file.save(tmp.name)
        tmp.close()

        try:
            Image.open(tmp.name).verify()
        except Exception:
            os.unlink(tmp.name)
            return jsonify({"error": "invalid image"}), 400

        # ── 第1遍：全图 OCR ──
        result = ocr.ocr(tmp.name)
        lines_data = []
        if result and result[0]:
            for item in result[0]:
                if item and len(item) >= 2:
                    bbox = item[0]
                    text, conf = item[1][0], item[1][1]
                    if conf > 0.5:
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        lines_data.append({
                            "text": text,
                            "confidence": round(conf, 2),
                            "x": round(min(xs)),
                            "y": round(min(ys)),
                            "w": round(max(xs) - min(xs)),
                            "h": round(max(ys) - min(ys)),
                            "cx": round(sum(xs) / 4, 1),
                            "cy": round(sum(ys) / 4, 1),
                        })

        # ── 第2遍：合并同行拆分框 ──
        lines_data = merge_adjacent(lines_data, tmp.name)

        os.unlink(tmp.name)

        lines = [item["text"] for item in lines_data]
        return jsonify({
            "text": "\n".join(lines),
            "lines": len(lines),
            "lines_data": lines_data,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8899)
