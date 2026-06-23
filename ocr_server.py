from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import tempfile, os
from PIL import Image

app = Flask(__name__)
ocr = None

# ── 合并同行拆分文本框 ──
def merge_adjacent(lines_data, image_path):
    """
    第2遍：检测同行、间距小的文本框，裁剪合并区域重OCR。
    解决 PP-OCRv4 det 模型在低分辨率下把同行文字切成多块的问题。
    """
    if not ocr or len(lines_data) < 2:
        return lines_data

    # 按 y 中心排序
    indexed = sorted(enumerate(lines_data), key=lambda t: t[1]['cy'])
    pairs = []
    for i in range(len(indexed) - 1):
        ia, a = indexed[i]
        ib, b = indexed[i + 1]
        # 同行：y 中心差 < 平均高度
        same_row = abs(a['cy'] - b['cy']) < (a['h'] + b['h']) / 2
        # 间距：水平间隔 > 0 且 < 平均宽度
        gap = b['x'] - (a['x'] + a['w'])
        close = 0 < gap < (a['w'] + b['w']) / 2
        if same_row and close:
            pairs.append((ia, ib))

    if not pairs:
        return lines_data

    img = Image.open(image_path)
    pw, ph = img.size
    merged = list(lines_data)
    to_drop = set()

    for ia, ib in pairs:
        if ia in to_drop or ib in to_drop:
            continue
        a, b = lines_data[ia], lines_data[ib]
        # 裁剪合并区域（向外扩 3px）
        x1 = max(0, min(a['x'], b['x']) - 3)
        y1 = max(0, min(a['y'], b['y']) - 3)
        x2 = min(pw, max(a['x'] + a['w'], b['x'] + b['w']) + 3)
        y2 = min(ph, max(a['y'] + a['h'], b['y'] + b['h']) + 3)

        crop = img.crop((x1, y1, x2, y2))
        fd, crop_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        crop.save(crop_path)

        r = ocr.ocr(crop_path)
        if r and r[0]:
            # 取置信度最高的行
            best = max(r[0], key=lambda it: it[1][1])
            if best[1][1] > 0.5:
                bbox, (text, conf) = best
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                merged[ia] = {
                    'text': text,
                    'confidence': round(conf, 2),
                    'x': x1 + round(min(xs)),
                    'y': y1 + round(min(ys)),
                    'w': round(max(xs) - min(xs)),
                    'h': round(max(ys) - min(ys)),
                    'cx': x1 + round(sum(xs) / 4, 1),
                    'cy': y1 + round(sum(ys) / 4, 1),
                }
                to_drop.add(ib)
        os.unlink(crop_path)

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
