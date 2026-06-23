from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import tempfile, os
from PIL import Image, ImageEnhance

app = Flask(__name__)
ocr = None

@app.route("/ocr", methods=["POST"])
def do_ocr():
    global ocr
    try:
        if ocr is None:
            ocr = PaddleOCR(
                lang="ch",
                use_angle_cls=False,
                det_db_thresh=0.2,
                det_db_box_thresh=0.4,
                det_db_unclip_ratio=1.8,
            )
            print("PaddleOCR initialized")
        file = request.files.get("image")
        if not file: return jsonify({"error": "no image"}), 400
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        file.save(tmp.name); tmp.close()
        try:
            Image.open(tmp.name).verify()
        except:
            os.unlink(tmp.name)
            return jsonify({"error": "invalid image"}), 400
        result = ocr.ocr(tmp.name)
        lines = []
        lines_data = []
        if result and result[0]:
            for item in result[0]:
                if item and len(item) >= 2:
                    bbox = item[0]
                    text, conf = item[1][0], item[1][1]
                    if conf > 0.5:
                        lines.append(text)
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        lines_data.append({
                            "text": text,
                            "confidence": round(conf, 2),
                            "x": round(min(xs)), "y": round(min(ys)),
                            "w": round(max(xs) - min(xs)),
                            "h": round(max(ys) - min(ys)),
                            "cx": round(sum(xs)/4, 1),
                            "cy": round(sum(ys)/4, 1),
                        })
        os.unlink(tmp.name)
        return jsonify({
            "text": "\n".join(lines),
            "lines": len(lines),
            "lines_data": lines_data,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8899)
