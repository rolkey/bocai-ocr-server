#!/usr/bin/env python3
# OCR 服务 — PaddleX PP-OCRv6 管线
# 运行: /root/ocr-paddlex-venv/bin/python3 ocr_server.py
from flask import Flask, request, jsonify
from paddlex import create_pipeline
import tempfile, os, re, threading
from PIL import Image

app = Flask(__name__)
pipeline = None
ocr_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════
#  生肖知识库
# ═══════════════════════════════════════════════════════════

ZODIAC_ANIMALS = ["鼠", "牛", "虎", "兔", "龍", "龙", "蛇", "馬", "马", "羊", "猴", "雞", "鸡", "狗", "豬", "猪"]

# OCR 易混字符修正（特定字体下模型误识别 → 实际生肖字）
OCR_CORRECTIONS = {
    "免": "兔",
    "雌": "蛇",
    "鬼": "兔",
    "美": "羊",
    "炖": "马",
    "逸": "兔",
    "洋": "羊",
    "鸿": "鸡",
    "赢": "鼠",
}


def _correct_zodiac_text(text: str) -> str:
    has_zodiac = any(z in text for z in ZODIAC_ANIMALS)
    if not has_zodiac:
        return text
    for wrong, right in OCR_CORRECTIONS.items():
        if wrong in text:
            text = text.replace(wrong, right)
    for trad, simp in ZODIAC_T2S.items():
        text = text.replace(trad, simp)
    return text

ZODIAC_NUM = {
    "鼠": 1, "牛": 2, "虎": 3, "兔": 4, "龍": 5, "龙": 5,
    "蛇": 6, "馬": 7, "马": 7, "羊": 8, "猴": 9,
    "雞": 10, "鸡": 10, "狗": 11, "豬": 12, "猪": 12,
}

# 繁体→简体 生肖映射（OCR 输出统一为简体）
ZODIAC_T2S = {"龍": "龙", "馬": "马", "雞": "鸡", "豬": "猪", "兎": "兔"}

ZODIAC_ALIASES = {
    "老鼠": "鼠", "水牛": "牛", "黄牛": "牛", "老虎": "虎", "白兔": "兔",
    "青龙": "龙", "金龙": "龙", "飞龙": "龙", "小龙": "蛇", "花蛇": "蛇",
    "红马": "马", "黑马": "马", "白马": "马", "山羊": "羊", "绵羊": "羊",
    "金猴": "猴", "火猴": "猴", "公鸡": "鸡", "母鸡": "鸡", "黑狗": "狗",
    "黄狗": "狗", "白猪": "猪", "黑猪": "猪", "花猪": "猪",
}

WAVE_COLORS = {"红波": "red", "蓝波": "blue", "绿波": "green"}

ELEMENTS = ["金", "木", "水", "火", "土"]

LOTTERY_KEYWORDS = [
    "特码", "正码", "平码", "头数", "尾数", "生肖", "波色", "五行",
    "一肖", "二肖", "三肖", "四肖", "五肖", "六肖",
    "一码", "二码", "三码", "四码", "五码", "六码",
    "中特", "不中", "澳彩", "港彩", "六合彩",
]

# ═══════════════════════════════════════════════════════════
#  生肖分析函数
# ═══════════════════════════════════════════════════════════

def is_zodiac_related(text):
    """判断文本是否与生肖相关"""
    for animal in ZODIAC_ANIMALS:
        if animal in text:
            return True
    for alias in ZODIAC_ALIASES:
        if alias in text:
            return True
    for wc in WAVE_COLORS:
        if wc in text:
            return True
    for elem in ELEMENTS:
        if elem in text:
            return True
    for kw in LOTTERY_KEYWORDS:
        if kw in text:
            return True
    return False


def extract_zodiac_animals(text):
    """从文本中提取所有生肖动物"""
    found = []
    # 先检查别名（长词优先）
    for alias, animal in ZODIAC_ALIASES.items():
        if alias in text:
            found.append({"alias": alias, "zodiac": animal, "num": ZODIAC_NUM[animal]})
    # 再检查单字，避免别名已覆盖的情况
    for ch in text:
        if ch in ZODIAC_ANIMALS:
            already = any(f["zodiac"] == ch for f in found)
            is_part_of_alias = any(ch in f.get("alias", "") for f in found if f.get("alias"))
            if not already and not is_part_of_alias:
                found.append({"alias": None, "zodiac": ch, "num": ZODIAC_NUM[ch]})
    return found


def extract_numbers(text):
    return [int(n) for n in re.findall(r'\d+', text)]


def extract_wave_colors(text):
    return [{"cn": wc, "en": en} for wc, en in WAVE_COLORS.items() if wc in text]


def extract_elements(text):
    return [e for e in ELEMENTS if e in text]


def extract_keywords(text):
    return [kw for kw in LOTTERY_KEYWORDS if kw in text]


def build_zodiac_summary(lines_data):
    """基于 OCR 行数据构建生肖分析摘要"""
    all_zodiacs, seen_z = [], set()
    all_colors, seen_c = [], set()
    all_elements, seen_e = [], set()
    all_keywords, seen_k = [], set()
    zodiac_lines = []

    for item in lines_data:
        text = item["text"]
        if not is_zodiac_related(text):
            continue
        zodiac_lines.append(item)

        # 为每行追加生肖分析字段
        item["zodiac_animals"] = extract_zodiac_animals(text)
        item["numbers"] = extract_numbers(text)
        item["wave_colors"] = extract_wave_colors(text)
        item["elements"] = extract_elements(text)
        item["keywords"] = extract_keywords(text)

        for za in item["zodiac_animals"]:
            if za["zodiac"] not in seen_z:
                seen_z.add(za["zodiac"])
                all_zodiacs.append(za)
        for wc in item["wave_colors"]:
            if wc["cn"] not in seen_c:
                seen_c.add(wc["cn"])
                all_colors.append(wc)
        for e in item["elements"]:
            if e not in seen_e:
                seen_e.add(e)
                all_elements.append(e)
        for kw in item["keywords"]:
            if kw not in seen_k:
                seen_k.add(kw)
                all_keywords.append(kw)

    # 为不相关的行补空字段
    for item in lines_data:
        if "zodiac_animals" not in item:
            item["zodiac_animals"] = []
            item["numbers"] = []
            item["wave_colors"] = []
            item["elements"] = []
            item["keywords"] = []

    return {
        "zodiacs": all_zodiacs,
        "wave_colors": all_colors,
        "elements": all_elements,
        "keywords": all_keywords,
        "zodiac_count": len(all_zodiacs),
        "primary_zodiac": all_zodiacs[0] if all_zodiacs else None,
        "zodiac_lines": len(zodiac_lines),
    }


# ═══════════════════════════════════════════════════════════
#  合并同行拆分文本框 (原有的第2遍逻辑)
# ═══════════════════════════════════════════════════════════

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
            group = [row[i]]
            j = i + 1
            while j < len(row):
                prev = group[-1][1]
                curr = row[j][1]
                gap = curr['x'] - (prev['x'] + prev['w'])
                if gap < 10:
                    group.append(row[j])
                    j += 1
                else:
                    break
            if len(group) >= 2:
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


# ═══════════════════════════════════════════════════════════
#  API 路由
# ═══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return jsonify({
        "service": "生肖文字 OCR 识别服务",
        "version": "3.0.0",
        "engine": "PaddleX PP-OCRv6 medium",
        "endpoints": {
            "POST /ocr": "上传图片 OCR 识别（含生肖分析）",
            "POST /ocr/extract": "精简模式（仅生肖摘要）",
            "GET /health": "健康检查",
        },
    })


@app.route("/health")
def health():
    global pipeline
    return jsonify({
        "status": "ok",
        "pipeline_ready": pipeline is not None,
    })


def _paddlex_to_lines(ocr_result) -> list[dict]:
    """Convert PaddleX OCRResult to legacy lines_data format."""
    j = ocr_result.json
    res = j.get("res", {})
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])
    polys = res.get("rec_polys", [])

    lines = []
    for text, score, poly in zip(texts, scores, polys):
        if score < 0.5:
            continue
        if poly and len(poly) >= 4:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            lines.append({
                "text": text,
                "confidence": round(score, 2),
                "x": round(min(xs)),
                "y": round(min(ys)),
                "w": round(max(xs) - min(xs)),
                "h": round(max(ys) - min(ys)),
                "cx": round(sum(xs) / 4, 1),
                "cy": round(sum(ys) / 4, 1),
            })
    return lines


@app.route("/ocr", methods=["POST"])
def do_ocr():
    global pipeline
    with ocr_lock:
        try:
            if pipeline is None:
                pipeline = create_pipeline(config={
                    "pipeline_name": "OCR",
                    "text_type": "general",
                    "use_doc_preprocessor": False,
                    "use_textline_orientation": False,
                    "SubModules": {
                        "TextDetection": {
                            "model_name": "PP-OCRv6_medium_det",
                            "limit_side_len": 64,
                            "limit_type": "min",
                            "thresh": 0.3,
                            "box_thresh": 0.6,
                            "unclip_ratio": 1.5,
                        },
                        "TextRecognition": {
                            "model_name": "PP-OCRv6_medium_rec",
                            "score_thresh": 0.0,
                            "return_word_box": False,
                        },
                    },
                    "SubPipelines": {},
                })
                print("PaddleX OCR pipeline initialized (PP-OCRv6 medium, doc_preprocessor=off)")

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

            # PaddleX OCR 管线（含文档方向矫正 + 去扭曲 + 文字行方向矫正）
            lines_data = []
            for result in pipeline.predict(tmp.name):
                lines_data = _paddlex_to_lines(result)
            os.unlink(tmp.name)

            # 生肖分析（逻辑不变）
            for item in lines_data:
                item["text"] = _correct_zodiac_text(item["text"])
            summary = build_zodiac_summary(lines_data)
            lines_text = [item["text"] for item in lines_data]

            return jsonify({
                "text": "\n".join(lines_text),
                "lines": len(lines_text),
                "lines_data": lines_data,
                "summary": summary,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route("/ocr/extract", methods=["POST"])
def do_ocr_extract():
    global pipeline
    with ocr_lock:
        try:
            if pipeline is None:
                pipeline = create_pipeline(pipeline="OCR")

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

            lines_data = []
            for result in pipeline.predict(tmp.name):
                lines_data = _paddlex_to_lines(result)
            os.unlink(tmp.name)

            for item in lines_data:
                item["text"] = _correct_zodiac_text(item["text"])

            summary = build_zodiac_summary(lines_data)
            all_text = "".join(item["text"] for item in lines_data)

            return jsonify({
                "all_text": all_text,
                "summary": summary,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8899)
