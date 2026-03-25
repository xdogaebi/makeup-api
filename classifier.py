import math
from typing import Dict, List, Optional


def hex_to_rgb(hex_str: str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_lab(r, g, b):
    def linearize(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    lr, lg, lb = linearize(r), linearize(g), linearize(b)
    x = lr*0.4124 + lg*0.3576 + lb*0.1805
    y = lr*0.2126 + lg*0.7152 + lb*0.0722
    z = lr*0.0193 + lg*0.1192 + lb*0.9505
    def f(t):
        return t**(1/3) if t > 0.008856 else 7.787*t + 16/116
    fx, fy, fz = f(x/0.9505), f(y/1.0), f(z/1.0890)
    return 116*fy - 16, 500*(fx - fy), 200*(fy - fz)


TYPES = {
    "spring-warm":   {"name":"봄 웜",      "eng":"True Spring",   "season":"spring", "undertone":"warm", "contrast":["low","medium"],    "clarity":"clear",  "brightness":["high","very_high"]},
    "spring-bright": {"name":"봄 브라이트", "eng":"Bright Spring", "season":"spring", "undertone":"warm", "contrast":["medium","high"],   "clarity":"clear",  "brightness":["high","very_high"]},
    "spring-light":  {"name":"봄 라이트",   "eng":"Light Spring",  "season":"spring", "undertone":"warm", "contrast":["low"],             "clarity":"clear",  "brightness":["very_high"]},
    "summer-light":  {"name":"여름 라이트", "eng":"Light Summer",  "season":"summer", "undertone":"cool", "contrast":["low"],             "clarity":"muted",  "brightness":["very_high","high"]},
    "summer-mute":   {"name":"여름 뮤트",   "eng":"Soft Summer",   "season":"summer", "undertone":"cool", "contrast":["low","medium"],    "clarity":"muted",  "brightness":["high","medium"]},
    "summer-cool":   {"name":"여름 쿨",     "eng":"True Summer",   "season":"summer", "undertone":"cool", "contrast":["medium"],          "clarity":"clear",  "brightness":["high","medium"]},
    "autumn-warm":   {"name":"가을 웜",     "eng":"True Autumn",   "season":"autumn", "undertone":"warm", "contrast":["medium"],          "clarity":"muted",  "brightness":["medium","low"]},
    "autumn-mute":   {"name":"가을 뮤트",   "eng":"Soft Autumn",   "season":"autumn", "undertone":"warm", "contrast":["low","medium"],    "clarity":"muted",  "brightness":["medium"]},
    "autumn-deep":   {"name":"가을 딥",     "eng":"Deep Autumn",   "season":"autumn", "undertone":"warm", "contrast":["high","medium"],   "clarity":"muted",  "brightness":["low","very_low"]},
    "winter-deep":   {"name":"겨울 딥",     "eng":"Deep Winter",   "season":"winter", "undertone":"cool", "contrast":["high"],            "clarity":"clear",  "brightness":["low","very_low","medium"]},
    "winter-bright": {"name":"겨울 브라이트","eng":"Bright Winter","season":"winter", "undertone":"cool", "contrast":["high","medium"],   "clarity":"clear",  "brightness":["high","medium"]},
    "winter-cool":   {"name":"겨울 쿨",     "eng":"True Winter",   "season":"winter", "undertone":"cool", "contrast":["high"],            "clarity":"clear",  "brightness":["high","very_high"]},
}


class PersonalColorClassifier:

    def classify(self, color_result: Dict) -> Dict:
        skin_light = color_result.get("skin_light")
        skin_dark  = color_result.get("skin_dark")
        hair       = color_result.get("hair")
        lip        = color_result.get("lip")

        undertone  = self._get_undertone(skin_light, skin_dark, lip)
        contrast   = self._get_contrast(skin_light, hair)
        clarity    = self._get_clarity(skin_light)
        brightness = self._get_brightness(skin_light)

        scores = {}
        for tid, info in TYPES.items():
            score = 0
            if undertone == info["undertone"]: score += 40
            elif undertone == "neutral":        score += 15
            if contrast in info["contrast"]:    score += 25
            if clarity == info["clarity"]:      score += 20
            if brightness in info["brightness"]:score += 15
            scores[tid] = score

        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_id = sorted_types[0][0]
        best_info = TYPES[best_id]

        secondary = None
        if len(sorted_types) > 1:
            second_id = sorted_types[1][0]
            if (scores[best_id] - scores[second_id]) <= 15:
                secondary = {"type_id": second_id, "type_name": TYPES[second_id]["name"]}

        return {
            "type_id": best_id,
            "type_name": best_info["name"],
            "type_name_en": best_info["eng"],
            "season": best_info["season"],
            "undertone": undertone,
            "contrast_level": contrast,
            "clarity": clarity,
            "brightness": brightness,
            "secondary_type": secondary,
        }

    def get_all_types(self) -> List[Dict]:
        return [{"id": tid, "name": i["name"], "eng": i["eng"], "season": i["season"]} for tid, i in TYPES.items()]

    def _get_undertone(self, skin_light, skin_dark, lip) -> str:
        scores = []
        for h in [skin_light, skin_dark, lip]:
            if not h: continue
            r, g, b = hex_to_rgb(h)
            L, a, b_lab = rgb_to_lab(r, g, b)
            scores.append(b_lab - a * 0.5)
        if not scores: return "neutral"
        avg = sum(scores) / len(scores)
        if avg > 8: return "warm"
        elif avg < 2: return "cool"
        return "neutral"

    def _get_contrast(self, skin_light, hair) -> str:
        if not skin_light or not hair: return "medium"
        r1,g1,b1 = hex_to_rgb(skin_light)
        r2,g2,b2 = hex_to_rgb(hair)
        L1,_,_ = rgb_to_lab(r1,g1,b1)
        L2,_,_ = rgb_to_lab(r2,g2,b2)
        delta = abs(L1 - L2)
        if delta >= 55: return "high"
        elif delta >= 30: return "medium"
        return "low"

    def _get_clarity(self, skin_light) -> str:
        if not skin_light: return "muted"
        r,g,b = hex_to_rgb(skin_light)
        L,a,b_lab = rgb_to_lab(r,g,b)
        chroma = math.sqrt(a**2 + b_lab**2)
        return "clear" if chroma > 18 else "muted"

    def _get_brightness(self, skin_light) -> str:
        if not skin_light: return "medium"
        r,g,b = hex_to_rgb(skin_light)
        L,_,_ = rgb_to_lab(r,g,b)
        if L >= 72: return "very_high"
        elif L >= 58: return "high"
        elif L >= 44: return "medium"
        elif L >= 30: return "low"
        return "very_low"
```

---

**파일 4 — requirements.txt**
```
fastapi==0.111.0
uvicorn==0.29.0
python-multipart==0.0.9
opencv-python-headless==4.9.0.80
numpy==1.26.4
Pillow==10.3.0
