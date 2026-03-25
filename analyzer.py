import cv2
import numpy as np
from PIL import Image
import io
from typing import Dict, Optional


class FaceColorAnalyzer:

    def analyze(self, image_bytes: bytes) -> Dict:
        try:
            img = self._bytes_to_cv2(image_bytes)
            if img is None:
                return {"face_detected": False, "confidence": 0}
            return self._analyze_simple(img)
        except Exception as e:
            return {"face_detected": False, "confidence": 0, "error": str(e)}

    def _bytes_to_cv2(self, image_bytes: bytes):
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_array = np.array(pil_img)
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _analyze_simple(self, img) -> Dict:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))

        if len(faces) == 0:
            return {"face_detected": False, "confidence": 0}

        x, y, fw, fh = faces[0]
        face_roi = img[y:y+fh, x:x+fw]
        h, w = face_roi.shape[:2]

        forehead = face_roi[int(h*0.05):int(h*0.25), int(w*0.3):int(w*0.7)]
        cheek_l  = face_roi[int(h*0.4):int(h*0.65), int(w*0.05):int(w*0.3)]
        cheek_r  = face_roi[int(h*0.4):int(h*0.65), int(w*0.7):int(w*0.95)]
        eye_roi  = face_roi[int(h*0.25):int(h*0.42), int(w*0.15):int(w*0.85)]
        lip_roi  = face_roi[int(h*0.68):int(h*0.88), int(w*0.3):int(w*0.7)]

        hair_y1 = max(0, y - int(fh * 0.2))
        hair_roi = img[hair_y1:y, x:x+fw]

        skin_light = self._dominant_color(forehead)
        cheek_avg  = self._average_color([
            self._dominant_color(cheek_l),
            self._dominant_color(cheek_r)
        ])
        skin_dark  = cheek_avg
        hair_color = self._dominant_color(hair_roi) if hair_roi.size > 0 else None
        eye_color  = self._dominant_color(eye_roi)
        lip_color  = self._dominant_color(lip_roi)

        confidence = self._evaluate_lighting(img)

        return {
            "face_detected": True,
            "confidence": confidence,
            "skin_light": self._to_hex(skin_light),
            "skin_dark":  self._to_hex(skin_dark),
            "hair":       self._to_hex(hair_color),
            "eye":        self._to_hex(eye_color),
            "lip":        self._to_hex(lip_color),
        }

    def analyze_face_shape(self, image_bytes: bytes) -> Optional[str]:
        try:
            img = self._bytes_to_cv2(image_bytes)
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
            if len(faces) == 0:
                return None
            x, y, fw, fh = faces[0]
            ratio = fh / fw
            if ratio > 1.4:
                return "long"
            elif ratio < 1.05:
                return "round"
            elif ratio < 1.15:
                return "square"
            else:
                return "oval"
        except Exception:
            return None

    def _dominant_color(self, region):
        if region is None or region.size == 0:
            return None
        try:
            pixels = region.reshape(-1, 3).astype(np.float32)
            if len(pixels) < 3:
                mean = pixels.mean(axis=0)
                return (int(mean[2]), int(mean[1]), int(mean[0]))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, min(3, len(pixels)), None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
            counts = np.bincount(labels.flatten())
            dominant = centers[np.argmax(counts)]
            return (int(dominant[2]), int(dominant[1]), int(dominant[0]))
        except Exception:
            mean = region.mean(axis=(0, 1))
            return (int(mean[2]), int(mean[1]), int(mean[0]))

    def _average_color(self, colors):
        colors = [c for c in colors if c is not None]
        if not colors:
            return None
        r = sum(c[0] for c in colors) // len(colors)
        g = sum(c[1] for c in colors) // len(colors)
        b = sum(c[2] for c in colors) // len(colors)
        return (r, g, b)

    def _to_hex(self, color):
        if color is None:
            return None
        return "#{:02X}{:02X}{:02X}".format(color[0], color[1], color[2])

    def _evaluate_lighting(self, img) -> float:
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_mean = lab[:, :, 0].mean()
            score = 1.0 - abs(l_mean - 120) / 120
            return float(np.clip(score, 0.4, 1.0))
        except Exception:
            return 0.65
