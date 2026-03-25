from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import json
from typing import Optional
from analyzer import FaceColorAnalyzer
from classifier import PersonalColorClassifier

app = FastAPI(title="퍼스널컬러 분석 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = FaceColorAnalyzer()
classifier = PersonalColorClassifier()

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

@app.get("/types")
async def get_types():
    return {"success": True, "types": classifier.get_all_types()}

@app.post("/analyze/face")
async def analyze_face(
    image: UploadFile = File(...),
    options: Optional[str] = Form(default="{}")
):
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="JPEG 또는 PNG 파일을 사용해 주세요.")

    contents = await image.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="파일 크기는 10MB 이하여야 합니다.")

    try:
        opts = json.loads(options)
    except Exception:
        opts = {}

    try:
        color_result = analyzer.analyze(contents)

        if not color_result["face_detected"]:
            return JSONResponse(status_code=422, content={
                "success": False,
                "error": "얼굴을 감지하지 못했습니다. 정면 사진을 사용해 주세요.",
                "error_code": "FACE_NOT_DETECTED"
            })

        type_result = classifier.classify(color_result)
        face_shape = analyzer.analyze_face_shape(contents)

        return {
            "success": True,
            "type_id": type_result["type_id"],
            "type_name": type_result["type_name"],
            "type_name_en": type_result["type_name_en"],
            "season": type_result["season"],
            "undertone": type_result["undertone"],
            "confidence": round(color_result["confidence"] * 100),
            "colors": {
                "skin_light": color_result["skin_light"],
                "skin_dark": color_result["skin_dark"],
                "hair": color_result["hair"],
                "eye": color_result["eye"],
                "lip": color_result["lip"],
            },
            "analysis": {
                "contrast_level": type_result["contrast_level"],
                "clarity": type_result["clarity"],
                "brightness": type_result["brightness"],
            },
            "face_shape": face_shape,
            "secondary_type": type_result.get("secondary_type"),
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": f"분석 중 오류가 발생했습니다: {str(e)}",
            "error_code": "ANALYSIS_ERROR"
        })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
