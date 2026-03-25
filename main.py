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
import httpx

@app.post("/apply/makeup")
async def apply_makeup(
    image: UploadFile = File(...),
    lip_color: Optional[str] = Form(default=None),
    blush_color: Optional[str] = Form(default=None),
    eye_color: Optional[str] = Form(default=None),
):
    import base64
    contents = await image.read()
    image_b64 = base64.b64encode(contents).decode()
    image_data_url = f"data:image/jpeg;base64,{image_b64}"

    replicate_token = os.getenv("REPLICATE_API_TOKEN", "")

    payload = {
        "version": "7af9a66f36f97fee2fece7dcc927551a951f0022d13463328f06c694d6e1b3a0",
        "input": {
            "image": image_data_url,
            "lip_color": lip_color or "#C0395A",
            "eyeshadow_color": eye_color or "#5B4A8A",
            "blush_color": blush_color or "#C85A7A",
        }
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {replicate_token}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        result = response.json()

        prediction_id = result.get("id")
        for _ in range(30):
            import asyncio
            await asyncio.sleep(3)
            poll = await client.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers={"Authorization": f"Token {replicate_token}"}
            )
            poll_result = poll.json()
            if poll_result.get("status") == "succeeded":
                return {
                    "success": True,
                    "output_image": poll_result.get("output")
                }
            elif poll_result.get("status") == "failed":
                return {"success": False, "error": "메이크업 적용 실패"}

    return {"success": False, "error": "타임아웃"}
```

**Commit changes** 클릭

---

**STEP 4 — Render에 API 토큰 환경변수 추가**

Render → makeup-api → **Environment** → **Add Environment Variable**

- Key: `REPLICATE_API_TOKEN`
- Value: STEP 2에서 복사한 토큰

**Save Changes** → **Manual Deploy** → **Deploy latest commit**

---

**STEP 5 — Base44 채팅창에 입력**
```
메이크업 시뮬레이터에 Replicate AI 메이크업 합성 기능을 추가해줘.

컬러 스워치를 클릭하면:
1. 선택한 컬러 HEX값을 POST https://makeup-api-30zj.onrender.com/apply/makeup 으로 전송
   - image: 업로드된 사진
   - lip_color: 선택한 립 컬러 HEX (입술 탭일 때)
   - blush_color: 선택한 블러셔 컬러 HEX (블러셔 탭일 때)
   - eye_color: 선택한 아이섀도우 컬러 HEX (아이섀도우 탭일 때)
2. 로딩 중 "AI가 메이크업을 적용하고 있어요..." 표시
3. 응답의 output_image URL을 원본 사진 자리에 표시
4. "다시 선택" 버튼으로 원본 사진으로 돌아가기
