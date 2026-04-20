import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from src.flow_estimator import FarnebackFlow
from src.visualizer import visualize_dense_flow

app = FastAPI(title="Optical Flow API")

_fb = FarnebackFlow()


def _decode_gray(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return frame


@app.post(
    "/set-reference",
    summary="Store a reference frame for subsequent flow requests",
)
async def set_reference(image: UploadFile = File(...)):
    app.state.reference = _decode_gray(await image.read())
    h, w = app.state.reference.shape
    return {"status": "ok", "width": w, "height": h}


@app.post(
    "/flow",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
    summary="Compute Farneback flow against the stored reference, return JPEG visualization",
)
async def compute_flow(image: UploadFile = File(...)):
    if not hasattr(app.state, "reference") or app.state.reference is None:
        raise HTTPException(
            status_code=400,
            detail="No reference frame set. POST an image to /set-reference first.",
        )

    curr = _decode_gray(await image.read())
    ref = app.state.reference

    # Silently align resolution if caller uploads a different size
    if curr.shape != ref.shape:
        curr = cv2.resize(curr, (ref.shape[1], ref.shape[0]))

    flow = _fb.compute(ref, curr)
    vis = visualize_dense_flow(flow)

    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode response image")

    return Response(content=buf.tobytes(), media_type="image/jpeg")
