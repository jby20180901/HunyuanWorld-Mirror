import argparse
import base64
import io
import os
import uuid
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except Exception as exc:
    raise RuntimeError(
        "worldmirror_service requires fastapi, pydantic and uvicorn. "
        "Install with: pip install fastapi uvicorn pydantic"
    ) from exc

from src.models.models.worldmirror import WorldMirror


DEFAULT_LOCAL_MODEL_PATH = "/home/jiangbaoyang/HuggingFace-Download-Accelerator/hf_hub/HunyuanWorld-Mirror"


class Build3DGSRequest(BaseModel):
    image_base64: str


class RenderPoseRequest(BaseModel):
    scene_id: str
    pose: list
    intrinsics: Optional[list] = None
    width: int
    height: int


class PredictPoseRequest(BaseModel):
    image_base64: str


class WorldMirrorService:
    def __init__(self, model_id: str = DEFAULT_LOCAL_MODEL_PATH, target_size: int = 518):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not model_id:
            raise ValueError("model_id cannot be empty")
        self.model = WorldMirror.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.target_size = target_size
        self.scene_store: Dict[str, Dict[str, Any]] = {}

    @torch.no_grad()
    def build_scene(self, image: Image.Image) -> Dict[str, Any]:
        tensor = self._image_to_tensor(image)
        views = {"img": tensor}

        use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        with torch.amp.autocast('cuda', enabled=bool(use_amp), dtype=amp_dtype):
            predictions = self.model(views=views, cond_flags=[0, 0, 0])

        if "splats" not in predictions:
            raise RuntimeError("WorldMirror predictions do not contain splats")

        scene_id = str(uuid.uuid4())
        splats = {
            key: value.detach().cpu()
            for key, value in predictions["splats"].items()
            if torch.is_tensor(value)
        }

        default_pose = predictions["camera_poses"][0, 0].detach().cpu()
        default_intrinsics = predictions["camera_intrs"][0, 0].detach().cpu()

        self.scene_store[scene_id] = {
            "splats": splats,
            "default_pose": default_pose,
            "default_intrinsics": default_intrinsics,
        }
        return {"scene_id": scene_id}

    @torch.no_grad()
    def render_pose(self, scene_id: str, pose: np.ndarray, intrinsics: Optional[np.ndarray], width: int, height: int) -> Image.Image:
        if scene_id not in self.scene_store:
            raise KeyError(f"scene_id not found: {scene_id}")

        scene = self.scene_store[scene_id]
        splats = {
            key: value.to(self.device)
            for key, value in scene["splats"].items()
        }

        if pose.shape != (4, 4):
            raise ValueError(f"pose must be [4,4], got {pose.shape}")
        pose_t = torch.from_numpy(pose).float().to(self.device).view(1, 1, 4, 4)

        if intrinsics is None:
            intr_t = scene["default_intrinsics"].to(self.device).view(1, 1, 3, 3)
        else:
            intr_np = np.asarray(intrinsics, dtype=np.float32)
            if intr_np.shape != (3, 3):
                raise ValueError(f"intrinsics must be [3,3], got {intr_np.shape}")
            intr_t = torch.from_numpy(intr_np).float().to(self.device).view(1, 1, 3, 3)

        colors, _, _ = self.model.gs_renderer.rasterizer.rasterize_batches(
            splats["means"],
            splats["quats"],
            splats["scales"],
            splats["opacities"],
            splats["sh"] if "sh" in splats else splats["colors"],
            pose_t,
            intr_t,
            width=int(width),
            height=int(height),
            sh_degree=min(self.model.gs_renderer.sh_degree, 0) if "sh" in splats else None,
        )

        rgb = colors[0, 0].detach().cpu().clamp(0, 1).numpy()
        rgb_u8 = (rgb * 255.0).astype(np.uint8)
        return Image.fromarray(rgb_u8)

    @torch.no_grad()
    def predict_pose(self, image: Image.Image) -> Dict[str, Any]:
        tensor = self._image_to_tensor(image)
        views = {"img": tensor}

        use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        with torch.amp.autocast('cuda', enabled=bool(use_amp), dtype=amp_dtype):
            predictions = self.model(views=views, cond_flags=[0, 0, 0])

        pose = predictions["camera_poses"][0, 0].detach().cpu().numpy().tolist()
        intr = predictions["camera_intrs"][0, 0].detach().cpu().numpy().tolist()
        return {"pose": pose, "intrinsics": intr}

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = image.resize((self.target_size, self.target_size), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)


def _decode_image(image_base64: str) -> Image.Image:
    raw = base64.b64decode(image_base64.encode("utf-8"))
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_app(service: WorldMirrorService) -> FastAPI:
    app = FastAPI(title="HunyuanWorld-Mirror Service", version="1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "device": str(service.device)}

    @app.post("/build_3dgs")
    def build_3dgs(req: Build3DGSRequest):
        try:
            image = _decode_image(req.image_base64)
            return service.build_scene(image)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/render_pose")
    def render_pose(req: RenderPoseRequest):
        try:
            pose = np.asarray(req.pose, dtype=np.float32)
            intr = None if req.intrinsics is None else np.asarray(req.intrinsics, dtype=np.float32)
            image = service.render_pose(
                scene_id=req.scene_id,
                pose=pose,
                intrinsics=intr,
                width=req.width,
                height=req.height,
            )
            return {"image_base64": _encode_image(image)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/predict_pose")
    def predict_pose(req: PredictPoseRequest):
        try:
            image = _decode_image(req.image_base64)
            return service.predict_pose(image)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return app


def main():
    parser = argparse.ArgumentParser(description="Run HunyuanWorld-Mirror local service for Lingbot")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--model_id", type=str, default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument("--target_size", type=int, default=518)
    args = parser.parse_args()

    if not os.path.exists(args.model_id):
        raise FileNotFoundError(
            f"Local model path not found: {args.model_id}. Please set --model_id to an existing local directory."
        )

    service = WorldMirrorService(model_id=args.model_id, target_size=args.target_size)
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
