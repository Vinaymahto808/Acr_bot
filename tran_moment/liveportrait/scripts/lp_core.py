"""LivePortrait on CPU for Neha's portrait, without InsightFace (its buffalo_l models are non-commercial).

The face crop comes from our own landmarks (tran_moment/prepared/face.json, MediaPipe-derived) as a fixed square
box; LivePortrait's 203-point landmark model (landmark.onnx, part of the MIT LivePortrait release) only measures the
source eye / lip ratios for the retargeting modules. Nothing in the downloaded repo is executed as a script: this
module imports its network classes by explicit path.

    tran_moment/liveportrait/.venv/Scripts/python.exe -I tran_moment/liveportrait/scripts/<script>.py --repo ...
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


class LP:
    def __init__(self, repo: Path, threads: int = 8):
        sys.path.insert(0, str(repo))
        import torch  # noqa: PLC0415
        torch.set_num_threads(threads)
        from src.config.inference_config import InferenceConfig  # noqa: PLC0415
        from src.live_portrait_wrapper import LivePortraitWrapper  # noqa: PLC0415
        from src.utils.camera import get_rotation_matrix  # noqa: PLC0415
        from src.utils.human_landmark_runner import LandmarkRunner  # noqa: PLC0415
        self.torch, self.rot = torch, get_rotation_matrix
        cfg = InferenceConfig()
        cfg.flag_force_cpu = True
        cfg.flag_use_half_precision = False
        cfg.flag_do_torch_compile = False
        t0 = time.time()
        self.w = LivePortraitWrapper(cfg)
        self.lmk_runner = LandmarkRunner(ckpt_path=str(repo / "pretrained_weights" / "liveportrait" / "landmark.onnx"), onnx_provider="cpu")
        self.load_s = time.time() - t0

    def prepare(self, crop_rgb: np.ndarray):
        """crop_rgb: square RGB uint8 (any size; the network sees 256x256)"""
        torch, w = self.torch, self.w
        with torch.no_grad():
            I_s = w.prepare_source(crop_rgb)
            info = w.get_kp_info(I_s)
            self.info = info
            self.f_s = w.extract_feature_3d(I_s)
            self.x_s = w.transform_keypoint(info)
            self.R_s = self.rot(info["pitch"], info["yaw"], info["roll"])
        out = self.lmk_runner.run(crop_rgb)
        self.lmk = out["pts"] if isinstance(out, dict) else out
        return self

    def render(self, *, lip=None, eye=None, smile=0.0, eyebrow=0.0, pout=0.0, purse=0.0, grin=0.0, open_=0.0,
               gaze_x=0.0, gaze_y=0.0, yaw=0.0, pitch=0.0, roll=0.0, stitch=True) -> np.ndarray:
        """one 512x512 RGB frame; the expression deltas are the official image-editing ones (src/gradio_pipeline.py)"""
        torch, w, info = self.torch, self.w, self.info
        with torch.no_grad():
            d = info["exp"].clone()
            if gaze_x or gaze_y:
                if gaze_x > 0:
                    d[0, 11, 0] += gaze_x * 0.0007; d[0, 15, 0] += gaze_x * 0.001
                else:
                    d[0, 11, 0] += gaze_x * 0.001; d[0, 15, 0] += gaze_x * 0.0007
                d[0, 11, 1] += gaze_y * -0.001; d[0, 15, 1] += gaze_y * -0.001
                b = -gaze_y / 2.0
                d[0, 11, 1] += b * -0.001; d[0, 13, 1] += b * 0.0003; d[0, 15, 1] += b * -0.001; d[0, 16, 1] += b * 0.0003
            if smile:
                d[0, 20, 1] += smile * -0.01; d[0, 14, 1] += smile * -0.02; d[0, 17, 1] += smile * 0.0065
                d[0, 17, 2] += smile * 0.003; d[0, 13, 1] += smile * -0.00275; d[0, 16, 1] += smile * -0.00275
                d[0, 3, 1] += smile * -0.0035; d[0, 7, 1] += smile * -0.0035
            if eyebrow:
                if eyebrow > 0:
                    d[0, 1, 1] += eyebrow * 0.001; d[0, 2, 1] += eyebrow * -0.001
                else:
                    d[0, 1, 0] += eyebrow * -0.001; d[0, 2, 0] += eyebrow * 0.001
                    d[0, 1, 1] += eyebrow * 0.0003; d[0, 2, 1] += eyebrow * -0.0003
            if pout:
                d[0, 19, 0] += pout
            if purse:
                d[0, 14, 1] += purse * 0.001; d[0, 3, 1] += purse * -0.0005; d[0, 7, 1] += purse * -0.0005; d[0, 17, 2] += purse * -0.0005
            if grin:
                d[0, 20, 2] += grin * -0.001; d[0, 20, 1] += grin * -0.001; d[0, 14, 1] += grin * -0.001
            if open_:
                d[0, 19, 1] += open_ * 0.001; d[0, 19, 2] += open_ * 0.0001; d[0, 17, 1] += open_ * -0.0001
            R_d = self.rot(info["pitch"] + pitch, info["yaw"] + yaw, info["roll"] + roll)
            R_new = (R_d @ self.R_s.permute(0, 2, 1)) @ self.R_s
            x_d = info["scale"] * (info["kp"] @ R_new + d) + info["t"]
            if eye is not None:
                c_s = np.array([[float(np.linalg.norm(self.lmk[6] - self.lmk[18]) / (np.linalg.norm(self.lmk[0] - self.lmk[12]) + 1e-6)),
                                 float(np.linalg.norm(self.lmk[30] - self.lmk[42]) / (np.linalg.norm(self.lmk[24] - self.lmk[36]) + 1e-6))]], np.float32)
                comb = torch.cat([torch.from_numpy(c_s), torch.tensor([[float(eye)]])], dim=1)
                x_d = x_d + w.retarget_eye(self.x_s, comb)
            if lip is not None:
                c_s = np.array([[float(np.linalg.norm(self.lmk[90] - self.lmk[102]) / (np.linalg.norm(self.lmk[48] - self.lmk[66]) + 1e-6))]], np.float32)
                comb = torch.cat([torch.from_numpy(c_s), torch.tensor([[float(lip)]])], dim=1)
                x_d = x_d + w.retarget_lip(self.x_s, comb)
            if stitch:
                x_d = w.stitching(self.x_s, x_d)
            out = w.warp_decode(self.f_s, self.x_s, x_d)
            return w.parse_output(out["out"])[0]

    def source_ratios(self):
        L = self.lmk
        eye = [float(np.linalg.norm(L[6] - L[18]) / np.linalg.norm(L[0] - L[12])), float(np.linalg.norm(L[30] - L[42]) / np.linalg.norm(L[24] - L[36]))]
        lip = float(np.linalg.norm(L[90] - L[102]) / np.linalg.norm(L[48] - L[66]))
        return eye, lip


def face_crop_box(face_json: Path, size_k: float = 1.95) -> list[int]:
    """square crop around the face from our own landmarks: centred between the eyes and the chin, side = size_k x the
    eye-to-chin height (LivePortrait expects the whole face with some forehead and chin margin)"""
    p = json.loads(face_json.read_text(encoding="utf-8"))["points"]
    eye_y = 0.5 * (p["eye_l_top"][1] + p["eye_r_top"][1])
    cx = 0.5 * (p["eye_l_outer"][0] + p["eye_r_outer"][0])
    h = p["chin"][1] - eye_y
    side = int(round(size_k * h * 1.25))
    cy = 0.5 * (eye_y + p["chin"][1]) - 0.06 * side
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    return [x0, y0, x0 + side, y0 + side]


def load_rgb(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3]); im = bg
    return np.asarray(im.convert("RGB"))
