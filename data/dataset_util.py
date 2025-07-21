import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from nptdms import TdmsFile
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, footprint_rectangle
from scipy import ndimage as ndi

# ------------- Helper to identify wavelength axis -----------------
def _is_lambda_channel(ch):
    # 1) float 배열이고 단조증가(monotonic)하면 λ-axis
    try:
        arr = ch[:]  # numpy array
        if isinstance(arr, np.ndarray) and arr.dtype.kind == 'f' \
           and arr.ndim == 1 and arr.size > 1 \
           and np.all(np.diff(arr) > 0):
            return True
    except:
        pass

    # 2) (기존 로직) 이름/단위 기반 식별
    name = ch.name.lower()
    if any(key in name for key in ("wave", "lambda", "wl", "wavelength", "nm")):
        return True
    unit = str(ch.properties.get("unit_string")
               or ch.properties.get("Unit") or "").lower()
    return "nm" in unit

# -------------------- TDMS -> cube --------------------------------
def tdms_to_cube(path: Path,
                 image_shape: Optional[Tuple[int, int]] = None):
    """
    MATLAB‑버전과 동일한 로직으로 TDMS → (H,W,L) cube 변환
    """
    td = TdmsFile.read(path)
    ch_all = [ch for g in td.groups() for ch in g.channels()]

    # ── 1) λ‑축 채널 식별 ──────────────────────────────
    def _is_lambda(ch):
        v = ch[:]
        if v.ndim == 1 and v.size > 1 and np.all(np.diff(v) > 0):
            return True
        name = ch.name.lower()
        if any(k in name for k in ("wave", "lambda", "wl", "wavelength", "nm")):
            return True
        unit = str(ch.properties.get("unit_string")
                   or ch.properties.get("Unit") or "").lower()
        return "nm" in unit

    wl_ch   = next((c for c in ch_all if _is_lambda(c)), None)
    if wl_ch is None:
        raise RuntimeError("λ‑axis channel not found")
    wl      = wl_ch[:].astype(np.float32)

    # ── 2) 스펙트럼 채널 목록 (λ‑축과 길이가 같은 채널) ──
    specs   = [c for c in ch_all if len(c) == len(wl) and c is not wl_ch]
    Nspec   = len(specs)
    if Nspec == 0:
        raise RuntimeError("No spectrum channels found")

    # ── 3) (행, 열) 추론 — MATLAB 방식 우선 적용 ──────
    rows = cols = None

    if image_shape is not None:
        rows, cols = image_shape

    else:
        # 3‑A) 각 채널 Property 8,9,10 → pcol, startRow, endRow
        prop8 = specs[0].properties.get(8)   # 모든 채널에 동일
        prop9 = specs[0].properties.get(9)
        prop10 = specs[0].properties.get(10)

        if prop8 and prop9 is not None and prop10 is not None:
            cols = int(prop8)
            rows = int(prop10 - prop9 + 1)

        # 3‑B) NI_ArrayRow / NI_ArrayColumn 메타
        if rows is None or cols is None:
            row_list = [c.properties.get("NI_ArrayRow") for c in specs
                        if "NI_ArrayRow" in c.properties]
            col_list = [c.properties.get("NI_ArrayColumn") for c in specs
                        if "NI_ArrayColumn" in c.properties]
            if row_list and col_list:
                rows = max(row_list) + 1
                cols = max(col_list) + 1

        # 3‑C) root 'strips', 'top pixel', 'bottom pixel'
        if rows is None or cols is None:
            strips = td.properties.get("strips")
            top    = td.properties.get("top pixel")
            bot    = td.properties.get("bottom pixel")
            if strips and top is not None and bot is not None:
                cols = int(bot - top + 1)
                rows = int(strips)

        # 3‑D) perfect‑square fallback
        if rows is None or cols is None:
            if int(np.sqrt(Nspec)) ** 2 == Nspec:
                rows = cols = int(np.sqrt(Nspec))

        # 3‑E) 마지막 보루: root Image_Height / Width
        if rows is None or cols is None:
            rows = td.properties.get("Image_Height")
            cols = td.properties.get("Image_Width")

        if rows is None or cols is None or rows * cols != Nspec:
            raise RuntimeError("Cannot infer image shape")

    # ── 4) MATLAB 과 동일한 채널 → 픽셀 매핑 ────────────
    #   MATLAB: for c = 1:cols, for r = 1:rows
    #   → column 우선, row 2nd → (r 변수가 더 빨리 변함)
    #   채널의 이름(또는 NI_ArrayColumn,Row)이 해당 순서라고 가정
    def chan_key(ch):
        # NI_ArrayColumn/Row 우선, 없으면 이름
        col = ch.properties.get("NI_ArrayColumn")
        row = ch.properties.get("NI_ArrayRow")
        return (col if col is not None else 0,
                row if row is not None else 0,
                ch.name)

    specs.sort(key=chan_key)

    # 벡터 → cube 변환 (C‑order → MATLAB column‑major 맞추기)
    stack = np.vstack([ch[:] for ch in specs])  # (Nspec, L)
    cube  = (stack.reshape(cols, rows, len(wl), order='C').transpose(1, 0, 2))                      # (rows, cols, L)

    return cube.astype(np.float32), wl.astype(np.float32)

# ------------------- Remaining helpers ----------------------------
def flatfield_correct(cube: np.ndarray,
                      wvl: np.ndarray,
                      white_path: Path,
                      dark_path: Path) -> np.ndarray:
    """
    cube:  (H, W, L) — 이미 preprocess()로 crop된 데이터
    wvl:   (L,)       — preprocess() 후 self.wvl
    white_path, dark_path: raw TDMS 파일 경로
    """
    # 1) white/dark 모두 load 후,
    #    sample과 똑같이 파장 축 crop
    w_cube, w_wvl = tdms_to_cube(white_path)
    d_cube, d_wvl = tdms_to_cube(dark_path)

    # wavelength matching
    idxs = [int(np.argmin(np.abs(w_wvl - v))) for v in wvl]
    w_crop = w_cube[:, :, idxs]    # → shape (H_w, W_w, L)
    d_crop = d_cube[:, :, idxs]

    # 2) 공간 방향 평균해서 1D 참조 스펙트럼으로
    w_ref = w_crop.mean(axis=(0, 1))   # shape (L,)
    d_ref = d_crop.mean(axis=(0, 1))   # shape (L,)

    # 3) broadcast 보정
    num = cube - d_ref[None, None, :]
    den = np.clip((w_ref - d_ref)[None, None, :], 1e-9, None)
    return num / den

def crop_and_bg(cube, wavelengths, args):

    m = (wavelengths >= args["CROP_RANGE_NM"][0]) & (wavelengths <= args["CROP_RANGE_NM"][1])
    cube = cube[:, :, m]
    wavelengths = wavelengths[m]

    if args["BACKGROUND_PERC"] > 0:

        bg = np.quantile(cube, args["BACKGROUND_PERC"], axis=2, keepdims=True)
        cube = np.maximum(cube - bg, 0)

    return cube.astype(np.float32), wavelengths

def cube_to_rgb(cube, wavelengths):

    def idx(wl): return int(np.abs(wavelengths - wl).argmin())

    r,g,b = cube[..., idx(650)], cube[..., idx(550)], cube[..., idx(450)]

    return np.clip(np.stack([r,g,b], axis=-1) / np.percentile(cube, 99), 0,1)

def label_particles(rgb, args):

    gray = rgb.mean(axis=2)
    thr = threshold_otsu(gray)
    mask = gray > thr * args["THRESH_HIGH"]
    mask = binary_closing(mask, footprint_rectangle((3, 3)))
    mask = remove_small_objects(mask, args["MIN_PIXELS_CLUS"])
    labels, num = ndi.label(mask)
    cents = np.array(ndi.center_of_mass(mask, labels, range(1, num+1)))

    return labels, cents
