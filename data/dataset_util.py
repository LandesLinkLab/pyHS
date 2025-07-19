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
def tdms_to_cube(path: Path, image_shape: Optional[Tuple[int,int]] = None):
    """
    Load a hyperspectral TDMS file into a (H, W, L) cube + wavelength vector.
    
    Parameters
    ----------
    path : Path
        Path to the .tdms file.
    image_shape : Tuple[int,int], optional
        If provided, force (rows, cols) = image_shape regardless of metadata.
    
    Returns
    -------
    cube : np.ndarray, shape (H, W, L)
    wl   : np.ndarray, shape (L,)
    """
    td = TdmsFile.read(path)
    # 모든 채널(flat list) 수집
    channels = [ch for g in td.groups() for ch in g.channels()]

    # λ-axis 채널 찾기
    def _is_lambda_channel(ch):
        try:
            arr = ch[:]
            if isinstance(arr, np.ndarray) and arr.dtype.kind == 'f' \
               and arr.ndim == 1 and arr.size > 1 \
               and np.all(np.diff(arr) > 0):
                return True
        except:
            pass
        name = ch.name.lower()
        if any(k in name for k in ("wave","lambda","wl","wavelength","nm")):
            return True
        unit = str(ch.properties.get("unit_string") or ch.properties.get("Unit") or "").lower()
        return "nm" in unit

    wl_chs = [ch for ch in channels if _is_lambda_channel(ch)]
    if not wl_chs:
        raise RuntimeError("λ-axis channel not found.")
    wl = wl_chs[0][:]

    # 스펙트럼 채널들만 골라
    specs = [ch for ch in channels if len(ch) == len(wl) and ch not in wl_chs]
    if not specs:
        raise RuntimeError("No spectrum channels found.")
    N = len(specs)

    # 1) config로 강제 shape
    rows = cols = None
    if image_shape is not None:
        rows, cols = image_shape
    else:
        # 2) NI_ArrayRow / NI_ArrayColumn 메타
        row_props = [ch.properties.get("NI_ArrayRow")    for ch in specs if "NI_ArrayRow"    in ch.properties]
        col_props = [ch.properties.get("NI_ArrayColumn") for ch in specs if "NI_ArrayColumn" in ch.properties]

        if row_props and col_props:

            n_rows = max(row_props) + 1
            n_cols = max(col_props) + 1

            if n_rows * n_cols == N:

                rows, cols = n_rows, n_cols

                print('[info] NI_ArrayRow and NI_ArrayColumn used for image shape')

            else: RuntimeError("[error] Shape of image mismatch")

        # 3) root metadata 'strips' / 'top pixel' / 'bottom pixel'
        if rows is None or cols is None:

            strips = td.properties.get("strips")
            top    = td.properties.get("top pixel")
            bot    = td.properties.get("bottom pixel")

            if strips is not None and top is not None and bot is not None:

                n_cols = int(bot - top + 1)
                n_rows = int(strips)

                if n_rows * n_cols == N:

                    rows, cols = n_rows, n_cols

                    print('[info] strips, top pixel, and bottom pixel used for image shape')

                else: RuntimeError("[error] Shape of image mismatch")

    # 4) fallback: perfect square
    if rows is None or cols is None:

        if int(np.sqrt(N))**2 == N:

            rows = cols = int(np.sqrt(N))
        
        else:
            # 5) TDMS root 속성 Image_Height/Image_Width
            rows = td.properties.get("Image_Height")
            cols = td.properties.get("Image_Width")

            print('[info] Image_Height and Image_Width used for image shape')

            if rows is None or cols is None:

                raise RuntimeError("[error] Cannot infer image size")

    # 이제 reshape
    specs.sort(key=lambda c: c.name)
    # cube = np.vstack([ch[:] for ch in specs]).reshape(rows, cols, len(wl))
    # cube = np.vstack([ch[:] for ch in specs]).reshape((rows, cols, len(wl)), order='F')
    cube = np.vstack([ch[:] for ch in specs]).reshape((cols, rows, len(wl))).transpose(1, 0, 2)

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
