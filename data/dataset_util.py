from pathlib import Path
import numpy as np
from nptdms import TdmsFile
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, square
from scipy import ndimage as ndi

# ------------- Helper to identify wavelength axis -----------------
def _is_lambda_channel(ch):

    name = ch.name.lower()

    if any(key in name for key in ("wave", "lambda", "nm")):

        return True

    unit = str(ch.properties.get("unit_string") or ch.properties.get("Unit") or "").lower()

    return "nm" in unit

# -------------------- TDMS -> cube --------------------------------
def tdms_to_cube(path: Path, image_shape=None):

    td = TdmsFile.read(path)
    channels = [ch for g in td.groups() for ch in g.channels()]

    wl_channels = [ch for ch in channels if _is_lambda_channel(ch)]

    if not wl_channels:

        raise RuntimeError("λ-axis channel not found.")

    wl = wl_channels[0][:]

    specs = [ch for ch in channels if len(ch) == len(wl) and ch not in wl_channels]

    if not specs:

        raise RuntimeError("No spectrum channels found.")

    N = len(specs)

    if image_shape:

        rows, cols = image_shape

    elif int(np.sqrt(N))**2 == N:

        rows = cols = int(np.sqrt(N))

    else:

        rows = td.properties.get("Image_Height")
        cols = td.properties.get("Image_Width")

        if rows is None or cols is None:
            
            raise RuntimeError("Cannot infer image size; use --image_shape.")

    specs.sort(key=lambda c: c.name)
    cube = np.vstack([ch[:] for ch in specs]).reshape(rows, cols, len(wl))

    return cube.astype(np.float32), wl.astype(np.float32)

# ------------------- Remaining helpers ----------------------------
def flatfield_correct(cube, white_path: Path, dark_path: Path):

    w, _ = tdms_to_cube(white_path)
    d, _ = tdms_to_cube(dark_path)

    return (cube - d) / np.clip(w - d, 1e-9, None)

def crop_and_bg(cube, wavelengths, cfg):

    m = (wavelengths >= cfg.CROP_RANGE_NM[0]) & (wavelengths <= cfg.CROP_RANGE_NM[1])
    cube = cube[:, :, m]
    wavelengths = wavelengths[m]

    if cfg.BACKGROUND_PERC > 0:

        bg = np.quantile(cube, cfg.BACKGROUND_PERC, axis=2, keepdims=True)
        cube = np.maximum(cube - bg, 0)

    return cube.astype(np.float32), wavelengths

def cube_to_rgb(cube, wavelengths):

    def idx(wl): return int(np.abs(wavelengths - wl).argmin())

    r,g,b = cube[..., idx(650)], cube[..., idx(550)], cube[..., idx(450)]

    return np.clip(np.stack([r,g,b], axis=-1) / np.percentile(cube, 99), 0,1)

def label_particles(rgb, cfg):

    gray = rgb.mean(axis=2)
    thr = threshold_otsu(gray)
    mask = gray > thr * cfg.THRESH_HIGH
    mask = binary_closing(mask, square(3))
    mask = remove_small_objects(mask, cfg.MIN_PIXELS_CLUS)
    labels, num = ndi.label(mask)
    cents = np.array(ndi.center_of_mass(mask, labels, range(1, num+1)))

    return labels, cents
