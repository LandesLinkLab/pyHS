import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def get_peak(spec, wavelengths):
    idx = int(np.argmax(spec))

    return float(wavelengths[idx]), float(spec[idx])

def pick_representatives(cube, labels, wavelengths, cfg):
    reps = []
    for lab in np.unique(labels):

        if lab == 0: continue

        coords = np.argwhere(labels == lab)

        if coords.shape[0] < cfg.MIN_PIXELS_CLUS: continue

        peak_pos = np.array([get_peak(cube[r, c], wavelengths)[0] for r, c in coords])

        if peak_pos.std() > cfg.PEAK_TOL_NM: continue

        ints = np.array([get_peak(cube[r, c], wavelengths)[1] for r, c in coords])
        sel = ints.argmax() if cfg.REP_CRITERION == "max_int" else 0
        r_sel, c_sel = map(int, coords[sel])
        reps.append(dict(row=r_sel, col=c_sel, wl_peak=float(peak_pos[sel]), intensity=float(ints[sel])))

    return reps

def fit_lorentz(y, x, cfg):
    
    def lorentz(x, a, x0, g): 

        return a*g**2 / ((x-x0)**2 + g**2)

    idx = int(np.argmax(y)); p0 = [float(y[idx]), float(x[idx]), 20.0]
    
    try:
        popt,_ = curve_fit(lorentz, x, y, p0=p0, maxfev=10000)
        y_fit = lorentz(x, *popt)
        rsq = 1 - np.sum((y-y_fit)**2)/np.sum((y-y.mean())**2)
    
        return y_fit, dict(a=popt[0], x0=popt[1], gamma=popt[2]), float(rsq)
    
    except Exception:
    
        return np.zeros_like(y), {}, 0.0

def plot_spectrum(x, y, y_fit, title, out_png, dpi=300):
    
    fig, ax = plt.subplots()
    ax.plot(x, y, label="raw")
    ax.plot(x, y_fit, "--", label="fit")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(title); ax.legend()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)

def save_markers(rgb, reps, out_png):
    
    ys = [r["row"] for r in reps]; xs = [r["col"] for r in reps]
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.scatter(xs, ys, s=60, edgecolors="red", facecolors="none", linewidths=1.5)
    
    for i,(x,y) in enumerate(zip(xs,ys)):
    
        ax.text(x+4, y-4, str(i), color="yellow", fontsize=8, weight="bold")
    
    ax.set_axis_off()
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
