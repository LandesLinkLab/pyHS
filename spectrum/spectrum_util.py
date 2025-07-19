import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def get_peak(spec, wavelengths):

    idx = int(np.argmax(spec))

    return float(wavelengths[idx]), float(spec[idx])

def pick_representatives(cube, labels, wavelengths, args):

    reps = []

    for lab in np.unique(labels):

        if lab == 0: continue

        coords = np.argwhere(labels == lab)

        if coords.shape[0] < args["MIN_PIXELS_CLUS"]: continue

        peak_pos = np.array([get_peak(cube[r, c], wavelengths)[0] for r, c in coords])

        if peak_pos.std() > args["PEAK_TOL_NM"]: continue

        ints = np.array([get_peak(cube[r, c], wavelengths)[1] for r, c in coords])
        sel = ints.argmax() if args["REP_CRITERION"] == "max_int" else 0
        r_sel, c_sel = map(int, coords[sel])
        reps.append(dict(row=r_sel, col=c_sel, wl_peak=float(peak_pos[sel]), intensity=float(ints[sel])))

    return reps

def fit_lorentz(y, x, args):
    
    def lorentz(x, A, x0, gamma):

        return (2 * A / np.pi) * (gamma / (4 * (x - x0)**2 + gamma**2))

    idx = int(np.argmax(y)); 
    peak = float(y[idx])
    gamma0 = 20.0
    A0 = (np.pi / 2) * peak * gamma0
    p0 = [A0, float(x[idx]), gamma0]

    try:
        bounds = ([0, x.min(),  1.0], [np.inf, x.max(), 200.0])
        popt, _ = curve_fit(lorentz, x, y, p0=p0, bounds=bounds, maxfev=5000, method='trf')
        # popt,_ = curve_fit(lorentz, x, y, p0=p0, maxfev=10000)
        y_fit = lorentz(x, *popt)
        rsq = 1 - np.sum((y-y_fit)**2)/np.sum((y-y.mean())**2)
        A_opt, x0_opt, g_opt = popt
        a_peak = 2 * A_opt / (np.pi * g_opt)
    
        return y_fit, dict(a=a_peak, A=A_opt, x0=x0_opt, gamma=g_opt), float(rsq)
    
    except Exception:
    
        return np.zeros_like(y), {}, 0.0

def plot_spectrum(x, y, y_fit, title, out_png, dpi=300, params=None, snr=None):
    fig, ax = plt.subplots()
    ax.plot(x, y, label="raw")
    ax.plot(x, y_fit, "--", label="fit")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(title)
    ax.legend()

    if params is not None and snr is not None:
        txt = (
            rf"$\lambda_{{\max}}$ = {params['x0']:.0f} nm" "\n"
            rf"$\gamma$ = {params['gamma']:.0f} nm" "\n"
            rf"S/N = {snr:.0f}"
        )
        ax.text(
            0.95, 0.95, txt,
            transform=ax.transAxes,
            va="top", ha="right",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def save_markers(cube, reps, out_png, dpi=300):
    """
    cube:  (H, W, L) 3D 스펙트럼 큐브
    reps:  [{'row':…, 'col':…}, …]
    """
    # 1) max‐intensity projection
    proj = cube.max(axis=2)      # (H, W)

    vmin, vmax = np.percentile(proj, [2, 98])

    fig, ax = plt.subplots(figsize=(6,6))
    # 전체 dynamic range, 반전(gray_r) 컬러맵
    ax.imshow(proj,
              cmap='gray',
              origin='lower',
              vmin = vmin,
              vmax = vmax,)
    ax.set_axis_off()

    # 2) 초록색 원(circle)과 숫자 표시
    for i, r in enumerate(reps):
        col = r['col']
        row = r['row']
        
        circ = plt.Circle((col, row),
                          radius=3,
                          edgecolor='green',
                          facecolor='none',
                          linewidth=1)
        ax.add_patch(circ)
        ax.text(col + 4,
                row - 4,
                str(i),
                color='green',
                fontsize=8,
                weight='bold')

    # 크롭 옵션 없이 전체 맵 + 마커를 저장
    fig.savefig(out_png, dpi=dpi)
    # print("proj:", proj.min(), proj.max(), "shape:", proj.shape)
    plt.close(fig)

def dump_mapping_diagnostic(cube, reps, out_png, grid_step=50, dpi=200):
    """
    cube : ndarray (H, W, L) – 전처리 완료 큐브 사용 권장
    reps : [{'row':…, 'col':…}, …] – 대표 픽셀 리스트 (없어도 됨)
    out_png : 저장 경로
    grid_step : 행/열 그리드 간격(pixel)
    """
    proj = cube.max(axis=2)
    vmin, vmax = np.percentile(proj, [2, 98])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(proj, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)

    # 1) 격자선으로 행/열 뒤집힘 확인
    for r in range(0, cube.shape[0], grid_step):
        ax.axhline(r, color='red', lw=0.3, alpha=0.4)
    for c in range(0, cube.shape[1], grid_step):
        ax.axvline(c, color='red', lw=0.3, alpha=0.4)

    # 2) (선택) 대표 픽셀 마커도 함께
    for i, r in enumerate(reps):
        ax.plot(r['col'], r['row'], 'go', ms=4)
        ax.text(r['col'] + 3, r['row'] - 3, str(i),
                color='lime', fontsize=6, weight='bold')

    ax.set_axis_off()
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # 콘솔용 간단 통계
    peaks = proj[[p['row'] for p in reps], [p['col'] for p in reps]] if reps else []
    if len(peaks):
        print(f"[diag] Max‑intensities at reps — min/med/max : "
              f"{np.min(peaks):.3f} / {np.median(peaks):.3f} / {np.max(peaks):.3f}")    
