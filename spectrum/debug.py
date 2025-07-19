# utils/debug.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
