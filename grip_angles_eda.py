import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Display options for pandas output in terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

ANGLES_DIR = os.path.join("output_data", "angles")
OUT_DIR = os.path.join("output_data", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_KEYS = ["thumb_ip_234", "index_pip_567", "index_dip_678", "v_angle_502"]

def load_angle_jsons(angles_dir: str):
    rows = []
    files = sorted(glob.glob(os.path.join(angles_dir, "*_angles.json")))
    for fp in files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        # 원본 파일명(angles json은 "{원본파일명}_angles.json" 형태)
        base = os.path.basename(fp)
        # 예: "forehand_1.webp_angles.json" -> "forehand_1.webp"
        original = base.replace("_angles.json", "")

        row = {"file": original}
        for k in TARGET_KEYS:
            v = data.get(k, None)
            row[k] = float(v) if v is not None else np.nan
        rows.append(row)

    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame):
    summary_rows = []
    for k in TARGET_KEYS:
        vals = df[k].dropna().values.astype(float)
        if len(vals) == 0:
            summary_rows.append({
                "angle": k, "n": 0, "mean": np.nan, "std": np.nan,
                "min": np.nan, "p5": np.nan, "p25": np.nan, "median": np.nan,
                "p75": np.nan, "p95": np.nan, "max": np.nan
            })
            continue

        summary_rows.append({
            "angle": k,
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,  # 표본표준편차
            "min": float(np.min(vals)),
            "p5": float(np.percentile(vals, 5)),
            "p25": float(np.percentile(vals, 25)),
            "median": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "p95": float(np.percentile(vals, 95)),
            "max": float(np.max(vals)),
        })

    return pd.DataFrame(summary_rows)

def plot_histograms(df: pd.DataFrame):
    n = len(TARGET_KEYS)
    if n == 0:
        return
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, k in zip(axes, TARGET_KEYS):
        vals = df[k].dropna().values.astype(float)
        if len(vals) == 0:
            ax.set_title(f"{k} (n=0)")
            continue
        ax.hist(vals, bins=20)
        ax.set_title(f"{k} (n={len(vals)})")
        ax.set_xlabel("degrees")
        ax.set_ylabel("count")

    for ax in axes[len(TARGET_KEYS):]:
        ax.axis("off")

    fig.suptitle("Angle Distributions", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_boxplots(df: pd.DataFrame):
    data = [df[k].dropna().values.astype(float) for k in TARGET_KEYS]
    plt.figure(figsize=(max(6, 1.5 * len(TARGET_KEYS)), 5))
    plt.boxplot(data, labels=TARGET_KEYS, vert=True)
    plt.title("Angle Boxplots")
    plt.ylabel("degrees")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

def find_outliers_iqr(df: pd.DataFrame):
    """
    IQR 기반 이상치(전형적인 EDA 방식) 이미지 목록을 뽑습니다.
    """
    outlier_rows = []
    for k in TARGET_KEYS:
        vals = df[k].dropna()
        if len(vals) < 4:
            continue

        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr

        mask = (df[k] < lo) | (df[k] > hi)
        sub = df.loc[mask, ["file", k]].copy()
        sub["angle"] = k
        sub["iqr_lo"] = lo
        sub["iqr_hi"] = hi
        outlier_rows.append(sub)

    if len(outlier_rows) == 0:
        return pd.DataFrame(columns=["file", "value", "angle", "iqr_lo", "iqr_hi"])

    out_df = pd.concat(outlier_rows, ignore_index=True)
    out_df = out_df.rename(columns={k: "value"}) if "value" not in out_df.columns else out_df
    # 위 rename이 각 key마다 다르므로 정리
    if "value" not in out_df.columns:
        # fallback 정리
        pass
    return out_df

def main():
    if not os.path.exists(ANGLES_DIR):
        print(f"❌ angles dir not found: {ANGLES_DIR}")
        print("먼저 grip_extract_vector.py를 실행해서 output_data/angles/*.json이 생성되게 해주세요.")
        return

    df = load_angle_jsons(ANGLES_DIR)
    if len(df) == 0:
        print("❌ No angle json files found.")
        return

    # ===== Terminal logging =====
    print("\n===== Loaded angle files =====")
    print(f"count: {len(df)}")

    # Per-image angles (one line per file)
    print("\n===== Per-image angles (degrees) =====")
    for _, r in df.iterrows():
        parts = [f"file={r['file']}"]
        for k in TARGET_KEYS:
            v = r.get(k, np.nan)
            parts.append(f"{k}={v:.2f}" if pd.notna(v) else f"{k}=NaN")
        print(" | ".join(parts))

    # 1) 원본 데이터 CSV
    df.to_csv(os.path.join(OUT_DIR, "angles_raw.csv"), index=False)

    # 2) 요약 통계
    summary = summarize(df)
    summary.to_csv(os.path.join(OUT_DIR, "angles_summary.csv"), index=False)

    print("\n===== Summary statistics (degrees) =====")
    # Print a compact table to terminal
    print(summary.to_string(index=False))

    # Also print quick min/mean/max per key (single-line)
    print("\n===== Quick min/mean/max =====")
    for k in TARGET_KEYS:
        vals = df[k].dropna().values.astype(float)
        if len(vals) == 0:
            print(f"{k}: n=0")
            continue
        print(f"{k}: n={len(vals)} | min={np.min(vals):.2f} | mean={np.mean(vals):.2f} | max={np.max(vals):.2f}")

    # 3) 히스토그램/박스플롯 화면 표시
    plot_histograms(df)
    plot_boxplots(df)

    # ===== Correlation heatmap (screen only) =====
    angle_df = df[TARGET_KEYS].dropna()
    if len(angle_df) >= 2:
        corr = angle_df.corr()
        plt.figure(figsize=(6, 5))
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="correlation")
        plt.xticks(range(len(TARGET_KEYS)), TARGET_KEYS, rotation=45, ha="right")
        plt.yticks(range(len(TARGET_KEYS)), TARGET_KEYS)
        plt.title("Angle Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    # 4) IQR 이상치 목록
    # (간단히 key별로 이상치 뽑아 CSV로 저장)
    outlier_rows = []
    for k in TARGET_KEYS:
        vals = df[k].dropna()
        if len(vals) < 4:
            continue
        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        m = (df[k] < lo) | (df[k] > hi)
        sub = df.loc[m, ["file", k]].copy()
        sub = sub.rename(columns={k: "value"})
        sub["angle"] = k
        sub["iqr_lo"] = lo
        sub["iqr_hi"] = hi
        outlier_rows.append(sub)

    if outlier_rows:
        out_df = pd.concat(outlier_rows, ignore_index=True)
        out_df.to_csv(os.path.join(OUT_DIR, "angles_outliers_iqr.csv"), index=False)
    else:
        # 파일은 비워서라도 만들어두면 편함
        pd.DataFrame(columns=["file", "value", "angle", "iqr_lo", "iqr_hi"]).to_csv(
            os.path.join(OUT_DIR, "angles_outliers_iqr.csv"), index=False
        )

    print("✅ EDA outputs saved to:", OUT_DIR)
    print(" - angles_raw.csv")
    print(" - angles_summary.csv")
    print(" - hist_*.png / box_*.png")
    print(" - angles_outliers_iqr.csv")

if __name__ == "__main__":
    main()
def plot_gt_vs_nongt_histograms(df: pd.DataFrame, gt_mask: pd.Series, LABEL_GT="GT", LABEL_NGT="Non-GT"):
    """
    Overlaid histograms and smooth trend lines for GT vs Non-GT for each angle key.
    gt_mask: boolean Series, True for GT, False for Non-GT.
    """
    n = len(TARGET_KEYS)
    if n == 0:
        return
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, k in zip(axes, TARGET_KEYS):
        gt_vals = df.loc[gt_mask, k].dropna().values.astype(float)
        ngt_vals = df.loc[~gt_mask, k].dropna().values.astype(float)
        # Overlaid histograms
        if len(ngt_vals) > 0:
            ax.hist(ngt_vals, bins=20, alpha=0.35, density=True, label=LABEL_NGT)
            # Trend line (KDE)
            if len(ngt_vals) >= 5:
                kde_ngt = gaussian_kde(ngt_vals)
                xs = np.linspace(np.min(ngt_vals), np.max(ngt_vals), 200)
                ax.plot(xs, kde_ngt(xs), linewidth=2)

        if len(gt_vals) > 0:
            ax.hist(gt_vals, bins=20, alpha=0.35, density=True, label=LABEL_GT)
            # Trend line (KDE)
            if len(gt_vals) >= 5:
                kde_gt = gaussian_kde(gt_vals)
                xs = np.linspace(np.min(gt_vals), np.max(gt_vals), 200)
                ax.plot(xs, kde_gt(xs), linewidth=2)

        ax.set_title(f"{k} (GT n={len(gt_vals)}, Non-GT n={len(ngt_vals)})")
        ax.set_xlabel("degrees")
        ax.set_ylabel("density")
        ax.legend()
    for ax in axes[len(TARGET_KEYS):]:
        ax.axis("off")
    fig.suptitle("GT vs Non-GT Angle Distributions", fontsize=14)
    plt.tight_layout()
    plt.show()