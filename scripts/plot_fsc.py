import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def plot_fsc(file_path, save_path=None, figsize=(10, 7)):
    """绘制单条FSC曲线，自动标注0.143和0.5对应的分辨率"""
    data = np.loadtxt(file_path, comments="#")
    freqs = data[:, 0]
    fsc = np.clip(data[:, 1], 0, 1)

    valid = (freqs > 0) & np.isfinite(freqs) & np.isfinite(fsc)
    freqs, fsc = freqs[valid], fsc[valid]

    interp = interp1d(
        freqs,
        fsc,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    freq_dense = np.linspace(freqs.min(), freqs.max(), 1000)
    fsc_dense = interp(freq_dense)

    plt.figure(figsize=figsize)
    plt.plot(
        freq_dense,
        fsc_dense,
        "b-",
        linewidth=2,
        label=os.path.basename(file_path),
    )
    plt.axhline(y=0.143, color="gold", linestyle="--", linewidth=1.5, label="FSC=0.143")
    plt.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, label="FSC=0.5")

    for threshold, color in [(0.143, "gold"), (0.5, "gray")]:
        cross_idx = np.where(
            (fsc_dense[:-1] > threshold) & (fsc_dense[1:] <= threshold)
        )[0]
        if len(cross_idx) > 0:
            i = cross_idx[0]
            f_cross = (
                freq_dense[i]
                + (threshold - fsc_dense[i])
                * (freq_dense[i + 1] - freq_dense[i])
                / (fsc_dense[i + 1] - fsc_dense[i])
            )
            plt.plot(f_cross, threshold, "o", color="black", markersize=5)
            plt.text(
                f_cross,
                threshold + 0.03,
                f"{1 / f_cross:.2f}Å",
                fontsize=9,
                ha="center",
            )

    plt.xlabel("Frequency (1/Å)", fontsize=14)
    plt.ylabel("Fourier Shell Correlation", fontsize=14)
    plt.xlim(0, freqs.max())
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        # 如果save_path是目录，则自动生成文件名
        if os.path.isdir(save_path):
            base = os.path.basename(file_path)
            save_path = os.path.join(save_path, os.path.splitext(base)[0] + ".png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="绘制FSC曲线并标注0.143和0.5对应的分辨率"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入的数据文件路径（支持 .txt 或 .dat）"
    )
    parser.add_argument(
        "--output", "-o",
        help="保存路径，可以是 .png 文件或目录；若不指定则直接显示图形"
    )
    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.isfile(args.input):
        print(f"错误：数据文件 '{args.input}' 不存在。")
        sys.exit(1)

    # 绘图
    try:
        result = plot_fsc(args.input, save_path=args.output)
        if result:
            print(f"✅ 图片已保存至: {result}")
        else:
            print("✅ 图形已显示（未保存）。")
    except Exception as e:
        print(f"❌ 绘图失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
  