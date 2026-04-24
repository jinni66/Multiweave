from src.hurst.hurst import compute_Hc
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

raw_files = sorted(glob.glob("raw.csv"))
syn_files = sorted(glob.glob("syn.csv"))

results = []

for raw_file, syn_file in zip(raw_files, syn_files):
    df_raw = pd.read_csv(raw_file)
    df_syn = pd.read_csv(syn_file)

    pkt_raw = df_raw["pkt"].dropna().values
    byt_raw = df_raw["byt"].dropna().values
    pkt_syn = df_syn["pkt"].dropna().values
    byt_syn = df_syn["byt"].dropna().values


    H_pkt_raw, c_pkt_raw, data_pkt_raw = compute_Hc(pkt_raw, kind="change", simplified=False)
    H_byt_raw, c_byt_raw, data_byt_raw = compute_Hc(byt_raw, kind="change", simplified=False)
    H_pkt_syn, c_pkt_syn, data_pkt_syn = compute_Hc(pkt_syn, kind="change", simplified=False)
    H_byt_syn, c_byt_syn, data_byt_syn = compute_Hc(byt_syn, kind="change", simplified=False)

    results.append({
        "raw_file": os.path.basename(raw_file),
        "syn_file": os.path.basename(syn_file),
        "H_pkt_raw": H_pkt_raw,
        "H_pkt_syn": H_pkt_syn,
        "H_byt_raw": H_byt_raw,
        "H_byt_syn": H_byt_syn
    })

    f, axs = plt.subplots(2, 2, figsize=(10,8))
    axs = axs.flatten()

    axs[0].plot(data_pkt_raw[0], c_pkt_raw*data_pkt_raw[0]**H_pkt_raw, color="deepskyblue")
    axs[0].scatter(data_pkt_raw[0], data_pkt_raw[1], color="purple")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_title(f"{os.path.basename(raw_file)} pkt_raw")
    axs[0].grid(True)

    axs[1].plot(data_byt_raw[0], c_byt_raw*data_byt_raw[0]**H_byt_raw, color="deepskyblue")
    axs[1].scatter(data_byt_raw[0], data_byt_raw[1], color="purple")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_title(f"{os.path.basename(raw_file)} byt_raw")
    axs[1].grid(True)

    axs[2].plot(data_pkt_syn[0], c_pkt_syn*data_pkt_syn[0]**H_pkt_syn, color="deepskyblue")
    axs[2].scatter(data_pkt_syn[0], data_pkt_syn[1], color="purple")
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')
    axs[2].set_title(f"{os.path.basename(syn_file)} pkt_syn")
    axs[2].grid(True)

    axs[3].plot(data_byt_syn[0], c_byt_syn*data_byt_syn[0]**H_byt_syn, color="deepskyblue")
    axs[3].scatter(data_byt_syn[0], data_byt_syn[1], color="purple")
    axs[3].set_xscale('log')
    axs[3].set_yscale('log')
    axs[3].set_title(f"{os.path.basename(syn_file)} byt_syn")
    axs[3].grid(True)

    folder_name = os.path.basename(os.path.dirname(raw_file))
    plot_file = os.path.join("hurst_plots", f"{folder_name}_hurst.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

df_results = pd.DataFrame(results)
print(df_results)
output_file = "hurst_summary.csv"
if os.path.exists(output_file):
    df_results.to_csv(output_file, mode='a', index=False, header=False)
else:
    df_results.to_csv(output_file, index=False)



















# from hurst import compute_Hc
# import pandas as pd
# import glob
# import os
#
# # 原始和合成 CSV 文件路径
# raw_files = sorted(glob.glob("./NetShare-Dataset/cidds/raw.csv"))  # 包含 raw.csv 和 raw_agg_*.csv
# syn_files = sorted(glob.glob("./NetShare-Dataset/cidds/syn.csv"))  # 包含 syn.csv 和 syn_agg_*.csv
#
# # 检查文件数量一致
# assert len(raw_files) == len(syn_files), "原始和合成文件数量不一致！"
#
# results = []
#
# for raw_file, syn_file in zip(raw_files, syn_files):
#     # 读取 CSV
#     df_raw = pd.read_csv(raw_file)
#     df_syn = pd.read_csv(syn_file)
#
#     pkt_raw = df_raw["pkt"].values
#     byt_raw = df_raw["byt"].values
#     pkt_syn = df_syn["pkt"].values
#     byt_syn = df_syn["byt"].values
#
#     # 跳过长度不足100的序列
#     if len(pkt_raw) < 100 or len(byt_raw) < 100 or len(pkt_syn) < 100 or len(byt_syn) < 100:
#         print(f"跳过文件 {raw_file} / {syn_file}，序列长度不足100")
#         continue
#
#     # 计算 Hurst
#     H_pkt_raw, _, _ = compute_Hc(pkt_raw, kind="change", simplified=False)
#     H_byt_raw, _, _ = compute_Hc(byt_raw, kind="change", simplified=False)
#     H_pkt_syn, _, _ = compute_Hc(pkt_syn, kind="change", simplified=False)
#     H_byt_syn, _, _ = compute_Hc(byt_syn, kind="change", simplified=False)
#
#     # 保存结果
#     results.append({
#         "raw_file": os.path.basename(raw_file),
#         "syn_file": os.path.basename(syn_file),
#         "H_pkt_raw": H_pkt_raw,
#         "H_pkt_syn": H_pkt_syn,
#         "H_byt_raw": H_byt_raw,
#         "H_byt_syn": H_byt_syn
#     })
#
# # 转为 DataFrame
# df_results = pd.DataFrame(results)
#
# # 如果文件已存在，则追加；否则新建
# output_file = "hurst_summary.csv"
# if os.path.exists(output_file):
#     df_results.to_csv(output_file, mode='a', index=False, header=False)
# else:
#     df_results.to_csv(output_file, index=False)
#
# print(f"Hurst 汇总结果已保存到 {output_file}")
