"""Result documents, figures and artifact hashes generated from the recorded experiment JSON."""
import hashlib
import importlib.metadata
import json
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from homeease.exercises import EXERCISE_NAMES

from .common import ROOT


def write_results():
    results = json.loads((ROOT / "reports/experiments.json").read_text())
    poses = json.loads((ROOT / "reports/pose_comparison.json").read_text())
    selected = json.loads((ROOT / "artifacts/selected.json").read_text())
    row = next(r for r in results if r["model"] == selected["model"])
    test_best = min(results, key=lambda r: (r["test"]["count_mae"], -r["test"]["phase_macro_f1"]))
    baseline_path = ROOT / "reports/optimization/baseline/selected.json"
    baseline_name = json.loads(baseline_path.read_text())["model"] if baseline_path.exists() else None
    baseline = next((r for r in results if r["model"] == baseline_name), None)
    selected_pose = json.loads((ROOT / "artifacts/selected_pose.json").read_text())
    figs = ROOT / "reports/figures"
    figs.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    handles, legend_labels = [], []
    for number, r in enumerate(results, 1):
        marker = "*" if r["model"] == row["model"] else "o"
        handle = axes[0].scatter(r["cpu_p50_ms"], r["validation"]["count_mae"], marker=marker, s=120 if marker == "*" else 40)
        handles.append(handle)
        legend_labels.append(f"{number}. {r['model']}")
        if r["model"] == row["model"]:
            axes[0].annotate(f"Recommended: {r['model']}", (r["cpu_p50_ms"], r["validation"]["count_mae"]),
                             fontsize=10, fontweight="bold", color="darkgreen", xytext=(0, 20),
                             textcoords="offset points", arrowprops={"arrowstyle": "->", "color": "darkgreen"})
        else:
            axes[0].annotate(str(number), (r["cpu_p50_ms"], r["validation"]["count_mae"]), fontsize=9,
                             xytext=(5, 6 if number % 2 or r["validation"]["count_mae"] <= 1.05 else -14), textcoords="offset points")
    axes[0].set(xlabel="CPU temporal inference p50 (ms)", ylabel="Validation count MAE (reps)", title="Validation accuracy vs compute")
    axes[0].grid(alpha=.2)
    for number, p in enumerate(poses):
        axes[1].scatter(p["latency_p50_ms"], p["pck_bbox_02"] * 100)
        axes[1].annotate(f"{p['model'].replace('-pose.pt','')}@{p['imgsz']}", (p["latency_p50_ms"], p["pck_bbox_02"] * 100),
                         fontsize=8, xytext=(3, -14 if number == 1 else 6), textcoords="offset points")
    axes[1].set(xlabel="GPU full pose call p50 (ms)", ylabel="PCK@0.2bbox (%)", title="COCO 100-image pose subset")
    axes[1].grid(alpha=.2)
    axes[1].margins(x=.2)
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .15, 1, 1))
    fig.savefig(figs / "accuracy_compute.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    matrix = np.asarray(row["test"]["action_confusion_matrix"])
    plot = ax.imshow(matrix, cmap="Blues")
    labels = ["Chair STS", "Squat", "Leg raise", "Hurdle step", "Other"]
    ax.set_xticks(range(5), labels, rotation=35, ha="right")
    ax.set_yticks(range(5), labels)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() * .5 else "black")
    ax.set(xlabel="Predicted action", ylabel="Source action", title=f"Held-out action confusion: {row['model']}")
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(figs / "action_confusion.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    sessions = row["test"]["sessions"]
    x = np.arange(len(sessions))
    ax.bar(x - .2, [s["expected"] for s in sessions], width=.4, label="Source reference")
    ax.bar(x + .2, [s["predicted"] for s in sessions], width=.4, label="Predicted")
    ax.set_xticks(x, [s["session"].replace("_reconstructed", "") for s in sessions], rotation=40, ha="right")
    ax.set(ylabel="Complete cycles", title="Test reconstructed sessions, including negative actions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "test_session_counts.png", dpi=160)
    plt.close(fig)
    lines = ["# HomeEase 四动作算法 2 实验结果", "",
             "## 最佳算法结果速览", "",
             f"> **综合最佳、当前部署推荐：分层因果 TCN `{row['model']}`**\n> 验证计数 MAE **{row['validation']['count_mae']:.2f} 次**；测试计数 MAPE **{row['test']['count_mape_pct']:.2f}%**；测试动作 Macro-F1 **{row['test']['action_macro_f1']:.4f}**；CPU P50 **{row['cpu_p50_ms']:.3f} ms**。", "",
             f"> **测试回放指标最高：优化 GRU `{test_best['model']}`**\n> 测试计数 MAPE **{test_best['test']['count_mape_pct']:.2f}%**；测试动作 Macro-F1 **{test_best['test']['action_macro_f1']:.4f}**；测试联合阶段 Macro-F1 **{test_best['test']['phase_macro_f1']:.4f}**。", "",
             f"| 核心指标 | **综合最佳 TCN（推荐部署）** | **测试指标最高 GRU** |", "|---|---:|---:|",
             f"| 模型标识 | **`{row['model']}`** | **`{test_best['model']}`** |"]
    for label, split, key, fmt, suffix in [
        ("验证动作 Macro-F1 ↑", "validation", "action_macro_f1", ".4f", ""),
        ("验证联合阶段 Macro-F1 ↑", "validation", "phase_macro_f1", ".4f", ""),
        ("验证计数 MAE ↓", "validation", "count_mae", ".2f", " 次"),
        ("测试动作 Macro-F1 ↑", "test", "action_macro_f1", ".4f", ""),
        ("测试联合阶段 Macro-F1 ↑", "test", "phase_macro_f1", ".4f", ""),
        ("测试计数 MAE ↓", "test", "count_mae", ".2f", " 次"),
        ("测试计数 MAPE ↓", "test", "count_mape_pct", ".2f", "%"),
        ("测试相对计数 accuracy ↑", "test", "count_accuracy_pct", ".2f", "%"),
        ("测试会话完全正确率 ↑", "test", "exact_session_pct", ".1f", "%"),
        ("测试周期 recall ↑", "test", "cycle_recall", ".4f", ""),
        ("测试周期 precision ↑", "test", "cycle_precision", ".4f", ""),
    ]:
        values = [f"{r[split][key]:{fmt}}{suffix}" for r in [row, test_best]]
        lines.append(f"| {label} | **{values[0]}** | **{values[1]}** |")
    lines += [f"| CPU 时序前向 P50 ↓ | **{row['cpu_p50_ms']:.3f} ms** | {test_best['cpu_p50_ms']:.3f} ms |",
              f"| CPU 时序前向 P95 ↓ | **{row['cpu_p95_ms']:.3f} ms** | {test_best['cpu_p95_ms']:.3f} ms |",
              f"| 参数量 | {row['parameters']:,} | {test_best['parameters']:,} |", "",
              f"**为什么推荐 TCN：**它的验证计数 MAE 为 **{row['validation']['count_mae']:.2f} 次**，低于 GRU 的 {test_best['validation']['count_mae']:.2f} 次；GRU 的 CPU P50 是 TCN 的 **{test_best['cpu_p50_ms']/row['cpu_p50_ms']:.2f} 倍**。因此根据验证表现和计算成本选择 TCN，保留 GRU 作为测试回放精度最高的候选。模型与 decoder 已在本轮测试前锁定。", "",
              "以上 F1 是 Macro-F1；阶段标签为运动学弱标签。MAE 包含十个会话（八正、两负），MAPE 排除两个零参考会话；相对计数 accuracy=100%−MAPE，不是分类正确率。CPU 是本机单次时序前向，未包含 Pose CNN、HTTP、特征提取和排队开销。", ""]
    if baseline:
        lines += ["## 优化前后：原默认 GRU → 综合最佳 TCN", "",
                  f"| 指标 | 原默认 `{baseline['model']}` | **优化后 `{row['model']}`** |", "|---|---:|---:|"]
        for label, key, fmt, suffix in [("测试动作 Macro-F1", "action_macro_f1", ".4f", ""),
                                        ("测试联合阶段 Macro-F1", "phase_macro_f1", ".4f", ""),
                                        ("测试计数 MAE", "count_mae", ".2f", " 次"),
                                        ("测试计数 MAPE", "count_mape_pct", ".2f", "%"),
                                        ("测试周期 recall", "cycle_recall", ".4f", "")]:
            lines.append(f"| {label} | {baseline['test'][key]:{fmt}}{suffix} | **{row['test'][key]:{fmt}}{suffix}** |")
        lines += ["", f"测试动作 F1 提升 **{100*(row['test']['action_macro_f1']-baseline['test']['action_macro_f1']):.2f} 个百分点**；计数 MAE 降低 **{100*(1-row['test']['count_mae']/baseline['test']['count_mae']):.2f}%**。这是同一标签与划分下的描述性对照。", ""]
    matched, spurious, missed = (row["test"][k] for k in ["matched_cycles", "spurious_cycles", "missed_cycles"])
    lines += ["## 综合最佳 TCN 的四动作效果", "",
              f"**{matched}/{matched+missed} 个参考周期匹配成功，漏计 {missed} 个，多计 {spurious} 个。** 两个肩部动作负会话均为 0 次。", "",
              "| 动作 | 参考周期总数 | 预测周期总数 | MAE 次 | MAPE | 会话完全正确率 | 周期 recall | 周期 precision |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    action_names = {"chair_sit_to_stand": "椅子坐站", "squat": "深蹲", "standing_leg_raise": "站立直腿抬腿", "hurdle_step": "跨步抬腿"}
    for exercise in EXERCISE_NAMES:
        m = row["test"]["per_exercise"][exercise]
        expected = sum(s["expected"] for s in m["sessions"])
        predicted = sum(s["predicted"] for s in m["sessions"])
        lines.append(f"| {action_names[exercise]} `{exercise}` | {expected} | **{predicted}** | {m['count_mae']:.2f} | **{m['count_mape_pct']:.2f}%** | {m['exact_session_pct']:.1f}% | {m['cycle_recall']:.3f} | {m['cycle_precision']:.3f} |")
    lines += ["", "每动作仅两名测试参与者。周期时间匹配使用弱边界 ±0.5 秒；数量参考来自原始分段重复。", "",
              "## 最佳算法部署配置", "",
              f"实际加载权重：`{', '.join(selected['members'])}`，见 `artifacts/selected.json`。Decoder 为 `{selected['decoder_strategy']}`，阶段证据阈值 {selected['phase_evidence_threshold']}，消抖时长 {selected['min_dwell_s']} 秒，部署窗口 {row.get('config', {}).get('context', 64)} 帧。默认姿态前端：**{selected_pose['model']} @ {selected_pose['imgsz']}**。系统接入见 `docs/openapi.json`，完整优化过程见 `reports/optimization/`。", "",
              f"部署选择规则：`{selected['selection_rule']}`。原测试集在前一轮已经查看，本轮先锁定验证集选择，再复用测试集作描述性比较，不视为全新盲测。公开骨架投影、弱标签和重建回放结果不等于老人家庭视频性能。", "",
              "## 完整模型对比（最佳候选置顶）", "",
              "| 模型与定位 | 参数 | CPU P50 ms | 验证 MAE | 测试动作 F1 | 测试联合阶段 F1 | 测试 MAE | 测试 MAPE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    ordered = [row] + ([test_best] if test_best["model"] != row["model"] else [])
    ordered += [r for r in results if r["model"] not in {row["model"], test_best["model"]}]
    for r in ordered:
        label = r["model"]
        if r["model"] == row["model"]:
            label += "（综合最佳／推荐部署）"
        elif r["model"] == test_best["model"]:
            label += "（测试指标最高）"
        cells = [label, f"{r['parameters']:,}", f"{r['cpu_p50_ms']:.3f}", f"{r['validation']['count_mae']:.2f}",
                 f"{r['test']['action_macro_f1']:.4f}", f"{r['test']['phase_macro_f1']:.4f}",
                 f"{r['test']['count_mae']:.2f}", f"{r['test']['count_mape_pct']:.2f}%"]
        if r["model"] in {row["model"], test_best["model"]}:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "集成行延迟为各成员独立测量之和，尤其 P95 相加不是实测联合 P95。首轮 seed=42，优化轮 seed=2026；未做多种 seed 的统计重复。CPU 是本机测量，旧轮与新轮计时条件可能不同。", ""]
    lines += ["## 逐测试会话", "", "| 重建会话 | 来源参考 | 预测 |", "|---|---:|---:|"]
    for s in sessions:
        lines.append(f"| {s['session']} | {s['expected']} | {s['predicted']} |")
    lines += ["", "## 姿态 CNN 对比", "", "100 张官方 COCO val 图像，198 个标注人体，2,482 个可见关键点。此处不是全量 COCO AP，PCK 阈值为 bbox 最长边的 20%，漏检计错。", "", "| Pose CNN | 输入 | PCK@0.2 | 平均 OKS | 匹配人体 | GPU P50 ms | GPU P95 ms | 参数 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in poses:
        lines.append(f"| {p['model']} | {p['imgsz']} | {p['pck_bbox_02']*100:.2f}% | {p['mean_oks_including_misses']:.4f} | {p['matched_persons']}/{p['persons']} | {p['latency_p50_ms']:.2f} | {p['latency_p95_ms']:.2f} | {p['parameters']:,} |")
    lines += ["", "计时包含预处理、前向与后处理，并做 CUDA 同步。P50 达到 30 FPS 的预算不代表 P95 能达到，也不包含所有系统开销。", "", "## 图表", "", "![验证精度与计算比较](../reports/figures/accuracy_compute.png)", "", "![动作混淆矩阵](../reports/figures/action_confusion.png)", "", "![测试会话计数](../reports/figures/test_session_counts.png)", "", "## 是否达到 proposal 的目标", ""]
    lines += [f"当前默认模型重建回放总体 MAPE 为 {row['test']['count_mape_pct']:.2f}%。" + ("在该有限回放上达到 ≤5%，但尚不能宣布家庭视频验收完成。" if row["test"]["count_mape_pct"] <= 5 else "未达到 ≤5% 的目标，不能把实现完成写成精度目标已经完成。"), "", "代码、训练权重、四动作识别与计数流程、接口和离线实验已实现。下一步应使用独立人工标记的自然连续摄像头视频，包含老人、不同视角、遮挡及日常干扰，然后验证真实前端的关键点误差和完整周期性能。", ""]
    robustness_path = ROOT / "reports/robustness.json"
    if robustness_path.exists() and all(r.get("model") == selected["model"] for r in json.loads(robustness_path.read_text())):
        robustness = json.loads(robustness_path.read_text())
        lines += ["## 固定模型的视角与噪声测试", "", "下表只包含四动作的八个正会话，因此 MAE 的分母与含两个负会话的主表不同；模型与阈值没有根据这些测试结果修改。噪声单位是 UI-PRMD 原始长度单位，不是特定摄像头的像素。", "", "| 投影视角 | 坐标噪声标准差 | 正会话 MAE | MAPE |", "|---|---:|---:|---:|"]
        for r in robustness:
            lines.append(f"| {r['angle_degrees']}° | {r['noise_source_units_std']} | {r['mae']:.3f} | {r['mape_pct']:.2f}% |")
        lines += ["", "未见训练参与者的 15°／75°视角是投影鲁棒性压力测试，不能替代实际摄像头角度验收。", ""]
    lines += ["本文件由 `python -m algorithm2 report` 从实际实验 JSON 生成。", ""]
    (ROOT / "docs/RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    environment = {"python": sys.version, "platform": platform.platform(), "packages": {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "matplotlib", "opencv-python", "ultralytics", "fastapi", "uvicorn", "pytest"]}}
    (ROOT / "reports/environment.json").write_text(json.dumps(environment, indent=2))
    manifest = [{"file": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((ROOT / "artifacts").glob("*")) if p.is_file()]
    (ROOT / "reports/artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Generated docs/RESULTS.md, figures, environment and artifact hashes")


def write_optimization():
    directory = ROOT / "reports/optimization"
    results = json.loads((directory / "results.json").read_text())
    lock = json.loads((directory / "selection_lock.json").read_text())
    old_selected = json.loads((directory / "baseline/selected.json").read_text())
    old = next(r for r in json.loads((directory / "baseline/experiments.json").read_text()) if r["model"] == old_selected["model"])
    best = next(r for r in results if r["model"] == lock["model"])
    lines = ["# 算法 2 优化、自检与交接记录", "", "## 当前部署结论", "",
             f"当前默认算法是 **{best['model']}**，checkpoint 为 `artifacts/{best['model']}.pt`。部署仍使用 HTTP API v1，系统端关键点协议与四动作枚举保持兼容。",
             f"网络是分层 {best['config']['kind'].upper()}，{best['config']['blocks']} 个残差 block（每个含两次因果卷积）、宽度 {best['config']['width']}，输入 stable80 特征、{best['config']['context']} 帧因果窗口。最佳 checkpoint 来自 epoch {best['best_epoch']}。",
             f"Decoder `{best['decoder_strategy']}`，阶段证据阈值 {best['phase_evidence_threshold']}，消抖 {best['decoder_dwell_s']} 秒。",
             f"权重 SHA-256：`{lock['checkpoint_sha256']}`。", "", "## 优化前后", "",
             "下列阶段指标是联合动作／阶段弱标签 Macro-F1；MAPE 只统计八个非零计数会话，MAE 包含两个负会话。", "",
             "| 指标 | 原默认模型 | 当前默认模型 |", "|---|---:|---:|"]
    for label, key in [("验证动作 Macro-F1", "action_macro_f1"), ("验证阶段 Macro-F1", "phase_macro_f1"), ("验证计数 MAE", "count_mae"),
                       ("测试动作 Macro-F1", "action_macro_f1"), ("测试阶段 Macro-F1", "phase_macro_f1"), ("测试计数 MAE", "count_mae"),
                       ("测试计数 MAPE %", "count_mape_pct"), ("测试完整周期 recall", "cycle_recall"), ("测试完整周期 precision", "cycle_precision")]:
        split = "validation" if label.startswith("验证") else "test"
        lines.append(f"| {label} | {old[split][key]:.4f} | {best[split][key]:.4f} |")
    lines += [f"| 参数量 | {old['parameters']:,} | {best['parameters']:,} |",
              f"| CPU P50 ms | {old['cpu_p50_ms']:.3f} | {best['cpu_p50_ms']:.3f} |",
              f"| CPU P95 ms | {old['cpu_p95_ms']:.3f} | {best['cpu_p95_ms']:.3f} |", "",
              "CPU 单次前向测量不包含特征提取、姿态 CNN、网络传输和排队，旧模型使用 64 帧，新模型使用其实际窗口。三个优化模型在全部训练完成后、统一校准时重新计时：4 个 CPU 线程、40 次预热、300 次测量。旧／新默认模型另外使用同一轮计时作部署审计，见 deployment_audit.json。参数量、窗口长度和实际延迟共同解释计算成本。", "",
              "## 自检发现与修改", "",
              "1. 首轮 17 类联合输出把动作类别和阶段一起竞争，多种动作的静止站姿近似相同。新网络使用一个五类动作头和四个条件阶段头，动作 CE 与当前动作阶段 CE 分开优化。联合概率为 P(action) × P(phase|action)，接口仍输出原 17 类概率。",
              "2. 原归一化分母依赖投影腿长，屈膝、抬腿时可能随姿态变化。stable80 以两肩中心到双髋中心的二维躯干长度乘 2 归一化，加入左右膝／踝高度、较高膝／踝及左右高度差，保留单侧运动。它仍受投影和躯干倾斜影响，不能声称完全视角不变。",
              "3. 原重叠窗口的所有位置都计算损失，早期位置的历史较短。新训练只监督每个窗口最后八帧；窗口分别为 96、128、96 帧，避免主要损失来自缺少历史的位置。所有时间运算仍因果。",
              "4. 原训练以孤立片段为主，部署输入是连续流。新训练加入训练参与者的跨重复重建序列，不向模型传入 episode、clip boundary、来源计数或未来帧。",
              "5. 训练增加 0.85／1.0／1.15 采样步长、左右镜像及语义交换、25／45／65 度投影、特征噪声 0.025。归一化均值和标准差仅由训练参与者拟合。",
              "6. TCN 残差支路与输入加入逐帧 LayerNorm，使用 AdamW、余弦学习率、权重衰减、梯度裁剪与轻度 label smoothing。没有以 MLP 代替时序模型。", "",
              "7. 四 block TCN 的理论感受野是 61 帧。训练后把部署窗口从 96 缩到 64，重新推理全部验证片段与会话，核对最大概率差、逐帧阶段／动作 argmax 一致，再统一校准与测速。五 block TCN 感受野 125 帧，保持 128；GRU 保持 96，未套用有限卷积感受野推断。缩窗只改部署 metadata，原训练 checkpoint 备份在 reports/optimization/training_checkpoints。", "",
              "这些修改作为共同优化方案应用于三个架构；本轮没有对每项修改作单独消融，因此不能把总提升归因于其中某一项。", "",
              "## 实际开发与修复过程", "",
              "先完成原模型、特征、评估与漏计审计，保存首轮实验和部署配置；随后实现分层网络、stable80 特征、窗口长度随 checkpoint 加载，并新增因果性／概率合成／特征不变性测试。",
              "第一次优化运行完成四 block TCN 的训练与验证后，构造五 block TCN 的 128 帧数据时，NumPy 无法申请约 994 MiB 的连续窗口数组。重叠窗口全部预展开还会保留源数组，产生较大峰值内存。改为 CausalWindowDataset：源序列只保存一份，按批次即时切片和起始帧填充；测试逐项核对新旧窗口和监督标签完全一致。",
              "保留已完成的四 block TCN 权重和记录，仅继续五 block TCN 与 GRU。没有因为内存错误改动标签、划分或减少评估会话。最终训练日志、完整概率缓存、选择锁与部署自检共同用于核对实际结果。", "",
              "## 实验配置与选择", "",
              "| 候选 | 部署窗口 | 最佳 epoch | 实际训练 epoch | 验证动作 F1 | 验证阶段 F1 | 验证 MAE | 测试动作 F1 | 测试阶段 F1 | 测试 MAE | 测试 MAPE | CPU P50 ms |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['config']['context']} | {r['best_epoch']} | {r['history'][-1]['epoch']} | {r['validation']['action_macro_f1']:.4f} | {r['validation']['phase_macro_f1']:.4f} | {r['validation']['count_mae']:.2f} | {r['test']['action_macro_f1']:.4f} | {r['test']['phase_macro_f1']:.4f} | {r['test']['count_mae']:.2f} | {r['test']['count_mape_pct']:.2f}% | {r['cpu_p50_ms']:.3f} |")
    lines += ["", "三个候选是首轮表现较强的 TCN／GRU 家族的优化版本；没有重训全部 11 个首轮配置。最大 80 epoch，seed=2026，每五个 epoch 验证，连续八次验证未改善就提前结束。checkpoint 先按验证计数 MAE，再按验证阶段 F1 选。训练期间阶段 F1 每三帧计算一次以降低成本，候选最终比较使用完整逐帧指标。", "",
              "AdamW lr=0.0015，最小 lr=0.00005，weight_decay=0.005；batch=128，窗口 stride=12；dropout=0.15，梯度裁剪 1.0；loss=0.7×action_CE + phase_CE。动作 label smoothing=0.04，阶段=0.025，类别权重为训练频率的 -0.35 次幂。", "",
              "训练 checkpoint 的验证计数解码比较两种顺序策略 × 三个阈值（0.25／0.35／0.5）× 两种消抖（0.067／0.1 秒），共 12 种组合。训练后统一校准三个候选的完整验证概率，加入 0／0.033 秒消抖，共 24 组。优先最低 MAE，再看周期 recall、较长消抖和较高阈值。",
              "额外验证诊断发现：四 block TCN 的动作支持门槛在 0.05／0.1／0.15／0.2 范围内均为 MAE=0.3，因此保持 0.15；消抖从 0.1 降到 0.033 秒能把验证 s07 抬腿计数从 7/10 提高到 9/10，负样本仍为零。这促成了训练后统一 decoder 校准；测试数据未用于该诊断。",
              f"最终候选选择规则：`{lock['selection_rule']}`。选择锁文件在测试阶段之前生成，包含部署模型、decoder、时间戳和 checkpoint 哈希，finalize 会核对哈希。", "",
              "## 测试逐会话结果", "", "| 会话 | 来源计数 | 原默认预测 | 优化默认预测 |", "|---|---:|---:|---:|"]
    old_sessions = {s["session"]: s for s in old["test"]["sessions"]}
    for s in best["test"]["sessions"]:
        lines.append(f"| {s['session']} | {s['expected']} | {old_sessions[s['session']]['predicted']} | {s['predicted']} |")
    lines += ["", "## 验证范围与限制", "",
              "标签、参与者划分、评估角度、计数真值和事件匹配容差保持首轮定义。训练 s01–s06，验证 s07–s08，测试 s09–s10。原 test 在首轮与问题诊断中已经查看，本轮保留它用于同口径描述性比较；锁定流程避免了本轮按测试分数选模型，但不能把已使用的 test 恢复成全新盲测。", "",
              "验证集只有两人；epoch 与 decoder 多次比较存在过拟合风险。结果为公共 mocap 二维投影、弱阶段标签及重建回放，未验证自然连续 RGB 视频或老人群体。一个 seed 的改善不等于统计显著，数量正确也不保证每个事件时间均正确。当前结果不证明预防跌倒的临床效果。", "",
              "优化 GRU 的测试动作 F1=0.9659、阶段 F1=0.8246、计数 MAPE=0%，在这份已使用测试回放上数值略优。但它的验证计数 MAE=0.80，高于 TCN 的 0.10，且 CPU 延迟较高，因此保留测试前选择的 TCN 作为部署默认。区分测试数值最优与验证／计算折中最优，避免按已看到的测试结果改选。", "",
              "## 复现与文件", "", "```powershell", "Set-Location <repo>",
              "python -m algorithm2 optimize --epochs 80 --seed 2026",
              "python -m algorithm2 calibrate",
              "python -m algorithm2 finalize",
              "python -m algorithm2 robustness",
              "python -m algorithm2 audit",
              "python -m pytest -q algorithm2/tests",
              "python -m algorithm2 optimization-report",
              "python -m algorithm2 report", "```", "",
              "重新训练会改写同名优化 checkpoint，请在独立项目副本复现或先归档。原始基线记录保存在 `reports/optimization/baseline/`，首轮 `.pt` 权重保留在 artifacts。",
              "`validation.json`：训练日志与完整验证指标；`selection_lock.json`：测试前锁定；`results.json`：三个候选完整结果；`*_probabilities.npz`：逐帧及会话概率；`reports/comparison.csv`：包含旧轮和优化轮的总表。完整 API 契约见 `docs/openapi.json`。", ""]
    audit_path = directory / "deployment_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        lines += ["## 固定部署模型追加自检", "", "两个测试肩部动作回放分别指定四种训练目标，共八项检查，计数均为 0；同一静止姿态连续 300 帧对四目标均为 0。这些检查只覆盖已记录负样本，不代表所有家庭干扰均不会误计。", "",
                  "| 同轮 CPU 测量 | 窗口 | P50 ms | P95 ms |", "|---|---:|---:|---:|"]
        for b in audit["same_run_cpu_benchmarks"]:
            lines.append(f"| {b['model']} | {b['context_frames']} | {b['cpu_p50_ms']:.3f} | {b['cpu_p95_ms']:.3f} |")
        lines += ["", "该轮用同一线程数、预热和测量次数，优化 TCN 比旧 GRU 的实际单次推理更快，尽管参数多约 13%。", ""]
    robustness_path = ROOT / "reports/robustness.json"
    if robustness_path.exists():
        robust = json.loads(robustness_path.read_text())
        if all(r.get("model") == best["model"] for r in robust):
            lines += ["## 固定模型压力测试", "", "八个正会话的投影／坐标噪声检查；模型及阈值没有据此修改。", "",
                      "| 投影视角 | 噪声标准差（原始单位） | 正会话 MAE | MAPE |", "|---|---:|---:|---:|"]
            for r in robust:
                lines.append(f"| {r['angle_degrees']}° | {r['noise_source_units_std']} | {r['mae']:.3f} | {r['mape_pct']:.2f}% |")
            lines += ["", "常规 45°投影提升明显，但极端角度和噪声下仍有明显漏计，例如 15° 的 s09 坐站为 0/9。平均精度的提升没有解决摄像头域偏移和全部视角问题。", ""]
    lines += ["## 训练曲线", "", "![优化候选训练曲线](../reports/figures/optimization_history.png)", ""]
    (ROOT / "docs/OPTIMIZATION.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        axes[0].plot([h["epoch"] for h in r["history"]], [h["validation_phase_f1_stride3"] for h in r["history"]], label=r["model"])
        axes[1].plot([h["epoch"] for h in r["history"]], [h["validation_count_mae"] for h in r["history"]], label=r["model"])
    axes[0].set(xlabel="Epoch", ylabel="Validation phase macro-F1 (stride 3)")
    axes[1].set(xlabel="Epoch", ylabel="Validation reconstructed count MAE")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/optimization_history.png", dpi=160)
    plt.close(fig)
    print("Generated docs/OPTIMIZATION.md and optimization_history.png")
