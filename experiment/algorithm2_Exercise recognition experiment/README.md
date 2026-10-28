# 算法2：四类康复动作识别与计数实验

本目录按算法1的方式集中放置实验源码、处理后数据、训练权重、对比结果和数据位置说明。
这里的**算法2就是系统中的 `system code/algorithms/exercise/`**。

整理基于仓库提交 `a34f45c13da352df926c0b47bdb658ed686a0be0`，日期为 2026-10-28。
系统原目录仍然保留；本目录为可独立运行的实验快照。实验重训与系统部署分别管理。
新增的 `verify.py` 会检查快照与系统原文件是否一致，并通过真实子进程和 HTTP 检查系统。

## 1. 任务和方法

| 数据动作 | 对外名称 | 功能 |
|---|---|---|
| m01 | `squat` 深蹲 | 识别动作、阶段与完整次数 |
| m02 | `hurdle_step` 跨栏步 | 识别动作、阶段与完整次数 |
| m05 | `chair_sit_to_stand` 坐站 | 识别动作、阶段与完整次数 |
| m06 | `standing_leg_raise` 站立抬腿 | 识别动作、阶段与完整次数 |
| m07 | 肩部动作 | 四类目标动作的负样本 |

输入为 COCO17 的 `[x, y, confidence]` 关键点；有效输入按时间重采样为 30Hz。
每类动作包含四个阶段，外加 `other`，通过有顺序的阶段循环确认完整动作。
默认时间模型为 `opt_tcn_b4_w48`，姿态模型为 `yolo11n-pose.pt`。
第一轮包含 11 个时间模型/特征候选，第二轮包含 3 个优化候选；设置见 `pipeline/config.py`。

## 2. 目录

```text
algorithm2_Exercise recognition experiment/
├── README.md                 本说明
├── data.txt                  数据位置、来源与恢复命令
├── requirements.txt          算法2实验及系统检查依赖
├── run.py                    实验统一入口，python run.py --help
├── verify.py                 数据/SDK/API/真实系统回归检查
├── pipeline/                 下载、准备、模型对比、优化、选型、审核和报告脚本
├── homeease/                 实验所需特征、网络、因果推理、计数器、姿态与独立API
├── data/
│   └── processed/            已提交的 records.json 与 skeletons.npz
├── artifacts/                训练权重、YOLO权重、默认模型配置
├── examples/                 四类动作的测试回放和接口样例
├── reports/                  已交付的实验指标、图、概率与权重校验清单
├── docs/                     独立API的 OpenAPI 契约
└── results/
    ├── runtime_check.json    本次实际运行的机器可读检查结果
    └── 检查说明.txt          本次检查结论、修复和未覆盖范围
```

`data/raw/uiprmd/` 和 `data/coco/` 按原约定不提交；有处理后数据、权重和样例即可运行默认模型。
运行数据详情请直接查看 [data.txt](data.txt)。本目录含完整实验副本，约 131MiB；不是系统运行时日志目录。

## 3. 环境与快速检查

推荐 Python 3.12。CPU 即可完成本次检查；使用 GPU 时另行安装匹配的 PyTorch/CUDA 版本。
本次实际包版本保存在 `results/runtime_check.json`，原实验环境保存在 `reports/environment.json`。

在仓库根目录执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r "experiment/algorithm2_Exercise recognition experiment/requirements.txt"
cd "experiment/algorithm2_Exercise recognition experiment"
..\..\.venv\Scripts\python.exe run.py --help
..\..\.venv\Scripts\python.exe verify.py
```

已激活环境时可直接使用 `python`。若本机没有 3.12，请使用已安装的兼容解释器创建环境。
完整系统检查需保留本目录在仓库内，以便找到 `system code/run.py` 及系统公共源码。
只运行此实验副本时：

```powershell
python verify.py --skip-system
```

`verify.py` 使用临时数据库、临时日志、随机本地端口和一段生成的空场景负样例视频，
不打开物理摄像头/麦克风。退出码为 0 表示所选检查全部通过，报告默认写入 `results/runtime_check.json`。
完整系统启动会同时尝试初始化其他模块；算法1/算法3的依赖、权重和可用性不属于本检查的通过条件。
如果要使用全部系统功能，应按 `system code/requirements.txt` 和系统 README 安装完整依赖。

## 4. 实验复现顺序

以下命令均在本实验目录运行，读取和写入的都是本目录的实验副本。
下载数据需要联网；训练、重新选型和生成报告会覆盖各阶段拥有的原结果，建议先保留快照。

```powershell
# 原始数据可选恢复（现有处理后数据可直接运行后续检查）
python run.py download
python run.py sources

# 从原始数据重新构建；包含离线弱标签和审核图
python run.py prepare

# 姿态CNN对比；下载 COCO 子集
python run.py pose --images 100

# 第一轮时间模型与特征对比
python run.py train --epochs 25 --seed 42
python run.py select

# 第二轮优化；在验证集校准、锁定模型后才评估测试集
python run.py optimize --epochs 80 --seed 2026
python run.py calibrate
python run.py finalize

# 固定模型鲁棒性、负样本/静止检查和SDK回放
python run.py robustness
python run.py audit
python run.py replay

# 从已有结果生成说明和图；不代表重新训练
python run.py optimization-report
python run.py report
```

`report` 生成本目录 `docs/RESULTS.md`，`optimization-report` 生成优化说明；
本次运行检查不执行这些会改写原实验结果的阶段。
直接浏览现有对比可查看 `reports/comparison.csv`、`reports/experiments.json`、
`reports/optimization/results.json`、`reports/robustness.json` 与 `reports/figures/`。

## 5. 独立API和系统入口

独立算法API（在本实验目录）：

```powershell
python -m uvicorn homeease.api:app --host 127.0.0.1 --port 8002 --workers 1
# 另一终端：用完整样例同时对比SDK与HTTP
python run.py replay --url http://127.0.0.1:8002
```

主要接口为 `/health`、`POST /v1/sessions`、`/v1/sessions/{id}/frames`、
`/v1/sessions/{id}/control`；批量输入最多 300 帧，`request_id` 支持幂等重试。
接口数据必须是17个COCO关键点，时间戳严格递增。

HomeEase 集成系统（在仓库根目录）：

```powershell
cd "system code"
python run.py --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000
# 结束时使用页面操作，或另一终端在同一目录运行：
python run.py --stop
```

系统算法2使用原目录 `system code/algorithms/exercise/`，经独立工作进程调用。
`verify.py` 检查网页资源、四类回放、会话导出、诊断保留、控制操作和真实上传视频处理。
它还检查摄像头模式的四类部署策略初始化；这不等于已经实测了物理摄像头采集。

## 6. 如何理解结果

原报告中默认模型的测试计数 MAPE 为 1.25%，来源是公开动作捕捉骨架重建投影回放。
它既不是本次检查重新测得的完整精度，也不是实际老人家庭录像上的准确率。
本次四个完整样例用于功能回归，结果见 `results/runtime_check.json`。

系统的真实相机/上传模式含 `training` 动作范围、部分动作的镜像融合及姿态循环计数扩展；
独立SDK和样例回放默认保留 `open` 范围与原始计数方式。
这些部署扩展不能直接沿用原报告精度；详情见 `system code/docs/exercise-integration.md`。

本次修复了 Windows CRLF 配置文件经过 Git 的 LF 规范化后，哈希清单失效的问题。
JSON 使用明确的 LF 规范化哈希，模型二进制继续按原始字节校验；未修改训练权重、阈值或计数规则。
实验与系统两份源码同步修复，并加入 LF/CRLF 和二进制校验回归检查。
