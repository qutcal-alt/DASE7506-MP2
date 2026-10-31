# DASE7506 MP2 独立交付版

固定方法的官方 test 成绩为 **393/467 = 84.1541755888651%**，validation 为 **755/798**。
本目录只提供这一版的推理权重与固定训练配方。没有加入后续的题目规则或其他候选模型。

最新修订报告为 **7 页 Word**：`report/Experiment_Report_Revised.docx`，文字源稿为 `report/Experiment_Report_Revised.md`。原 PDF 版本保留在 `report/Experiment_Report.pdf`。
官方代码基线为 `xudongwu-0/DASE7506-MP2` 的提交 `bb801f4f2e7629d270df765ab12c333b2dc616bb`。

## 目录

- `code/`：官方数据、原预测和评分工具、最终推理与独立训练代码。
- `code/runs/model_bundle/`：唯一默认推理 checkpoint，含全部权重、统计量和配置。
- `guide/GUIDE.md`：原始课程要求。
- `report/`：报告、可编辑源稿和报告用的数值证据。
- `records/`：历史选型、官方评分、重训、基线和固定权重消融记录。
- `CHECKPOINT_MANIFEST.json`：全部推理资产的 SHA-256 与未压缩大小。
- `METHOD_MANIFEST.json`：冻结推理代码与 checkpoint 的对应关系。
- `RELEASE_MANIFEST.json`：本次交付的文件清单与哈希。

## 复现冻结成绩

在本目录的 PowerShell 中执行：

```powershell
python -m venv .venv-eval
.\.venv-eval\Scripts\python.exe -m pip install -r code\requirements.txt
.\.venv-eval\Scripts\python.exe code\reproduce.py
```

Linux/macOS：

```bash
python3.12 -m venv .venv-eval
.venv-eval/bin/python -m pip install -r code/requirements.txt
.venv-eval/bin/python code/reproduce.py
```

只需官方 NumPy 依赖，不需要 GPU、PyTorch、LightGBM、SciPy 或网络。
`reproduce.py` 核验官方 19 个受保护文件、历史源码快照和默认推理资产，限制四线程，再分别启动官方预测器与独立评分器。
结果写在 `code/runs/reproduction/`。预期 test JSON 为：

```json
{"correct": 393, "n": 467, "accuracy": 84.1541755888651}
```

正式入口为 `code/submission.py`。`baseline.py` 在当前管线中提供来源特征，不是默认提交入口。

## 从头训练

```powershell
python -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install -r code\requirements-train.txt
.\.venv-train\Scripts\python.exe code\train_export.py
.\.venv-eval\Scripts\python.exe code\reproduce.py --checkpoint runs/retrained_bundle --output-dir runs/retrained-reproduction
```

固定环境为 Python 3.12、NumPy 2.5.3、SciPy 1.16.3、PyTorch 2.7.1+cu128、LightGBM 4.6.0。
训练需要兼容 CUDA 的单张 GPU；本机使用 RTX 5070。线性模型和小型树在 CPU 上拟合。
脚本只读 train 输入、train 参考答案及固定分组折，重新生成教师 OOF 分数、网络和树模型，不需要任何旧项目或既有训练预测缓存。
默认输出 `code/runs/retrained_bundle`，已存在时拒绝覆盖；可用 `--output` 指定新目录。

为便于独立复现，固定训练所需的函数集中在 `model_training.py`。这些函数从原搜索代码原样提取，原测试前源码快照保留在 `records/selection/source_snapshot/`。
参考答案只比较最终数值，因此标签中的“正确候选”不表示其每步推理都正确。

## 基线和推理消融

```powershell
.\.venv-eval\Scripts\python.exe code\evaluate_ablation.py
```

这项分析不重新拟合权重，不改变最终方法。各变体复用题目特征，并分别交给官方评分器评分，结果在 `records/ablations/`。
这些变体的共享特征计时不作为课程的完整新进程 CPU 用时。课程资源测量以 `reproduce.py` 的完整预测加独立评分为准。

## 资源和一致性

默认推理资产按未压缩 NPY 成员及配置计为 **643,525 字节**，低于 16 MiB。
本地 CPU 为 Intel Core Ultra 5 250K Plus；参考 Xeon Platinum 8457C 的 30 秒资格仍需在参考机上实测。
本次整理保留了原 84.15% 版的推理文件与 checkpoint；只整理固定训练依赖、复现工具和实验材料。
各环境重新拟合可能有浮点差异，直接使用默认冻结权重即可复现报告成绩。

## AI 和复用说明

AI 助手参与代码实现、实验运行、目录整合、报告撰写与编辑。
148 维特征与算术检查来自原项目；数量来源模块复用用户的 `quantity_features_v2.py`；树数组导出结构参考用户的 `tree_training.py`。
报告文字参考 Humanizer；Word 修订版进一步参考 Anti-Defensive Writing（https://github.com/Adkid-Zephyr/anti-defensive-writing-Skill），保留实际的 AI 协助说明与评测范围。
官方数据许可证与归属保留在 `code/data/LICENSE.txt` 和 `NOTICE.txt`。

开发期间已多次观察公开 test 分数。本报告将其作为已冻结配置的公开评测结果，不宣称一次性盲测。
最终课程提交仍需提供公开代码和 checkpoint 链接；本目录未自动上传或提交到外部平台。
