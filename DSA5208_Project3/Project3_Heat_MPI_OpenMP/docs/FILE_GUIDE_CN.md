# 中文文件说明（代码负责人交接用）

本项目选 2D Jacobi（二维网格的五点迭代）。每一步更新的单元格都依赖上一步的上下左右邻居，因此跨 MPI 进程的数据交换不可省略，不属于 embarrassingly parallel。

- `src/common.hpp`：所有版本共用的参数解析、网格初值、可以直接计算的离散解析解、误差统计、CSV 导出。共用这一文件确保串行和并行的初值及误差口径一致。
- `src/serial.cpp`：单进程、单线程的基准版本。每步计算新网格并交换新旧数组，避免读到已经修改的数据。
- `src/parallel.cpp`：按行分给 MPI 进程；邻居进程每一步交换各自边界行；内部行由 OpenMP 多线程计算。支持 `blocking` 和 `overlap` 两种通信方式，也可以比较 `static` 和 `dynamic` 调度。
- `Makefile`：一键编译 `make`，一键验证 `make test`，清理 `make clean`。
- `scripts/validate.py`：测试多组尺寸和迭代次数，验证解析解，再逐单元格对比串行和并行结果，捕捉 off-by-one、边界和通信错误。
- `scripts/benchmark.py`：重复测速，取中位数，自动算强/弱扩展效率，并输出 CSV；计时不包含 MPI 初始化及文件写入。
- `scripts/plot_results.py`：用真正的 CSV 画运行时间、加速比、效率三类图，不生成假数据。
- `cluster/run_pbs.pbs` / `cluster/run_slurm.sbatch`：两种集群调度器的示例提交脚本，使用前按 Atlas/Vanda 实际环境调整模块和资源参数。
- `docs/REPORT_HANDOFF.md`：给两位写 report 的同学的公式、实验方法、限制、AI 使用说明素材。
- `README.md`：英文运行手册，提交 source code 时供老师复现。
- `results/`：预留真实集群测试得到的 CSV 和图片，初始不包含捏造数据。

**给另外两位同学的重点**：最终 report 必须包含 Atlas/Vanda 实际测得的 runtime figures。跑完 `cluster/run_pbs.pbs`（或适用的 Slurm 脚本）后把 `results/*.csv` 和日志交给 report 负责人。所有姓名、学号和真实 AI 使用流程也需要写入 report。项目没有用外部真实数据，不需要编造数据分析结论。
