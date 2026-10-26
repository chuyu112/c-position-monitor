# C 位监测仓库维护

先读 README.md 和 docs/技术方案.md。当前入口为 c_monitor；c_detect 保留候选检测算法。legacy 是历史来源，不在其中开发新功能。

portrait 和 landscape 的最终标注有独立帧数、画布、音乐起点及 SHA256。修改基线须由用户明确要求；不得用早期 c5 或横屏版本覆盖当前数据。正常代码整理保留数据字节和最终成员累计值。

姓名与主舞角色来自人工复核，跟踪 ID 不等于成员姓名。不要将几何候选描述为已经实现全部业务语义。

检查运行 python -m unittest discover -s tests -v 以及两版 c_monitor validate。变更渲染时使用独立 outputs 子目录做真实或合成素材短片检查，顺序解码确认帧数和音轨。输出目录须为空或不存在，禁止覆盖原始素材及既有成品。
