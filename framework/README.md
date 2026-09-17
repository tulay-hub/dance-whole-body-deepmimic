# 全身舞蹈训练框架入口

`isaaclab_shared` 是到共享 Isaac Lab/DWAQ/DeepMimic 基座的软链接：
`frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot`。

全身舞蹈的实际配置在共享源码的
`source/legged_lab/legged_lab/tasks/locomotion/deepmimic/config/lens110/`，不要在项目目录再复制一份
DeepMimic 源码；项目特有的动作、checkpoint 和导出包只放在本项目的 `data/` 与 `exports/`。
