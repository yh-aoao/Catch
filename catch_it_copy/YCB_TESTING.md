# YCB测试接入

来源 https://github.com/elpis-lab/YCB_Dataset 。三个已下载物体：tennis_ball（0.058kg）、baseball（0.138kg）、tomato_soup_can（0.349kg）。每个对象的上游commit和文件SHA256记录于assets/ycb/*.provenance.json，许可证在UPSTREAM_LICENSE。下载器不执行上游代码。

## 用法

从catch_it_copy运行，保持object_eval=false：

```bash
python3 train_DCMM.py test=True task=Catching_TwoStage num_envs=1 object_motion=throw test_object=ycb:tennis_ball object_eval=false viewer=true checkpoint_catching=/home/isee/Catch/best_model/throw_catch_best_reward_497.09.pth
```

Roll/Bounce沿用对应模型已验证的测试命令，添加test_object=ycb:baseball或test_object=ycb:tomato_soup_can。Track必须用Track checkpoint，不要给checkpoint_tracking传Catch模型。默认test_object=default保持原任务。

若服务器没有同步资源，或需要添加其他上游物体：

```bash
python -m ycb_objects.download --objects tennis_ball baseball tomato_soup_can
python -m ycb_objects.download --objects lemon
```

下载器读取同一次确定的上游commit，再下载XML所引用的网格、纹理和许可证。新增物体结构仍需单环境验证；不是所有YCB物体适合本手或本发射任务。

## 物理与接触

使用真实外观网格和CoACD碰撞部件，不使用球形碰撞替代。额外保留不可见且不碰撞的object几何作为旧接口的尺寸标记；多部件的物体接触合并，手/底座接触中的物体部件ID映射回该标记。保持一个自由关节，不改变观测动作维数。

移除上游统一1e-3惯量，用各碰撞部件均分总质量后由MuJoCo推算惯量。这是近似，不是实测惯量；凸包重叠、均分质量都会影响精度。每次reset保留目标质量和惯量，物体接触参数跟随原任务，并同步到各碰撞部件。

网格与纹理不意味着材料动力学准确：真实网球弹性、罐子质量分布、空心壳体惯量仍未标定。旧奖励/成功判定中的球半径读取采用网格关于局部原点的包围半径近似；尤其罐子的保持区域、桌面支撑和离地检测不应直接用于正式泛化论文指标。未宣称完成任意形状评估标准的重设计。

## 验证状态

已做下载资源存在性、质量总和、非碰撞标记、多部件接触映射、路径校验及语法检查。本机MuJoCo DLL无法加载，未完成真实模型编译/重置/运行验证。先num_envs=1 viewer=true，确认形状、接触和放置正常后再统计成功率。

仅接入train_DCMM.py现有物体测试入口，蒸馏学生独立CLI未扩展；未修改既有Throw/Roll/Bounce默认行为。

## 2026-10-26 接触参数修复

占位 object 几何现在同时继承原任务的 friction、solref、solimp、condim，继续保持不可见、零质量、不可碰撞。
此前重置将占位体的默认接触参数复制到网格，覆盖了 Throw/Roll 初始化时设置的参数。
现在重置同步保留原任务参数；Bounce 在基础环境重置时更新占位体的参数后，仍同步这些最新值。
原来的网格形状、质量和惯量计算方式没有改变。

在能加载 MuJoCo 的服务器上，可不启动 viewer 检查网格编译、接触属性和惯量：

```bash
python -m ycb_objects.audit --objects tennis_ball baseball tomato_soup_can
```

输出包括碰撞部件数、质量、质心、主惯量及主轴姿态，并列出旧物体 XML 的惯量。
该命令验证独立物体编译，不替代完整任务 reset/step 和抓取效果测试。
当前本机仍有 MuJoCo DLL 初始化错误，因此不能声称已完成编译或动态验证。

Bounce 默认 Catch（5fe75d5f）和 Track（july22）配置有位置、方向、自旋随机化；
质量、半径、摩擦、反弹参数、关节阻尼及初始高度/速率的范围上下限相同，实际未开启这些参数的域随机化。
