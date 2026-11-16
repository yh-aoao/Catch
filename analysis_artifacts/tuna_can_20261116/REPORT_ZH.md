# YCB 金枪鱼罐替换验证

资源：elpis-lab/YCB_Dataset 的 tuna_fish_can，固定提交9e8c6488a2ff673d9aa48a91492fb89423c1b106。XML/OBJ/纹理/碰撞资源已下载，SHA256见 assets/ycb/tuna_fish_can.provenance.json。

实际使用真实扫描纹理外观、由扫描形状拟合的解析圆柱碰撞。矮圆柱使用PCA最短轴作为轴向，避免沿用高罐最长轴拟合导致直径/高度混淆。拟合直径约8.558cm、高3.416cm，导入质量171g；惯量为均匀实心圆柱近似，非实际罐体内部质量分布测量。滑动/滚动摩擦为工程先验，非YCB实测。

新目录 object_benchmark_creation_v2.json 删除 small_cylinder、保留无柠檬设定、加入 tuna_fish_can。旧目录和显式合成圆柱兼容代码不删除。Roll输出目录改为 roll_seven_object_tuna_{track,catch}_creation_dr_v2_seed123。Bounce不变，创建时采样参数；Roll Catch保持palm_z_v1。

验证：真实资产加载、碰撞尺寸范围断言通过；多次reset质量/惯量/尺寸不变，发射条件变化；有限奖励；修正后的矮圆柱40个不同发射种子均越过桌前边界（一个随机物理实例，不代表全部随机范围验证）。短轴修正后分别执行Track/Catch CPU PPO 16步运行检查。未进行完整训练或成功率评估。
