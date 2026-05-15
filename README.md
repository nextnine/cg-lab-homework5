# CG Lab Homework 5：Taichi Whitted-Style 光线追踪实验

本仓库实现了一个基于 **Taichi GPU Kernel** 的实时 Whitted-Style 光线追踪小实验。程序在 Kernel 中隐式构建场景几何体，通过迭代式光线弹射实现硬阴影、镜面反射，并额外加入了玻璃折射与 MSAA 抗锯齿选做功能。

![Taichi](https://img.shields.io/badge/Taichi-GPU%20Programming-blue)
![Ray Tracing](https://img.shields.io/badge/Rendering-Ray%20Tracing-orange)
![Python](https://img.shields.io/badge/Python-3.x-green)

---

## 目录

- [实验目标](#实验目标)
- [核心效果](#核心效果)
- [运行效果](#运行效果)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [实验原理](#实验原理)
- [任务实现说明](#任务实现说明)
- [交互控制](#交互控制)
- [关键代码设计](#关键代码设计)
- [选做功能](#选做功能)
- [常见问题](#常见问题)
- [总结](#总结)

---

## 实验目标

本实验围绕经典 **Whitted-Style Ray Tracing** 展开，目标包括：

1. **理论理解**
   - 理解 Ray Casting 与 Ray Tracing 的区别。
   - Ray Casting 只求主光线与场景的首次交点。
   - Ray Tracing 在首次交点基础上继续发射阴影、反射、折射等次级射线。

2. **全局光照入门**
   - 使用暗影射线实现硬阴影。
   - 使用反射射线实现理想镜面反射。
   - 使用折射射线模拟玻璃材质。

3. **GPU 编程思维**
   - 避免递归调用。
   - 将传统递归式光线追踪改写为 `for` 循环驱动的迭代式弹射。
   - 让每个像素在 Taichi Kernel 中独立并行计算。

---

## 核心效果

程序运行后会打开一个实时渲染窗口，画面包含：

- 无限大棋盘格地面。
- 左侧红色玻璃球。
- 右侧银色镜面球。
- 可移动点光源。
- 硬阴影。
- 镜面反射中的场景倒影。
- 折射与全反射效果。
- 可调 MSAA 采样数，用于减少边缘锯齿。

---

## 运行效果

<img width="480" height="388" alt="6eGo1V3t_converted" src="https://github.com/user-attachments/assets/79e02efd-9692-49f8-9f56-ec8387e8238f" />



## 运行环境

建议环境如下：

| 项目 | 要求 |
| --- | --- |
| Python | 3.8 或更高版本 |
| Taichi | 支持 `ti.ui.Window` 的版本 |
| GPU 后端 | CUDA / Vulkan / Metal / OpenGL，视平台而定 |

安装依赖：

```bash
pip install taichi
```

> 程序使用 `ti.init(arch=ti.gpu)` 初始化 GPU 后端。若当前环境没有可用 GPU，Taichi 可能会自动回退或报错，可根据本机情况改为 `ti.cpu` 进行调试。

---

## 快速开始

在仓库根目录运行：

```bash
python work5/main.py
```

运行后会出现标题为 **Ray Tracing Demo** 的窗口。右上角控制面板可实时调节光源位置、最大弹射次数和 MSAA 采样数。

---

## 项目结构

```text
cg-lab-homework5/
├── README.md          # 实验说明文档
└── work5/
    ├── main.py        # Taichi 光线追踪主程序
    └── imgui.ini      # UI 窗口布局配置文件
```

---

## 实验原理

### 1. Ray Casting 与 Ray Tracing

**Ray Casting** 的核心流程是：

1. 从相机向每个像素发射一条主光线。
2. 找到这条光线与场景中最近物体的交点。
3. 根据交点信息直接计算该像素颜色。

**Ray Tracing** 则在 Ray Casting 的基础上进一步发射次级射线：

- 阴影射线：判断交点是否被其他物体遮挡。
- 反射射线：模拟镜面反射。
- 折射射线：模拟透明材质中的光线传播。

因此，Ray Casting 更像是“看见物体”，Ray Tracing 则进一步模拟了“光线如何在场景中传播”。

### 2. Whitted-Style 光线追踪

本实验采用 Whitted-Style 模型。每条主光线从摄像机出发后，根据命中的材质类型执行不同逻辑：

```text
Primary Ray
    ├── 命中漫反射材质：计算光照与阴影，然后终止
    ├── 命中镜面材质：生成反射射线，继续追踪
    └── 命中玻璃材质：生成折射射线；若发生全反射，则改为反射射线
```

### 3. 反射向量

理想镜面反射方向由下式计算：

$$
\mathbf{R} = \mathbf{L}_{in} - 2(\mathbf{L}_{in} \cdot \mathbf{N})\mathbf{N}
$$

其中：

- $\mathbf{L}_{in}$：入射光线方向。
- $\mathbf{N}$：交点处表面法线。
- $\mathbf{R}$：反射光线方向。

代码中的 `reflect(I, N)` 函数即实现了该公式。

### 4. 折射向量

选做部分实现了基于斯涅尔定律的折射：

$$
\eta_i \sin\theta_i = \eta_t \sin\theta_t
$$

当光线从玻璃内部射向空气且入射角过大时，会出现根号项小于 0 的情况，此时没有有效折射方向，程序将其判定为 **全反射**。

---

## 任务实现说明

### 任务 1：隐式三维场景

本实验没有导入任何外部模型，而是在 Taichi 函数中直接定义几何体。

#### 1. 无限大棋盘格平面

- 位置：`y = -1.0`
- 法线：`(0, 1, 0)`
- 材质：漫反射
- 纹理：根据交点的 `x` 与 `z` 坐标生成黑白/灰白棋盘格

实现思路：

```python
p = ro + rd * t
ix = ti.floor(p.x * grid_scale)
iz = ti.floor(p.z * grid_scale)
if (ix + iz) % 2 == 0:
    hit_c = ti.Vector([0.3, 0.3, 0.3])
else:
    hit_c = ti.Vector([0.8, 0.8, 0.8])
```

#### 2. 左侧球体：红色玻璃球

- 位置：`(-1.2, 0.0, 0.0)`
- 半径：`1.0`
- 当前材质：玻璃材质 `MAT_GLASS`
- 颜色：略带红色，用于表现红色透射感

> 实验基础要求中左侧球体为红色漫反射球；本代码在此基础上完成了选做扩展，将其升级为红色玻璃球。

#### 3. 右侧球体：银色镜面球

- 位置：`(1.2, 0.0, 0.0)`
- 半径：`1.0`
- 材质：镜面反射 `MAT_MIRROR`
- 反射率：通过 `throughput *= 0.8 * obj_color` 模拟能量损失

#### 4. 材质 ID 系统

代码使用整数常量区分材质：

```python
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2
```

场景求交函数会同时返回：

```text
t, normal, color, material_id
```

这样主追踪函数就可以根据 `material_id` 选择不同的光线传播分支。

---

### 任务 2：迭代式光线弹射

GPU Kernel 不适合使用递归，因此程序将光线追踪写成循环形式。

核心变量：

| 变量 | 含义 |
| --- | --- |
| `final_color` | 当前像素最终累积颜色 |
| `throughput` | 光线能量/颜色衰减系数 |
| `ro` | 当前光线起点 |
| `rd` | 当前光线方向 |
| `alive` | 光线是否继续传播 |

伪代码如下：

```text
final_color = 0
throughput = 1

for bounce in range(max_bounces):
    求最近交点

    if 未命中:
        final_color += throughput * background
        break

    if 命中镜面:
        更新 ro 和 rd
        throughput *= 反射率
        continue

    if 命中玻璃:
        计算折射或全反射方向
        更新 ro 和 rd
        throughput *= 透射衰减
        continue

    if 命中漫反射:
        final_color += throughput * diffuse_shading
        break
```

该结构既符合 Whitted-Style 思想，又适合 GPU 并行执行。

---

### 任务 3：硬阴影与 Shadow Acne 处理

#### 硬阴影

当光线命中漫反射表面时，程序会从交点向点光源发射一条暗影射线：

```text
shadow_ray_origin = p + N * EPS
shadow_ray_direction = normalize(light_pos - p)
```

若暗影射线在到达光源之前击中了其他物体，则说明当前点位于阴影中，只保留环境光。

#### Shadow Acne

由于浮点数精度有限，如果暗影射线或反射射线直接从交点 `p` 出发，可能会立即再次命中自身表面，导致画面出现黑色噪点或错误阴影。

为解决该问题，程序使用一个很小的偏移量：

```python
EPS = 1e-4
```

并将新射线起点沿法线或新方向略微移出表面：

```python
shadow_ray_orig = p + N * EPS
ro = p + N * EPS
ro = p + new_rd * EPS
```

---

### 任务 4：UI 交互面板

程序使用 `ti.ui.Window` 创建实时交互窗口，并使用 `window.get_gui()` 添加滑动条。

可调参数包括：

| 控件 | 范围 | 默认值 | 作用 |
| --- | --- | --- | --- |
| Light X | -5.0 ~ 5.0 | 2.0 | 控制点光源 X 坐标 |
| Light Y | 1.0 ~ 8.0 | 4.0 | 控制点光源 Y 坐标 |
| Light Z | -5.0 ~ 5.0 | 3.0 | 控制点光源 Z 坐标 |
| Max Bounces | 1 ~ 5 | 3 | 控制最大光线弹射次数 |
| Samples / Pixel | 1 ~ 64 | 4 | 控制每像素采样数，数值越高抗锯齿越好但更慢 |

观察建议：

- 将 `Max Bounces` 调为 `1`：只能看到一次命中结果，镜面/玻璃效果明显受限。
- 将 `Max Bounces` 调为 `3` 或更高：镜面球中可以看到更完整的场景反射。
- 移动 `Light X/Y/Z`：可以观察地面和球体阴影实时变化。
- 增大 `Samples / Pixel`：边缘更平滑，但帧率会降低。

---

## 关键代码设计

### 1. 球体求交

球体求交基于二次方程。将射线代入球方程：

$$
\|\mathbf{O} + t\mathbf{D} - \mathbf{C}\|^2 = r^2
$$

求出最近的正根作为交点距离，并由交点到球心方向计算法线。

### 2. 平面求交

水平平面 `y = -1.0` 的求交只需要解：

$$
O_y + tD_y = -1.0
$$

因此：

$$
t = \frac{-1.0 - O_y}{D_y}
$$

当 `t > 0` 时，说明交点在光线前方。

### 3. 最近交点选择

`scene_intersect` 会依次测试两个球体和地面平面，并保存最小的有效 `t` 值。这样即使多个物体都与光线相交，也只会返回离相机最近的可见表面。

### 4. 漫反射着色

漫反射使用简化 Phong/Lambert 模型：

```text
color = ambient + diffuse
```

其中：

- 环境光：`0.2 * obj_color`
- 漫反射：`0.8 * max(0, dot(N, L)) * obj_color`

如果处于阴影中，则只保留环境光。

### 5. 背景色

当光线没有命中任何物体时，返回蓝绿色背景色：

```python
bg_color = ti.Vector([0.05, 0.15, 0.2])
```

这样镜面反射或折射最终离开场景时不会变成纯黑色。

---

## 选做功能

### 1. 折射与玻璃材质

本代码将左侧红色球体扩展为玻璃材质：

- 折射率：`GLASS_IOR = 1.5`
- 从空气进入玻璃和从玻璃进入空气时会自动切换折射率。
- 发生全反射时改用反射方向继续追踪。
- 使用 `throughput *= 0.96 * obj_color` 模拟玻璃的轻微能量吸收和红色透射。

### 2. MSAA 抗锯齿

每个像素内部随机采样多条主光线：

```python
jitter_x = ti.random(ti.f32)
jitter_y = ti.random(ti.f32)
```

多次采样后取平均：

```python
color /= ti.cast(samples_per_pixel[None], ti.f32)
```

这样可以减轻球体边缘和棋盘格边界处的锯齿。采样数越高，画面越平滑，但计算开销也越大。

---

## 常见问题

### 1. 为什么画面出现大量黑点或阴影异常？

通常是射线自相交导致的 Shadow Acne。需要确保暗影射线、反射射线和折射射线起点都加上 `EPS` 偏移。

### 2. 为什么 `Max Bounces = 1` 时镜面效果不明显？

弹射次数为 1 时，光线命中镜面后几乎没有继续追踪其他物体的机会，因此看不到完整反射。将其调为 3 或更高即可观察镜中场景。

### 3. 为什么 `Max Bounces = 2` 时玻璃效果不明显？

玻璃球的视觉效果通常需要“进入玻璃”和“离开玻璃”至少两次表面交互：主光线第一次命中左侧玻璃球时，只是计算折射方向并进入球体；第二次弹射通常还在球体内部传播，或刚刚命中背面的球面准备再次折射射出。由于 `Max Bounces = 2` 时循环次数已经耗尽，程序会提前用背景色收尾，后续的出射折射、全反射、以及折射后再命中地面/镜面球的路径都无法继续计算。因此它只能表现出很弱的透明感或偏红透射感，玻璃内部的层次、边缘折射和全反射高光都不明显。将 `Max Bounces` 提高到 `3`、`4` 或 `5` 后，光线有更多机会完成“入射—内部传播—出射—再次命中场景”的链条，玻璃效果会更清楚。

### 4. 为什么提高 Samples / Pixel 后帧率降低？

MSAA 本质上是在每个像素中发射更多主光线。例如从 4 增加到 64，理论上每帧计算量接近增加 16 倍，因此实时帧率会下降。

### 5. 为什么使用循环而不是递归？

GPU 上大量线程并行执行时，递归会带来栈空间、控制流和性能问题。固定最大弹射次数的循环更适合 GPU Kernel，也更容易控制性能上限。

---

## 总结

本实验完成了从基础 Ray Casting 到 Whitted-Style Ray Tracing 的扩展：

- 使用隐式几何体搭建三维场景。
- 通过材质 ID 区分漫反射、镜面和玻璃材质。
- 使用暗影射线实现硬阴影。
- 使用迭代循环实现多次光线弹射。
- 使用 `throughput` 累积光线能量衰减。
- 使用 `EPS` 偏移解决自相交问题。
- 通过 Taichi UI 实现光源、弹射次数和采样数实时交互。
- 额外完成折射/全反射和 MSAA 抗锯齿扩展。

该程序展示了光线追踪算法从数学公式到 GPU 并行实现的完整过程，是理解现代路径追踪、全局光照和实时渲染技术的重要基础。
