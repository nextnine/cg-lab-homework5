import taichi as ti

# 初始化 Taichi GPU 后端 (Mac 自动调用 Metal，Win 调用 CUDA/Vulkan)
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
samples_per_pixel = ti.field(ti.i32, shape=())


# 材质常量枚举
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

EPS = 1e-4
GLASS_IOR = 1.5  # 玻璃折射率，常用近似值为 1.5

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, ior):
    """
    根据斯涅尔定律计算折射方向。

    I: 当前入射光线方向，必须是单位向量
    N: 表面外法线，必须是单位向量
    ior: 材质折射率，例如玻璃约为 1.5

    返回:
    refr_dir: 折射方向；如果发生全反射，则这个值没有意义
    tir: 是否发生全反射，1 表示发生全反射，0 表示正常折射
    """
    refr_dir = ti.Vector([0.0, 0.0, 0.0])
    tir = 0

    # cosi = dot(I, N)
    # 如果 cosi < 0，说明光线从空气进入玻璃
    # 如果 cosi > 0，说明光线从玻璃内部射向空气
    cosi = ti.max(-1.0, ti.min(1.0, I.dot(N)))

    etai = 1.0
    etat = ior
    n = N

    if cosi < 0.0:
        # 从空气进入玻璃
        cosi = -cosi
    else:
        # 从玻璃内部进入空气
        # 交换入射介质和出射介质折射率
        temp = etai
        etai = etat
        etat = temp

        # 法线要反过来，因为此时光线在球体内部
        n = -N

    eta = etai / etat

    # k < 0 表示根号内部为负，物理意义是发生全反射
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)

    if k < 0.0:
        tir = 1
    else:
        refr_dir = normalize(eta * I + (eta * cosi - ti.sqrt(k)) * n)

    return refr_dir, tir


@ti.func
def intersect_sphere(ro, rd, center, radius):
    """球体求交，返回 (距离 t, 法线 normal)"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0.0:
        sqrt_delta = ti.sqrt(delta)

        t1 = (-b - sqrt_delta) / 2.0
        t2 = (-b + sqrt_delta) / 2.0

        if t1 > EPS:
            t = t1
        elif t2 > EPS:
            t = t2

        if t > 0.0:
            p = ro + rd * t
            normal = normalize(p - center)

    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    """水平无限大平面求交"""
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0]) # 法线永远朝上
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    """
    遍历场景，寻找最近交点。
    返回: (t, 法线 N, 颜色 color, 材质 mat_id)
    """
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 检测红色漫反射球
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.8, 0.1, 0.1])
        hit_mat = MAT_GLASS #MAT_DIFFUSE漫反射，这个是玻璃

    # 2. 检测银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9]) # 镜面反射基础色
        hit_mat = MAT_MIRROR

    # 3. 检测地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        # 生成棋盘格纹理
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        # 判断坐标的奇偶性来交替颜色
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3]) # 灰色格子
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8]) # 白色格子

    return min_t, hit_n, hit_c, hit_mat

@ti.func
def shade_diffuse(p, N, obj_color, light_pos):
    """
    漫反射材质着色，包括环境光、漫反射和硬阴影。
    """
    direct_light = ti.Vector([0.0, 0.0, 0.0])

    # 环境光
    ambient = 0.2 * obj_color
    direct_light += ambient

    L = normalize(light_pos - p)
    dist_to_light = (light_pos - p).norm()

    # 发射暗影射线，注意加 EPS 偏移，防止 Shadow Acne
    shadow_ray_orig = p + N * EPS
    shadow_t, shadow_n, shadow_c, shadow_mat = scene_intersect(shadow_ray_orig, L)

    in_shadow = 0.0
    if shadow_t < dist_to_light:
        in_shadow = 1.0

    if in_shadow == 0.0:
        diff = ti.max(0.0, N.dot(L))
        diffuse = 0.8 * diff * obj_color
        direct_light += diffuse

    return direct_light


@ti.func
def trace_ray(ro, rd, light_pos):
    """
    单条光线的迭代式 Whitted-Style 光线追踪。

    支持：
    1. 漫反射
    2. 镜面反射
    3. 玻璃折射
    4. 全反射
    """
    bg_color = ti.Vector([0.05, 0.15, 0.2])

    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])

    alive = 1

    for bounce in range(max_bounces[None]):
        if alive == 1:
            t, N, obj_color, mat_id = scene_intersect(ro, rd)

            # 没打到任何物体，看到背景色
            if t > 1e9:
                final_color += throughput * bg_color
                alive = 0

            else:
                p = ro + rd * t

                # 分支 1：镜面材质
                if mat_id == MAT_MIRROR:
                    new_rd = normalize(reflect(rd, N))

                    # 反射射线起点沿法线偏移
                    ro = p + N * EPS
                    rd = new_rd

                    # 镜面反射会吸收一部分能量
                    throughput *= 0.8 * obj_color

                # 分支 2：玻璃材质
                elif mat_id == MAT_GLASS:
                    refr_dir, tir = refract(rd, N, GLASS_IOR)

                    if tir == 1:
                        # 全反射：折射不存在，改走反射方向
                        new_rd = normalize(reflect(rd, N))

                        # 全反射发生在玻璃内部时，沿反射方向偏移更稳
                        ro = p + new_rd * EPS
                        rd = new_rd

                        # 全反射基本不损失能量，这里略微衰减，避免过亮
                        throughput *= 0.98

                    else:
                        # 正常折射：光线穿过玻璃
                        new_rd = refr_dir

                        # 对折射光线，用新方向偏移，避免刚折进去/折出来又撞到同一表面
                        ro = p + new_rd * EPS
                        rd = new_rd

                        # 玻璃吸收一点颜色。
                        # obj_color 带一点淡红色，让玻璃球有轻微红色透射感。
                        throughput *= 0.96 * obj_color

                # 分支 3：漫反射材质
                elif mat_id == MAT_DIFFUSE:
                    direct_light = shade_diffuse(p, N, obj_color, light_pos)

                    final_color += throughput * direct_light

                    # Whitted 风格下，漫反射材质终止该条光线
                    alive = 0

    # 如果达到最大弹射次数后光线仍然活着，给一点背景色作为收尾
    # 这样镜面/玻璃在弹射次数太小时不会完全变黑
    if alive == 1:
        final_color += throughput * bg_color

    return final_color


@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])

    for i, j in pixels:
        color = ti.Vector([0.0, 0.0, 0.0])

        # MSAA：每个像素内发射多条带随机扰动的主光线，然后取平均
        for s in range(samples_per_pixel[None]):
            # 在像素内部随机采样
            jitter_x = ti.random(ti.f32)
            jitter_y = ti.random(ti.f32)

            u = ((ti.cast(i, ti.f32) + jitter_x) - res_x / 2.0) / res_y * 2.0
            v = ((ti.cast(j, ti.f32) + jitter_y) - res_y / 2.0) / res_y * 2.0

            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            color += trace_ray(ro, rd, light_pos)

        color /= ti.cast(samples_per_pixel[None], ti.f32)

        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化光源位置和弹射次数
    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 3
    samples_per_pixel[None] = 4

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.22):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 5)
            #MSAA
            samples_per_pixel[None] = gui.slider_int("Samples / Pixel", samples_per_pixel[None], 1, 64)


        window.show()

if __name__ == '__main__':
    main()