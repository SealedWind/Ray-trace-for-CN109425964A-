import math

# ================= 数据：与 raytrace2.py / 专利表一、表二 一致 =================
# S[i] = (曲率 c, 厚度 t, 折射率 n_after)
S = [
    (0.7209805335255948, 0.590, 1.545),   # 2  L1前
    (0.14999250037498124, 0.190, 1.0),    # 3  L1后
    (-0.1145081873353945, 0.270, 1.681),  # 4  L2前
    (0.14690759512266782, 0.183, 1.0),    # 5  L2后
    (0.0, 0.169, 1.0),                    # 6  光阑
    (0.013415795757925382, 0.446, 1.65),  # 7  L3前
    (-0.09276437847866420, 0.653, 1.0),   # 8  L3后
    (-0.22104332449160035, 0.323, 1.584), # 9  L4前
    (-0.32573289902280133, 0.971, 1.0),   # 10 L4后
    (-0.49333991119881593, 0.320, 1.544), # 11 L5前
    (0.0, 0.300, 1.0),                    # 12 L5后
    (0.0, 0.175, 1.0),                    # 13 IRCF前
    (0.0, 0.339, 1.0),                    # 14 IRCF后
]
K = [0.69029, 6.651, 89.897, -13.238, 0.0, 90.0, -19.806, 3.9617, -6.8195, -1.5023, 0.0, 0.0, 0.0]
A = [
    [-3.9972e-2, 1.6028e-1, -1.1676, 4.611, -11.463, 17.876, -17.028, 9.0525, -2.0636],
    [-3.8756e-4, 8.7393e-2, -5.3386e-1, 3.0822, -10.539, 22.483, -29.044, 20.849, -6.3854],
    [7.6386e-2, -1.5911e-1, 2.0266, -10.033, 29.63, -53.903, 58.937, -35.396, 8.9104],
    [5.3243e-2, 5.5133e-1, -4.1735, 23.581, -82.593, 180.5, -238.77, 174.95, -54.518],
    [0.0]*9,
    [-1.0524e-1, -8.0546e-3, 4.5157e-1, -1.1287, 2.0514, -2.6113, 2.3993, -1.2812, 2.6854e-1],
    [-1.2209e-1, 6.3303e-2, -4.8264e-1, 1.7394, -3.6039, 4.6651, -3.6964, 1.655, -3.1784e-1],
    [5.9306e-4, -5.1750e-2, -3.9840e-1, 1.058, -1.5813, 1.4349, -0.7539, 0.21162, -2.4592e-2],
    [6.9263e-2, -1.1475e-1, 2.5594e-2, 1.8698e-2, -2.3712e-2, 2.2972e-2, -1.2610e-2, 3.2828e-3, -3.2342e-4],
    [8.0346e-2, -5.9756e-2, 2.1205e-2, -2.7673e-3, -9.3139e-4, 5.6769e-4, -1.2639e-4, 1.3463e-5, -5.6728e-7],
    [-3.0997e-3, -8.1723e-3, -1.8845e-3, 3.2523e-3, -1.5173e-3, 3.8461e-4, -5.7327e-5, 4.7637e-6, -1.6995e-7],
    [0.0]*9, [0.0]*9,
]
POW = [4, 6, 8, 10, 12, 14, 16, 18, 20]

EFL = 4.975      # 近轴有效焦距（表一 EFFL）
R_STOP = 0.9567  # 光阑半口径 = EPD/2
STOP_Z = -0.392  # 前置光圈到 L1 前顶点的距离

# 矢高公式 (Sag Formula)
def sag(i, y):
    """第 i 面在高度 y 处的矢高 z；超出圆锥定义域返回 None"""
    if abs(y) > 3.0:            # 保护：迭代发散时 y 会爆炸，直接判失败
        return None
    c = S[i][0]
    k = K[i]
    arg = 1.0 - (1.0 + k) * c * c * y * y
    if arg < 0:
        return None
    z = c * y * y / (1.0 + math.sqrt(arg)) if c else 0.0
    for a, p in zip(A[i], POW):
        z += a * y ** p
    return z

# 中心差分法（Central Difference）求数值导数，k=[f(y+h)-f(y-h)]/(2h)
def slope(i, y):
    h = 1e-6 # h 足够小，k 就足够精确
    a, b = sag(i, y - h), sag(i, y + h)
    if a is None or b is None:
        return None
    return (b - a) / (2 * h)


def trace(pupil, field_deg):
    """追一条光线。pupil = 光阑平面上的高度；field_deg = 视场角（度）。
    返回像面高度 y_img；追不到返回 None。"""
    th = math.radians(field_deg)
    uz, uy = math.cos(th), math.sin(th)   # 斜入射方向（唯一相对轴上版改动的地方）
    z = STOP_Z
    y = pupil
    n = 1.0
    Z = 0.0
    for i in range(len(S)):
        # 推进到第 i 面的顶点平面
        t = (Z - z) / uz
        z, y = z + t * uz, y + t * uy
        # 固定点迭代求交点：y = y_entry + tan(u)·sag(y)
        # 取当前光线的位置高度 y 作为初值
        yy = y
        for _ in range(50):
            s = sag(i, yy)  # 用非球面矢高公式计算曲面上对应的 z
            if s is None:
                return None
            newy = y + s * (uy / uz) # 根据曲面上的 z ,初始 y 移动相同比例得到new y
            # if abs(newy) > 3.0:      # 迭代发散（光线跑出面外），判失败
            #     return None
            # 计算 y 的误差，然后开始下一轮迭代(交点需同时满足直线与非球面公式，两个y误差缩小到一定程度就视为同一个点)
            if abs(newy - yy) < 1e-12:
                yy = newy
                break
            yy = newy
        s = sag(i, yy)
        if s is None:
            return None
        m = slope(i, yy)  # 交点处的切线斜率
        if m is None or not math.isfinite(m):
            return None
        # snell
        nr = math.hypot(m, 1.0) # 交点处的法线向量的长度（模）
        Ny, Nz = m / nr, -1.0 / nr # 转为单位法向量 (Ny, Nz)
        ci = -(uy * Ny + uz * Nz) # 入射角的余弦值 = 入射光线的方向向量·表面法线的方向向量
        if ci < 0:
            Ny, Nz = -Ny, -Nz
            ci = -(uy * Ny + uz * Nz)
        mu = n / S[i][2] # 相对折射率=当前折射率/下一个面的折射率
        st2 = mu * mu * (1.0 - ci * ci)
        if st2 > 1.0:
            return None
        ct = math.sqrt(1.0 - st2)
        uy = mu * uy + (mu * ci - ct) * Ny
        uz = mu * uz + (mu * ci - ct) * Nz
        n = S[i][2]
        z, y = Z + s, yy
        Z = Z + S[i][1]
    t = (Z - z) / uz
    return y + t * uy

# field_summary(fld) 内部会扫描整个光阑口径（pupil 从 −0.9567 到 +0.9567）追 21 条光线，每条都以此角度入射。
def field_summary(field_deg, nsamp=21):
    """返回 (像高, 畸变%, 子午RMS µm, 子午全宽 µm, 有效光线数)"""
    ys = []
    for k in range(nsamp):
        p = R_STOP * (2.0 * k / (nsamp - 1) - 1.0)   # 从 -R_STOP 到 +R_STOP
        v = trace(p, field_deg)
        if v is not None:
            ys.append(v)
    if not ys:
        return None
    mean = sum(ys) / len(ys)
    rms = math.sqrt(sum((v - mean) ** 2 for v in ys) / len(ys))
    width = max(ys) - min(ys)
    chief = trace(0.0, field_deg)
    ideal = EFL * math.tan(math.radians(field_deg))
    dist = (chief - ideal) / ideal * 100 if ideal else 0.0
    n_fail = nsamp - len(ys)
    return chief, dist, rms * 1000, width * 1000, len(ys), n_fail


if __name__ == "__main__":
    # 仅计算给定光阑高度的光线的像点高度（点球差）
    for p in (0.0, 0.3, 0.5, 0.7, 0.85, 0.9567):
        for sgn in (1, -1):
            yy = trace(sgn * p, 0)
            print("pupil %+0.4f -> y_img = %s" % (sgn * p, "fail" if yy is None else "%+.6f" % yy))
    # 计算畸变、子午弥散随视场角的变化
    print("视场 (°) | 像高 (mm) | 畸变 (%) | 子午RMS (µm) | 子午全宽 (µm) | 光线数/失败")
    print("-------|-----------|-----------|--------------|---------------|-------------")
    for fld in (0.0, 13.3, 23.2, 28.2, 33.2):
        r = field_summary(fld)
        if r is None:
            print("%6.1f | 追迹失败" % fld)
        else:
            h, dist, rms, w, n, nf = r
            print("%6.1f | %9.4f | %+8.2f | %12.2f | %13.2f | %5d/%3d" % (fld, h, dist, rms, w, n, nf))