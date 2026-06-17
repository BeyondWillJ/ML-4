import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)
import pandas as pd
import numpy as np
import copy
from itertools import product

# 依赖自定义库函数
from SFlib_I import *  # noqa: F401,F403
from SFlib_II import *  # noqa: F401,F403

# 数据文件路径
DATA_PATH = r'data\SiHits_3D_pvar_zvar_0.03_0.50_28_0.04_0.06_2000_v1.txt'

# 预加载数据
try:
    DF_ALL = pd.read_csv(DATA_PATH, header=None)
except Exception as e:
    print(f"读取数据文件失败: {e}")
    DF_ALL = None


def build_hits_for_event(event_id: int):
    """读取指定 event 的所有 hit 并附加极坐标(r, phi)与 rz theta.
    返回: a_s(list), realz0(float)
    原 a_s[i] 结构 (参考原脚本追加后的索引):
    [0event,1track,2px,3py,4pz,5x,6y,7z, (后面追加的) 8r,9phi, 10?,11?,12theta_rz]
    实际根据 c_to_p 返回值长度使用; 这里只保持与原脚本兼容。
    """
    eventi = DF_ALL[DF_ALL.iloc[:, 0] == event_id]
    if len(eventi) == 0:
        return [], None
    a_s = []
    for i in range(len(eventi)):
        row_list = eventi.iloc[i, :].tolist()
        # c_to_p 返回 (r, phi) (假设)
        a_s.append(row_list + c_to_p(row_list[7], row_list[8]))
    realz0 = a_s[0][1]
    # 追加 theta_rz （基于 z-realz0 与 r）
    for i in range(len(a_s)):
        a_s[i].append(c_to_p2(a_s[i][9] - realz0, a_s[i][10])[1])  # 索引 12
    return a_s, realz0


def cluster_points(a_s):
    """按照原脚本逻辑在 theta_rz 维度做简单差分聚类, 返回 ct_s (list of clusters)."""
    if not a_s:
        return []
    # 按 theta 排序
    a_s_sorted = sorted(a_s, key=lambda x: x[12])
    deltas = []
    for i in range(len(a_s_sorted) - 1):
        deltas.append(a_s_sorted[i + 1][12] - a_s_sorted[i][12])
    clusters = []
    flag = 0
    temp = []
    for i, d in enumerate(deltas):
        if d < 0.01:
            flag += 1
            temp.append(a_s_sorted[i])
        else:
            if flag != 0:
                flag = 0
                temp.append(a_s_sorted[i])
                if len(temp) > 3:
                    clusters.append(temp)
                temp = []
            else:
                pass
    # 处理末尾残留
    if flag != 0 and len(temp) > 3:
        temp.append(a_s_sorted[-1])
        clusters.append(temp)
    return clusters


def fit_circles_for_cluster(cluster, realz0):
    """按照原脚本对单个 cluster 进行枚举拟合, 返回 ct_II(list) 与 circles(list)."""
    ct_II = []
    circles = []
    # 分 track id
    track_groups = [[] for _ in range(5)]  # 假设 track id 在 0..4
    for hit in cluster:
        try:
            track_groups[int(hit[3])].append(copy.deepcopy(hit))
        except Exception:
            continue
    lengths = [len(g) for g in track_groups]
    # 顶点虚拟点
    ver = copy.deepcopy(cluster[-1])
    ver[7:13] = [0, 0, realz0, 0, 0, 0]

    # 生成笛卡尔积 (若有长度为0的组则结果为空)
    try:
        index_ranges = [range(n) for n in lengths]
        all_combos = list(product(*index_ranges))
    except Exception:
        all_combos = []

    for combo in all_combos:
        if len(combo) == 0:
            continue
        selected = [track_groups[k][combo[k]] for k in range(len(combo)) if lengths[k] > 0]
        if len(selected) < 3:  # 圆拟合至少3点
            continue
        try:
            (xc, yc), rc = FitCircle_simp([p[7:9] for p in selected], False)
        except Exception:
            continue
        # 残差
        sum_residual = 0.0
        for p in selected:
            try:
                sum_residual += abs(distance(p[7:9], (xc, yc)) - rc)
            except Exception:
                pass
        if sum_residual <= 0.1:
            try:
                pltf = PolarTransform((xc, yc), (0, 0))
                a_or_b = [0, 0]
                for p in selected:
                    r_t, theta_t = pltf.to_polar(p[7:9])
                    if 0 <= theta_t <= np.pi:
                        a_or_b[0] += 1
                    else:
                        a_or_b[1] += 1
                if a_or_b[0] == 0 or a_or_b[1] == 0:  # 单侧约束
                    circles.append([xc, yc, rc])
                    ct_II.append([[copy.deepcopy(ver)] + copy.deepcopy(selected), [xc, yc, rc]])
            except Exception:
                continue
    return ct_II, circles


def prepare_event(event_id):
    a_s, realz0 = build_hits_for_event(event_id)
    if not a_s:
        return {
            'a_s': [], 'realz0': None, 'zp': None, 'ct_s': []
        }
    # 顶点估计
    zp_s = Findzp(a_s)
    try:
        zp = SlideWindow(zp_s, 0.001)[0]
    except Exception:
        zp = None
    ct_s = cluster_points(a_s)
    return {
        'a_s': a_s,
        'realz0': realz0,
        'zp': zp,
        'ct_s': ct_s,
    }


class FindTrackUI:
    def __init__(self, master):
        self.master = master
        master.title("FindTrack UI")

        self.event_var = tk.StringVar(value="3")
        # 删除 cluster index 选择，改为整 event 绘制

        frm_inputs = ttk.Frame(master)
        frm_inputs.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        ttk.Label(frm_inputs, text="Event:").grid(row=0, column=0, sticky=tk.W, padx=2)
        ttk.Entry(frm_inputs, textvariable=self.event_var, width=8).grid(row=0, column=1, padx=2)
        # 按钮
        ttk.Button(frm_inputs, text="Load Event", command=self.load_event).grid(row=0, column=2, padx=10)
        ttk.Button(frm_inputs, text="Draw Event", command=self.plot_event).grid(row=0, column=3, padx=4)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(master, textvariable=self.status_var, anchor='w').pack(fill=tk.X, padx=8, pady=2)

        # Matplotlib Figure
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 数据缓存
        self.current_event_id = None
        self.event_data = None

    def set_status(self, msg):
        self.status_var.set(msg)
        self.master.update_idletasks()

    def load_event(self):
        try:
            event_id = int(self.event_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "Event ID must be an integer.")
            return
        if DF_ALL is None:
            messagebox.showerror("Data Error", "Data file failed to load.")
            return
        # range check
        try:
            max_event_id = int(DF_ALL.iloc[:, 0].max())
        except Exception:
            max_event_id = None
        if max_event_id is not None and event_id > max_event_id:
            messagebox.showerror("Out of Range", f"Event ID {event_id} exceeds maximum event ID {max_event_id}.")
            return
        self.set_status(f"Loading event {event_id} ...")
        self.event_data = prepare_event(event_id)
        self.current_event_id = event_id
        ct_len = len(self.event_data['ct_s'])
        self.set_status(f"Event {event_id} loaded, clusters: {ct_len}")
        messagebox.showinfo("Done", f"Event {event_id} loaded.\nClusters: {ct_len}")

    def plot_event(self):
        """绘制当前 event 所有 cluster 及其拟合结果。"""
        if not self.event_data:
            messagebox.showwarning("Info", "Please load an Event first.")
            return
        ct_s = self.event_data['ct_s']
        realz0 = self.event_data['realz0']
        zp = self.event_data['zp']
        a_s = self.event_data['a_s']
        if not ct_s:
            messagebox.showinfo("Info", "No available clusters for this Event.")
            return

        self.set_status("开始拟合所有 cluster ...")
        all_ctII = []  # 保存所有拟合结果

        # 清理绘图区
        self.ax3d.clear()
        # 顶点
        if realz0 is not None:
            self.ax3d.scatter(0, 0, realz0, c='r', marker='o')
        # 全部点统一蓝色（与原脚本一致）
        if a_s:
            xs_all = [p[7] for p in a_s]
            ys_all = [p[8] for p in a_s]
            zs_all = [p[9] for p in a_s]
            self.ax3d.scatter(xs_all, ys_all, zs_all, c='b', s=10, alpha=0.8)

        # 逐 cluster 拟合并绘制轨迹
        for cluster in ct_s:
            try:
                ct_II, circles = fit_circles_for_cluster(cluster, realz0)
            except Exception:
                ct_II = []
            all_ctII.extend(ct_II)
            for item in ct_II:
                pts = item[0]
                (xc, yc, rc) = item[1]
                color = random_color()
                try:
                    DrawEllipse([xc, yc], copy.deepcopy(pts), rc, ax=self.ax3d, color=color)
                except Exception:
                    theta = np.linspace(0, 2*np.pi, 120)
                    z_plane = realz0 if realz0 is not None else 0
                    self.ax3d.plot(xc + rc*np.cos(theta), yc + rc*np.sin(theta), [z_plane]*len(theta), color=color, linewidth=1)

        realz0_str = f"{realz0:.3f}" if realz0 is not None else 'NA'
        zp_str = f"{zp:.3f}" if isinstance(zp, (int, float)) else (zp if zp is not None else 'NA')
        self.ax3d.set_title(
            f"Event {self.current_event_id} | clusters={len(ct_s)} | fitted_tracks={len(all_ctII)}\n"
            f"z_real={realz0_str} z_pred={zp_str}"
        )
        self.ax3d.set_xlabel('x')
        self.ax3d.set_ylabel('y')
        self.ax3d.set_zlabel('z')
        try:
            self.ax3d.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        self.canvas.draw()
        self.set_status(f"绘制完成: clusters={len(ct_s)}, 拟合轨迹总数 {len(all_ctII)}")


def main():
    root = tk.Tk()
    app = FindTrackUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
