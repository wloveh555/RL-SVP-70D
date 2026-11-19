"""
60维 极速冲榜版 V3 (Ultimate-V3) 验证脚本 (带 Sympy 精确反推防伪)

模型: best_ppo_60d_ultimate_v3.pth
目标: 真实统计突破 LLL、逼近 BKZ、突破 BKZ 的比例。

用法:
  python evaluate_60d_ultimate_v3.py --max-lattices 50 --samples 2 --with-bkz
"""
import os, glob, math, random, time, argparse
import numpy as np
import sympy as sp

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from fpylll import IntegerMatrix, LLL, BKZ
except ImportError:
    raise ImportError("请安装 fpylll")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASIS_NUM_60 = 60

FILE_PATTERN = "random_lattice_60_*"

# ★ 必须与 V-ULTIMATE-V3 训练配置完全一致
START_LLL_DELTA = 0.7  # 验证时统一用 0.7，保证公平
K_SELECT = 52  # ★ V3 关键配置：52
LLL_FALLBACK_INTERVAL = 80  # ★ V3 关键配置：80
MAX_STEPS_EPISODE = 350  # ★ V3 关键配置：350
RESET_UNIMODULAR_MAX_VAL = 3

STATE_DIM = BASIS_NUM_60 + BASIS_NUM_60 * (BASIS_NUM_60 - 1) // 2 + 2
HIDDEN_DIM = 768
MODEL_PATH = "best_ppo_60d_ultimate_v3.pth"  # ★ V3 模型路径


# ===================== 网络结构 =====================
class ActorCritic60(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(STATE_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.selector = nn.Linear(HIDDEN_DIM, BASIS_NUM_60)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        x = torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0), -10.0, 10.0)
        h = self.fc2(self.fc1(x))
        return torch.nan_to_num(self.selector(h), nan=0.0), torch.nan_to_num(self.critic(h).squeeze(-1), nan=0.0)


# ===================== 工具函数 =====================
def norm_sq(row): return sum(int(x) * int(x) for x in row)


def min_norm_sq(basis): return min(norm_sq(row) for row in basis)


def has_zero_row(basis): return any(all(int(x) == 0 for x in row) for row in basis)


def sqrt_float_from_sq(nsq): return 0.0 if nsq <= 0 else math.exp(0.5 * math.log(nsq))


def load_lattice_60d(file_name):
    fp = os.path.join(LATTICE_DIR, file_name)
    if not os.path.exists(fp):
        for ext in ['.txt', '']:
            cand = os.path.join(LATTICE_DIR, file_name + ext)
            if os.path.exists(cand): fp = cand; break
        else:
            raise FileNotFoundError(f"未找到 {file_name}")
    with open(fp, 'r') as f:
        content = f.read().strip().replace('[', ' ').replace(']', ' ').replace('\n', ' ')
    nums = content.split()
    n = int(round(math.sqrt(len(nums))))
    return [[int(nums[i * n + j]) for j in range(n)] for i in range(n)]


def lll_reduction_safe(basis, delta=0.99):
    n, m = len(basis), len(basis[0])
    B_mat = IntegerMatrix.from_matrix([[int(x) for x in row] for row in basis])
    try:
        result = LLL.reduction(B_mat, delta=delta)
        B_out = result[0] if isinstance(result, tuple) else result
        new_basis = [[int(B_out[i, j]) for j in range(m)] for i in range(n)]
        return basis if has_zero_row(new_basis) else new_basis
    except Exception:
        return basis


def enumerate_sub_lattice(basis_60d, selected_rows):
    k = len(selected_rows)
    if k < 2: return None, -1
    try:
        sub_basis = [[int(x) for x in basis_60d[r]] for r in selected_rows]
        B = IntegerMatrix.from_matrix(sub_basis)
        par = BKZ.Param(block_size=k, strategies=BKZ.DEFAULT_STRATEGY, max_loops=1)
        BKZ.reduction(B, par)
        best_sq, best_v = float('inf'), None
        for i in range(k):
            row = [int(B[i, j]) for j in range(BASIS_NUM_60)]
            sq = sum(x * x for x in row)
            if 0 < sq < best_sq: best_sq, best_v = sq, row
        return (best_v, best_sq) if best_v else (None, -1)
    except Exception:
        return None, -1


def compute_state(basis, episode_start_sq, step, max_steps):
    n = BASIS_NUM_60
    norm_sqs = [max(norm_sq(row), 1) for row in basis]
    log_lengths = [0.5 * math.log(nsq) for nsq in norm_sqs]
    cos_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            dot = sum(int(basis[i][k]) * int(basis[j][k]) for k in range(n))
            if dot == 0: cos_vals.append(0.0); continue
            log_cos_sq = 2 * math.log(abs(dot)) - math.log(norm_sqs[i]) - math.log(norm_sqs[j])
            cos_vals.append(min(abs(math.exp(0.5 * min(0.0, log_cos_sq))), 1.0))
    cur_min = min_norm_sq(basis)
    start_ratio = math.log(max(episode_start_sq, 1)) - math.log(
        max(cur_min, 1)) if episode_start_sq > 0 and cur_min > 0 else 0.0
    state = np.concatenate([np.array(log_lengths, dtype=np.float32), np.array(cos_vals, dtype=np.float32),
                            np.array([start_ratio, step / max(max_steps, 1)], dtype=np.float32)])
    return np.clip(np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0).astype(np.float32)


# ===================== Sympy 精确反推 =====================
def verify_lattice_vector(orig_basis, v_min):
    B = sp.Matrix(orig_basis)
    v = sp.Matrix([int(x) for x in v_min])
    try:
        c = B.T.solve(v)
        for x in c:
            if not x.is_Integer: return False, "非整数系数(伪向量)"
        if B.T * c != v: return False, "矩阵重建失败"
        return True, "真实格点"
    except Exception as e:
        return False, f"异常:{str(e)[:15]}"


# ===================== 单格评估 =====================
def eval_one(model, info, samples, deterministic, seed):
    best_sq, best_start, best_v = None, None, None
    zero_inc = 0
    with torch.no_grad():
        for s in range(samples):
            rng = random.Random(seed + s)
            basis = lll_reduction_safe(info['basis'], delta=START_LLL_DELTA)
            n = BASIS_NUM_60
            U = [[0] * n for _ in range(n)]
            for i in range(n): U[i][i] = 1
            for i in range(1, n):
                for j in range(i): U[i][j] = rng.randint(-RESET_UNIMODULAR_MAX_VAL, RESET_UNIMODULAR_MAX_VAL + 1)
            new_basis = [[sum(int(U[i][k]) * int(basis[k][j]) for k in range(n)) for j in range(n)] for i in range(n)]
            if not has_zero_row(new_basis): basis = new_basis

            ep_start_sq = min_norm_sq(basis)
            best_ep_sq = ep_start_sq
            best_ep_v = basis[np.argmin([norm_sq(r) for r in basis])][:]

            for step in range(MAX_STEPS_EPISODE):
                state = compute_state(basis, ep_start_sq, step, MAX_STEPS_EPISODE)
                st_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

                logits, _ = model(st_tensor)
                if deterministic:
                    start_idx = logits.argmax(dim=-1).item()
                else:
                    start_idx = Categorical(logits=logits).sample().item()

                sel = [(start_idx + k) % n for k in range(K_SELECT)]

                prev_b = [r[:] for r in basis]
                v, v_sq = enumerate_sub_lattice(basis, sel)
                if v is not None and v_sq > 0 and v_sq < best_ep_sq:
                    basis[np.argmax([norm_sq(r) for r in basis])] = v
                    if has_zero_row(basis): basis = prev_b; zero_inc += 1

                if LLL_FALLBACK_INTERVAL > 0 and (step + 1) % LLL_FALLBACK_INTERVAL == 0:
                    new_b = lll_reduction_safe(basis, delta=0.99)
                    if not has_zero_row(new_b): basis = new_b

                if has_zero_row(basis): basis = prev_b; zero_inc += 1

                new_sq = min_norm_sq(basis)
                if new_sq < best_ep_sq:
                    best_ep_sq = new_sq
                    best_ep_v = basis[np.argmin([norm_sq(r) for r in basis])][:]

            if best_sq is None or best_ep_sq < best_sq:
                best_sq, best_start, best_v = best_ep_sq, ep_start_sq, best_ep_v

    return {'final_sq': best_sq, 'start_sq': best_start, 'zero': zero_inc, 'best_v': best_v}


# ===================== 主程序 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lattices", type=int, default=50)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--with-bkz", action="store_true", help="是否使用 BKZ(40) 作为目标参考")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型: {MODEL_PATH}")
    model = ActorCritic60().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✓ 加载模型: {MODEL_PATH} (K={K_SELECT}, EPISODE={MAX_STEPS_EPISODE})")

    files = sorted(glob.glob(os.path.join(LATTICE_DIR, FILE_PATTERN)))
    if not files: raise FileNotFoundError(f"未找到格文件: {LATTICE_DIR}")
    if args.max_lattices > 0: files = files[:args.max_lattices]
    print(f"\n从 {LATTICE_DIR} 加载 {len(files)} 个格文件...")

    infos = []
    for i, fp in enumerate(files):
        try:
            basis = load_lattice_60d(os.path.basename(fp))
            lll_min_sq = min_norm_sq(lll_reduction_safe(basis, delta=0.99))

            bkz_min_sq = lll_min_sq
            if args.with_bkz:
                try:
                    B_mat = IntegerMatrix.from_matrix([[int(x) for x in r] for r in basis])
                    par = BKZ.Param(block_size=40, strategies=BKZ.DEFAULT_STRATEGY, max_loops=1)
                    BKZ.reduction(B_mat, par)
                    bkz_min_sq = min_norm_sq([[int(B_mat[i, j]) for j in range(60)] for i in range(60)])
                except:
                    pass

            infos.append({'file': os.path.basename(fp), 'basis': basis, 'lll_min_sq': int(lll_min_sq),
                          'bkz_min_sq': int(bkz_min_sq)})
            if (i + 1) % 10 == 0:
                print(
                    f"  [{i + 1}/{len(files)}] {os.path.basename(fp)}: LLL={sqrt_float_from_sq(lll_min_sq):.1f}, BKZ40={sqrt_float_from_sq(bkz_min_sq):.1f}")
        except Exception as e:
            print(f"  跳过 {fp}: {e}")

    random.seed(args.seed);
    np.random.seed(args.seed);
    torch.manual_seed(args.seed)
    print(f"\n开始评估 {len(infos)} 个格 (包含 Sympy 精确反推验证)...\n")
    print("=" * 145)
    print(
        f"{'#':>3} | {'LLL':>9} | {'BKZ40':>9} | {'PPO_Final':>10} | {'vs LLL':>8} | {'vs BKZ':>8} | {'改进%':>7} | {'零向量':>5} | {'突破?':>5} | {'Sympy 验证'}")
    print("=" * 145)

    t0 = time.time()
    all_results = []
    total_zero = 0
    valid_count = 0

    for idx, info in enumerate(infos):
        r = eval_one(model, info, args.samples, args.deterministic, args.seed)
        r['lll_sq'] = info['lll_min_sq']
        r['bkz_sq'] = info['bkz_min_sq']
        all_results.append(r)
        total_zero += r['zero']

        is_valid, msg = verify_lattice_vector(info['basis'], r['best_v'])
        if is_valid: valid_count += 1

        start = sqrt_float_from_sq(r['start_sq'])
        final = sqrt_float_from_sq(r['final_sq'])
        lll = sqrt_float_from_sq(r['lll_sq'])
        bkz = sqrt_float_from_sq(r['bkz_sq'])

        improve = (start - final) / start * 100 if start > 0 else 0
        vs_lll = (final - lll) / lll * 100 if lll > 0 else 0
        vs_bkz = (final - bkz) / bkz * 100 if bkz > 0 else 0

        broke_lll = "✓" if r['final_sq'] < r['lll_sq'] else "✗"

        print(
            f"{idx + 1:>3} | {lll:>9.2f} | {bkz:>9.2f} | {final:>10.2f} | {vs_lll:>7.2f}% | {vs_bkz:>7.2f}% | {improve:>6.2f}% | {r['zero']:>5} | {broke_lll:>5} | {msg}")

        if (idx + 1) % 10 == 0:
            print(f"  ... 进度 {idx + 1}/{len(infos)}, 耗时 {time.time() - t0:.1f}s")

    print("=" * 145)
    n = len(all_results)

    broke_lll = sum(1 for r in all_results if r['final_sq'] < r['lll_sq'])
    reached_bkz = sum(1 for r in all_results if r['final_sq'] <= r['bkz_sq'])
    close_to_bkz = sum(1 for r in all_results if r['final_sq'] <= r['bkz_sq'] * 1.15)

    imp = [(sqrt_float_from_sq(r['start_sq']) - sqrt_float_from_sq(r['final_sq'])) / sqrt_float_from_sq(
        r['start_sq']) * 100 for r in all_results]

    print(f"\n===== V-ULTIMATE-V3 评估总结 (60维 K=52, Episode=350) =====")
    print(f"评估格数:                       {n}")
    print(f"突破 LLL 比例:                  {broke_lll}/{n} ({broke_lll / n * 100:.2f}%)")
    print(f"逼近 BKZ40 (误差<15%) 比例:     {close_to_bkz}/{n} ({close_to_bkz / n * 100:.2f}%)")
    print(f"突破/达到 BKZ40 比例:           {reached_bkz}/{n} ({reached_bkz / n * 100:.2f}%)")
    print(f"平均改进率 (相对起点):           {np.mean(imp):.2f}%")
    print(f"Sympy 验证通过率:               {valid_count}/{n} ({valid_count / n * 100:.2f}%)")
    print(f"零向量拦截总数:                 {total_zero}")


if __name__ == "__main__":
    main()