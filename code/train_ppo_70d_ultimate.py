"""
70维 连续窗口训练代码 V-ULTIMATE-70D (针对 5080 优化版)

核心修改：
  ★ 维度：70维，STATE_DIM = 2487，HIDDEN_DIM = 1024
  ★ 动作：K_SELECT = 55
  ★ 并行：NUM_ENVS = 8，MAX_STEPS_EPISODE = 300
  ★ GPU 优化：BATCH_SIZE 提升至 2048，让 5080 满负荷
"""
import os, glob, math, random, logging, time, argparse
import numpy as np
from multiprocessing import Pool, cpu_count
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

try:
    from fpylll import IntegerMatrix, LLL, BKZ
except ImportError:
    raise ImportError("请安装 fpylll")

# 如果显存不够，可以打开下面这行限制显存比例
# torch.cuda.set_per_process_memory_fraction(0.9, 0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

# ===================== 70维配置 =====================
BASIS_NUM_70 = 70
LATTICE_DIR = r"/mnt/d/generator/七十维"  # ★ 路径需确认
FILE_PATTERN = "random_lattice_70_*"

START_LLL_DELTAS = [0.5, 0.7, 0.8]
K_SELECT = 55  # ★ 70维设定
LLL_FALLBACK_INTERVAL = 80
MAX_STEPS_EPISODE = 300  # ★ 设定
RESET_UNIMODULAR_MAX_VAL = 3
NO_IMPROVE_STEP_LIMIT = 100

# PPO 参数 (针对 5080 优化)
PPO_EPOCHS = 10
CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR_MAX = 2e-4
LR_MIN = 1e-5
WARMUP_STEPS = 300
ENTROPY_COEF_HI = 0.08
ENTROPY_COEF_LO = 0.02
TOTAL_TRAIN_STEPS = 300_000
NUM_ENVS = 8  # ★ 设定
BATCH_SIZE = 2048  # ★ 拉爆 5080 显存，加速网络更新
HIDDEN_DIM = 1024  # ★ 70维状态扩容，网络加宽

K_USE_BKZ = True

MODEL_SAVE_PATH = "best_ppo_70d_ultimate.pth"
LOG_FILE = "ppo_training_log_70d_ultimate.txt"

# STATE_DIM = 70 + 70*69/2 + 2 = 2487
STATE_DIM = BASIS_NUM_70 + BASIS_NUM_70 * (BASIS_NUM_70 - 1) // 2 + 2

logging.basicConfig(filename=LOG_FILE, filemode='w',
                    level=logging.INFO, format='%(asctime)s - %(message)s')
console = logging.StreamHandler();
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger()


# ===================== 工具函数 =====================
def norm_sq(row): return sum(int(x) * int(x) for x in row)


def min_norm_sq(basis): return min(norm_sq(row) for row in basis)


def has_zero_row(basis): return any(all(int(x) == 0 for x in row) for row in basis)


def sqrt_float_from_sq(nsq): return 0.0 if nsq <= 0 else math.exp(0.5 * math.log(nsq))


def load_lattice_70d(file_name, lattice_dir=None):
    if lattice_dir is None: lattice_dir = LATTICE_DIR
    fp = os.path.join(lattice_dir, file_name)
    if not os.path.exists(fp):
        for ext in ['.txt', '']:
            cand = os.path.join(lattice_dir, file_name + ext)
            if os.path.exists(cand): fp = cand; break
        else:
            base = file_name.replace('.txt', '')
            all_files = glob.glob(os.path.join(lattice_dir, "*"))
            matched = [f for f in all_files if os.path.basename(f).replace('.txt', '') == base]
            if not matched: raise FileNotFoundError(f"未找到 {file_name}")
            fp = matched[0]
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


def enumerate_sub_lattice(basis_70d, selected_rows):
    k = len(selected_rows)
    if k < 2: return None, -1
    try:
        sub_basis = [[int(x) for x in basis_70d[r]] for r in selected_rows]
        B = IntegerMatrix.from_matrix(sub_basis)
        par = BKZ.Param(block_size=k, strategies=BKZ.DEFAULT_STRATEGY, max_loops=1)
        BKZ.reduction(B, par)
        best_sq, best_v = float('inf'), None
        for i in range(k):
            row = [int(B[i, j]) for j in range(BASIS_NUM_70)]
            sq = sum(x * x for x in row)
            if 0 < sq < best_sq: best_sq, best_v = sq, row
        return (best_v, best_sq) if best_v else (None, -1)
    except Exception:
        return None, -1


def compute_state(basis, episode_start_sq, step, max_steps):
    n = BASIS_NUM_70
    norm_sqs = [max(norm_sq(row), 1) for row in basis]
    log_lengths = [0.5 * math.log(nsq) for nsq in norm_sqs]
    cos_vals = []
    # ★ 微优化：使用 zip 替代 range 索引，提高纯 Python 循环速度
    for i in range(n):
        row_i = basis[i]
        nsq_i = norm_sqs[i]
        for j in range(i + 1, n):
            row_j = basis[j]
            dot = sum(a * b for a, b in zip(row_i, row_j))
            if dot == 0: cos_vals.append(0.0); continue
            log_cos_sq = 2 * math.log(abs(dot)) - math.log(nsq_i) - math.log(norm_sqs[j])
            cos_vals.append(min(abs(math.exp(0.5 * min(0.0, log_cos_sq))), 1.0))

    cur_min = min_norm_sq(basis)
    start_ratio = math.log(max(episode_start_sq, 1)) - math.log(
        max(cur_min, 1)) if episode_start_sq > 0 and cur_min > 0 else 0.0
    state = np.concatenate([np.array(log_lengths, dtype=np.float32), np.array(cos_vals, dtype=np.float32),
                            np.array([start_ratio, step / max(max_steps, 1)], dtype=np.float32)])
    return np.clip(np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0).astype(np.float32)


# ===================== 多进程 Worker =====================
def _worker_step(args):
    basis, selected_rows = args
    prev_basis = [r[:] for r in basis]
    old_min_sq = min_norm_sq(basis)
    zero_incidents = 0

    v, v_sq = enumerate_sub_lattice(basis, list(selected_rows))
    if (v is not None and v_sq > 0 and v_sq < old_min_sq and any(x != 0 for x in v)):
        lens = [norm_sq(r) for r in basis]
        idx_worst = int(np.argmax(lens))
        basis[idx_worst] = v
        if has_zero_row(basis):
            basis = prev_basis
            zero_incidents += 1

    if has_zero_row(basis):
        basis = prev_basis
        zero_incidents += 1

    return basis, zero_incidents


# ===================== 环境 =====================
class SVPEnv70:
    def __init__(self, lat_list, start_delta=0.7):
        self.lat_list = lat_list
        self.start_delta = start_delta
        self._bind_lattice(random.choice(lat_list))
        self.basis = None
        self.steps = 0
        self.no_improve = 0
        self.best_shortest_sq = 0
        self.best_basis = None
        self.episode_start_sq = 0
        self.zero_row_incidents = 0
        self.has_broken_lll = False
        self.has_reached_bkz = False

    def _bind_lattice(self, info):
        self.orig_basis = info['basis']
        self.lll_min_sq = int(info['lll_min_sq'])
        self.bkz_min_sq = int(info.get('bkz_min_sq', self.lll_min_sq))

    def reset(self, fixed_lattice=None):
        if fixed_lattice is not None:
            self._bind_lattice(fixed_lattice)
        else:
            self._bind_lattice(random.choice(self.lat_list))

        self.basis = lll_reduction_safe(self.orig_basis, delta=self.start_delta)

        n = BASIS_NUM_70
        U = [[0] * n for _ in range(n)]
        for i in range(n): U[i][i] = 1
        for i in range(1, n):
            for j in range(i): U[i][j] = random.randint(-RESET_UNIMODULAR_MAX_VAL, RESET_UNIMODULAR_MAX_VAL + 1)
        new_basis = [[sum(int(U[i][k]) * int(self.basis[k][j]) for k in range(n)) for j in range(n)] for i in range(n)]
        if not has_zero_row(new_basis): self.basis = new_basis

        self.steps = 0
        self.no_improve = 0
        self.best_shortest_sq = min_norm_sq(self.basis)
        self.best_basis = [r[:] for r in self.basis]
        self.episode_start_sq = self.best_shortest_sq
        self.zero_row_incidents = 0
        self.has_broken_lll = False
        self.has_reached_bkz = False

        return compute_state(self.basis, self.episode_start_sq, 0, MAX_STEPS_EPISODE)

    def apply_step_result(self, new_basis, zero_inc):
        self.basis = new_basis
        self.zero_row_incidents += zero_inc
        self.steps += 1

        if LLL_FALLBACK_INTERVAL > 0 and self.steps % LLL_FALLBACK_INTERVAL == 0:
            new_b = lll_reduction_safe(self.basis, delta=0.99)
            if not has_zero_row(new_b): self.basis = new_b

        new_min_sq = min_norm_sq(self.basis)
        if new_min_sq < self.best_shortest_sq:
            self.best_shortest_sq = new_min_sq
            self.best_basis = [r[:] for r in self.basis]
            self.no_improve = 0
        else:
            self.no_improve += 1


class PoolEnv70:
    def __init__(self, lat_list, num_envs):
        self.envs = []
        for i in range(num_envs):
            delta = START_LLL_DELTAS[i % len(START_LLL_DELTAS)]
            self.envs.append(SVPEnv70(lat_list, start_delta=delta))
        self.pool = Pool(processes=min(num_envs, cpu_count()))

    def reset_all(self):
        return np.stack([e.reset() for e in self.envs], axis=0)

    def step_all(self, selections):
        args_list = [(env.basis, sel) for env, sel in zip(self.envs, selections)]
        results = self.pool.map(_worker_step, args_list)

        s_next, rewards, dones, infos = [], [], [], []

        for i, (env, (new_b, zero_inc)) in enumerate(zip(self.envs, results)):
            old_min_sq = min_norm_sq(env.basis)
            env.apply_step_result(new_b, zero_inc)
            new_min_sq = min_norm_sq(env.basis)

            old_len = sqrt_float_from_sq(old_min_sq)
            new_len = sqrt_float_from_sq(new_min_sq)
            bkz_len = sqrt_float_from_sq(env.bkz_min_sq)

            reward = 0.0
            if old_len > 1e-8 and new_len > 1e-8:
                step_gain = math.log(old_len) - math.log(new_len)
                if step_gain > 0: reward += step_gain * 100.0

            if new_min_sq >= old_min_sq: reward -= 0.2

            if new_min_sq < env.lll_min_sq and not env.has_broken_lll:
                reward += 100.0
                env.has_broken_lll = True

            if env.bkz_min_sq > 0:
                ratio_bkz = new_min_sq / env.bkz_min_sq
                if ratio_bkz <= 1.00 and not env.has_reached_bkz:
                    reward += 5000.0
                    env.has_reached_bkz = True
                elif ratio_bkz <= 1.05:
                    reward += 500.0
                elif ratio_bkz <= 1.10:
                    reward += 200.0
                elif ratio_bkz <= 1.15:
                    reward += 50.0

            if env.no_improve > NO_IMPROVE_STEP_LIMIT: reward -= 5.0
            reward = float(np.clip(reward, -10.0, 5000.0))

            done = env.steps >= MAX_STEPS_EPISODE
            s_next.append(compute_state(env.basis, env.episode_start_sq, env.steps, MAX_STEPS_EPISODE))
            rewards.append(reward)
            dones.append(done)
            infos.append({
                'cur_shortest': new_len, 'lll_min': sqrt_float_from_sq(env.lll_min_sq),
                'bkz_min': bkz_len, 'zero': env.zero_row_incidents
            })

        return np.stack(s_next), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.float32), infos

    def reset_single(self, idx):
        return self.envs[idx].reset()

    def close(self):
        self.pool.close();
        self.pool.join()


# ===================== 网络与 PPO 逻辑 =====================
class ActorCritic70(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(STATE_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.selector = nn.Linear(HIDDEN_DIM, BASIS_NUM_70)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        x = torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0), -10.0, 10.0)
        h = self.fc2(self.fc1(x))
        return torch.nan_to_num(self.selector(h), nan=0.0), torch.nan_to_num(self.critic(h).squeeze(-1), nan=0.0)


class PPOAgent70:
    def __init__(self):
        self.model = ActorCritic70().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_MAX)

    def get_action(self, states, det=False):
        s = torch.from_numpy(np.clip(np.nan_to_num(states, nan=0.0), -10.0, 10.0)).to(DEVICE)
        with torch.no_grad():
            logits, values = self.model(s)
            if det:
                start_idx = logits.argmax(dim=-1)
            else:
                start_idx = Categorical(logits=logits).sample()

            B = start_idx.shape[0]
            arange_k = torch.arange(K_SELECT, device=DEVICE).unsqueeze(0)
            selected = (start_idx.unsqueeze(1) + arange_k) % BASIS_NUM_70

            log_probs = F.log_softmax(logits, dim=-1).gather(1, start_idx.unsqueeze(-1)).squeeze(-1)
            log_probs = torch.nan_to_num(log_probs, nan=-math.log(BASIS_NUM_70))

        return selected.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def update(self, states, selections, old_lp, adv, ret, entropy_coef):
        self.model.train()
        s = torch.from_numpy(np.clip(np.nan_to_num(states, nan=0.0), -10.0, 10.0)).to(DEVICE)
        sel = torch.LongTensor(selections).to(DEVICE)
        old = torch.FloatTensor(old_lp).to(DEVICE)
        adv = torch.FloatTensor(adv).to(DEVICE)
        ret = torch.FloatTensor(ret).to(DEVICE)

        adv_mean, adv_std = adv.mean(), adv.std()
        adv = (adv - adv_mean) / (adv_std + EPS) if adv_std.item() > 1e-6 else adv - adv_mean
        old, adv, ret = torch.nan_to_num(old, nan=0.0), torch.nan_to_num(adv, nan=0.0), torch.nan_to_num(ret, nan=0.0)

        total_loss, num, skipped = 0, 0, 0
        for _ in range(PPO_EPOCHS):
            idxs = np.random.permutation(len(s))
            for st in range(0, len(s), BATCH_SIZE):
                idx = idxs[st:st + BATCH_SIZE]
                logits, values = self.model(s[idx])

                start_idx = sel[idx][:, 0]
                new_logp = F.log_softmax(logits, dim=-1).gather(1, start_idx.unsqueeze(-1)).squeeze(-1)
                new_logp = torch.nan_to_num(new_logp, nan=0.0)

                p = F.softmax(logits, dim=-1)
                ent = torch.nan_to_num(-(p * (p + 1e-10).log()).sum(dim=-1).mean(), nan=0.0)

                ratio = torch.exp(torch.clamp(new_logp - old[idx], -10.0, 10.0))
                surr1, surr2 = ratio * adv[idx], torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv[idx]
                loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(values, ret[idx]) - entropy_coef * ent

                if torch.isnan(loss) or torch.isinf(loss): skipped += 1; continue

                self.optimizer.zero_grad();
                loss.backward()
                if any(p_.grad is not None and (torch.isnan(p_.grad).any() or torch.isinf(p_.grad).any()) for p_ in
                       self.model.parameters()):
                    skipped += 1;
                    self.optimizer.zero_grad();
                    continue

                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item();
                num += 1
        if skipped > 0: logger.warning(f"  [PPO] 跳过 {skipped} 个 NaN batch")
        return total_loss / max(num, 1)


class RewardRM:
    def __init__(self): self.mean, self.var, self.count = 0, 1, 1e-4

    def update(self, x):
        x = np.clip(np.nan_to_num(x, nan=0), -1000, 5000)
        d = np.mean(x) - self.mean;
        total = self.count + len(x)
        self.mean += d * len(x) / total
        self.var = max((self.var * self.count + np.var(x) * len(x) + d ** 2 * self.count * len(x) / total) / total, EPS)
        self.count = total

    def normalize(self, x): return (np.clip(np.nan_to_num(x, nan=0), -1000, 5000) - self.mean) / np.sqrt(self.var + EPS)


def get_lr(step, total):
    if step < WARMUP_STEPS: return LR_MAX * (step + 1) / WARMUP_STEPS
    if step < total * 0.8: return LR_MAX
    progress = (step - WARMUP_STEPS) / max(total * 0.2, 1)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))


# ===================== 训练器 =====================
class Trainer70:
    def __init__(self, infos):
        self.infos = infos
        self.agent = PPOAgent70()
        self.rms = RewardRM()
        self.steps = 0
        self.start_time = time.time()

    def train(self):
        vec_env = PoolEnv70(self.infos, NUM_ENVS)
        states = vec_env.reset_all()
        logger.info(f"开始训练 70维 V-ULTIMATE！设备：{DEVICE}")
        logger.info(f"K_SELECT={K_SELECT}  EPISODE={MAX_STEPS_EPISODE}  NUM_ENVS={NUM_ENVS}")

        try:
            while self.steps < TOTAL_TRAIN_STEPS:
                current_lr = get_lr(self.steps, TOTAL_TRAIN_STEPS)
                for pg in self.agent.optimizer.param_groups: pg['lr'] = current_lr
                progress = self.steps / TOTAL_TRAIN_STEPS
                entropy_coef = ENTROPY_COEF_HI * (1 - progress) + ENTROPY_COEF_LO * progress

                buf_s, buf_sel, buf_lp, buf_r, buf_d, buf_v, buf_i = [], [], [], [], [], [], []

                for step in range(MAX_STEPS_EPISODE):
                    selections, lp, v = self.agent.get_action(states)
                    current_states = states.copy()
                    s_next, r, d, infos = vec_env.step_all(selections)

                    for i in range(NUM_ENVS):
                        if d[i]: s_next[i] = vec_env.reset_single(i)

                    buf_s.append(current_states);
                    buf_sel.append(selections);
                    buf_lp.append(lp)
                    buf_r.append(r);
                    buf_d.append(d);
                    buf_v.append(v);
                    buf_i.append(infos)
                    states = s_next;
                    self.steps += 1

                    if step % 50 == 0:
                        b_idx = min(range(NUM_ENVS), key=lambda i: infos[i].get('cur_shortest', float('inf')))
                        b = infos[b_idx]
                        speed = self.steps / max(time.time() - self.start_time, 1)
                        logger.info(f"Step {self.steps} ({speed:.2f}/s) | "
                                    f"Best E{b_idx} | PPO={b['cur_shortest']:.1f} | LLL={b['lll_min']:.1f} | BKZ={b['bkz_min']:.1f}")

                _, _, last_v = self.agent.get_action(states)
                buf_v.append(last_v)

                T = MAX_STEPS_EPISODE
                all_s = np.stack(buf_s, 0).transpose(1, 0, 2).reshape(-1, STATE_DIM)
                all_sel = np.stack(buf_sel, 0).transpose(1, 0, 2).reshape(-1, K_SELECT)
                all_lp = np.stack(buf_lp, 0).transpose(1, 0).reshape(-1)
                all_r = np.stack(buf_r, 0).transpose(1, 0).reshape(-1)
                all_d = np.stack(buf_d, 0).transpose(1, 0).reshape(-1)
                all_v = np.stack(buf_v[:-1], 0).transpose(1, 0).reshape(-1)

                self.rms.update(all_r)
                all_rn = self.rms.normalize(all_r)

                all_adv, all_ret = np.zeros_like(all_rn), np.zeros_like(all_rn)
                for ei in range(NUM_ENVS):
                    st = ei * T
                    ga = np.zeros(T, dtype=np.float32)
                    last_gae = 0
                    for t in reversed(range(T)):
                        mask = 1 - all_d[st + t]
                        nv = 0 if t == T - 1 else all_v[st + t + 1]
                        delta = all_rn[st + t] + GAMMA * nv * mask - all_v[st + t]
                        last_gae = delta + GAMMA * GAE_LAMBDA * mask * last_gae
                        ga[t] = last_gae
                    all_adv[st:st + T] = ga
                    all_ret[st:st + T] = ga + all_v[st:st + T]

                loss = self.agent.update(all_s, all_sel, all_lp, all_adv, all_ret, entropy_coef)

                if self.steps % 2000 < NUM_ENVS:
                    torch.save(self.agent.model.state_dict(), MODEL_SAVE_PATH)
                    logger.info(f"    [Save] 模型已保存 (Step {self.steps})")
        finally:
            vec_env.close()


# ===================== 主程序 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lattices", type=int, default=50)
    parser.add_argument("--steps", type=int, default=TOTAL_TRAIN_STEPS)
    args = parser.parse_args()
    TOTAL_TRAIN_STEPS = args.steps

    logger.info("正在预处理格子（计算 LLL 和 BKZ40 真值）...")
    files = sorted(glob.glob(os.path.join(LATTICE_DIR, FILE_PATTERN)))[:args.max_lattices]
    infos = []
    for i, fp in enumerate(files):
        basis = load_lattice_70d(os.path.basename(fp))
        lll_sq = min_norm_sq(lll_reduction_safe(basis, 0.99))

        try:
            # 70维下，BKZ(40)作为参考基准
            B_mat = IntegerMatrix.from_matrix([[int(x) for x in r] for r in basis])
            par = BKZ.Param(block_size=40, strategies=BKZ.DEFAULT_STRATEGY, max_loops=1)
            BKZ.reduction(B_mat, par)
            bkz_sq = min_norm_sq([[int(B_mat[i, j]) for j in range(70)] for i in range(70)])
        except Exception:
            bkz_sq = lll_sq

        infos.append(
            {'file': os.path.basename(fp), 'basis': basis, 'lll_min_sq': int(lll_sq), 'bkz_min_sq': int(bkz_sq)})
        if (i + 1) % 5 == 0:
            logger.info(
                f"  [{i + 1}/{len(files)}] {os.path.basename(fp)}: LLL={sqrt_float_from_sq(lll_sq):.1f}, BKZ40={sqrt_float_from_sq(bkz_sq):.1f}")

    trainer = Trainer70(infos)
    trainer.train()