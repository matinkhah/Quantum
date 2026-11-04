# %% [markdown]
# # F-IVQNN — complete experiment notebook
# **Federated Improved Variational Quantum Neural Network (F-IVQNN) optimised via PSO for network-security situation awareness**
#
# Runs every computational experiment (E1–E10), builds figures/tables, and exports a compact `PASTE_BACK` JSON block for paper revision.
#
# **How to use:** run the cells top-to-bottom. Choose `QUICK_MODE` in Cell 2 (smoke test, ~20 min) or `FULL_MODE`
# (all experiments, checkpointed to Google Drive; safe to re-run after a disconnect — finished (experiment, dataset, seed) items are skipped).
#
# **Honesty rules built in:** nothing is hard-coded or simulated "as if real"; anything that cannot be computed prints `NOT COMPUTED: reason`.
# Same tuning budget for every method, train-only fitting of all preprocessing, test set touched once per (model, seed).
#
# **Cell index:** 1 installs · 2 config · 3 storage/logging/utils · 4 data loaders · 5 preprocessing/splits · 6 circuits+simulator ·
# 7 simulator validation · 8 models/metrics/stats · 9 optimisers · 10 federated · 11 run helpers · 12 calibration/budget plan ·
# 13 E1 · 14 E2 · 15 E3 · 16 E4 · 17 E5 · 18 E6 · 19 E7 · 20 E8 · 21 E9 · 22 E10 · 23 figures · 24 tables · 25 export + PASTE_BACK · 26 re-run guide

# %% [markdown]
# ## Cell 1 — Installs and library versions

# %%
import subprocess, sys, platform

def _pip(*pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True, text=True)
    if r.returncode != 0:
        print("pip install problem for", pkgs, "->", r.stderr[-400:])

# numpy / scipy / pandas / scikit-learn / matplotlib / torch ship with Colab.
_pip("pennylane", "statsmodels", "tqdm", "kagglehub", "huggingface_hub", "pyarrow")

def env_report() -> str:
    import importlib
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}"]
    for m in ["numpy", "scipy", "pandas", "sklearn", "torch", "pennylane", "statsmodels", "matplotlib",
              "huggingface_hub", "kagglehub", "xgboost"]:
        try:
            mod = importlib.import_module(m)
            lines.append(f"{m} {getattr(mod, '__version__', '?')}")
        except Exception:
            lines.append(f"{m} NOT INSTALLED")
    return "\n".join(lines)

ENV_TXT = env_report()
print(ENV_TXT)

# %% [markdown]
# ## Cell 2 — ONE config cell (all settings live here)
# `QUICK_MODE=True` → smoke test (2 seeds, small subsamples, few rounds). `QUICK_MODE=False` → FULL_MODE.
# Every design choice the paper leaves open is listed in `DESIGN` (Cell 3) and exported in the paste-back block.

# %%
import os, json, time, math, gc, re, io, zlib, hashlib, itertools, warnings, traceback, zipfile, shutil, copy
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
import numpy as np, pandas as pd
import scipy.stats as st
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, **k): return x if x is not None else range(0)
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

QUICK_MODE = False          # <<< set False for FULL_MODE

@dataclass
class Cfg:
    quick: bool = True
    seeds: tuple = (0, 1)                 # E1 seeds
    seeds_small: tuple = (0, 1)           # E2/E3/E4/E8/E9 seeds
    datasets: tuple = ("kddcup99", "unsw_nb15")
    sec_datasets: tuple = ("kddcup99",)   # datasets for E2/E3/E9 (planner may shrink)
    # ---- data
    kdd_percent10: bool = True
    kdd_urls: tuple = ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",)  # tried at runtime only
    unsw_urls: tuple = ()                 # paste verified official train/test CSV URLs here (optional)
    unsw_kaggle_slugs: tuple = ("mrwellsdavid/unsw-nb15",)   # tried at runtime; schema is verified after download
    allow_upload: bool = True             # last-resort manual upload prompt (blocks until you upload / cancel)
    dedup: bool = True                    # de-duplicate BEFORE splitting (prevents train/test leakage)
    log1p: bool = True
    skew_thr: float = 2.0                 # log1p applied to non-negative columns with train skew > thr
    ohe_min_freq: int = 50
    n_train_max: int = 4000
    n_val_max: int = 2000
    n_test_max: int = 4000
    balance_train: bool = True
    val_frac: float = 0.2
    kdd_test_frac: float = 0.2
    q_max: int = 10
    # ---- circuit
    Q: int = 8
    L: int = 3
    Qp: int = 4
    gateset: str = "real"                 # 'real' = RY+CNOT(+CRY pooling); 'full' adds RZ/CRZ (complex simulation, slower)
    entangle: str = "ring"                # 'ring' (CNOT ring / brick conv) or 'full' (all-to-all; used for complexity study)
    conv_share: bool = True               # QCNN-style weight sharing inside each conv layer
    head: str = "affine"                  # 'affine': p=sigmoid(gain*s*f+b) with learnable s,b ; 'linear': p=(1+f)/2
    head_gain: float = 4.0
    init_std: float = 0.5
    # ---- PSO
    P: int = 10
    T: int = 8
    w: float = 0.7
    phi1: float = 1.5
    phi2: float = 1.5
    vmax: float = 0.5
    eps: float = 1e-8
    pso_batch: int = 64                   # fitness mini-batch (0 => all local data)
    swarm_init: str = "gauss_global"      # gauss_global | random | global_plus_random
    init_sigma: float = 0.3
    resample_every: int = 0               # 0 => one mini-batch per round/run; k => new mini-batch every k iterations
    pso_backend: str = "auto"             # auto | numpy | torch
    # ---- gradient / SPSA
    adam_lr: float = 0.05
    adam_batch: int = 64
    local_adam_steps: int = 20
    spsa_a: float = 0.15
    spsa_c: float = 0.1
    # ---- federated
    C: int = 3
    R: int = 5
    partition: str = "iid"                # iid | dirichlet
    alpha: float = 0.5
    eval_cap: int = 600
    # ---- tuning (3 candidates per method, validation split, seed 0)
    tune_adam_steps: int = 60
    tune_pso_T: int = 15
    grid_adam_lr: tuple = (0.01, 0.03, 0.1)
    grid_pso_w: tuple = (0.5, 0.7, 0.9)
    grid_spsa_a: tuple = (0.05, 0.15, 0.5)
    grid_lr_C: tuple = (0.1, 1.0, 10.0)
    grid_mlp: tuple = ((16,), (32,), (64, 32))
    # ---- sweeps / experiment sizes
    fed_C_list: tuple = (3, 5)
    alpha_list: tuple = (0.1, 1.0)
    sweep_P: tuple = (5, 10)
    sweep_T: tuple = (5, 10)
    sweep_L: tuple = (1, 2)
    sweep_Q: tuple = (4, 6)
    bp_Q: tuple = (4, 6, 8)
    bp_inits: int = 60
    bp_batch: int = 32
    cplx_Q: tuple = (4, 6, 8, 10)
    cplx_L: tuple = (1, 2, 3, 4)
    cplx_Qp: tuple = (2, 3, 4, 5, 6)
    # ---- forecasting / noise / claims
    window_sec: int = 60
    min_win_count: int = 30
    max_raw_rows: int = 600000            # 0 => all rows of the raw UNSW files
    forecast_tol_mse: float = 0.01        # tolerance fixed in advance (claim vii)
    noise_p: tuple = (0.0, 0.002, 0.01, 0.03)
    noise_n: int = 100
    ece_thr: float = 0.05
    gap_pass: float = 0.01
    gap_fail: float = 0.03
    time_budget_h: float = 12
    resume: bool = True
    use_xgboost: bool = True

def make_cfg(quick: bool) -> Cfg:
    c = Cfg(quick=quick)
    if not quick:
        c.seeds = (0, 1, 2, 3, 4); c.seeds_small = (0, 1, 2)
        c.sec_datasets = ("kddcup99", "unsw_nb15")
        c.n_train_max, c.n_val_max, c.n_test_max = 20000, 5000, 10000
        c.P, c.T, c.R, c.C = 20, 20, 20, 5
        c.local_adam_steps = 20
        c.eval_cap = 1000
        c.tune_adam_steps, c.tune_pso_T = 100, 30
        c.fed_C_list = (3, 5, 10); c.alpha_list = (0.1, 0.5, 1.0)
        c.sweep_P = (5, 10, 20, 40); c.sweep_T = (5, 10, 20, 40); c.sweep_L = (1, 2, 3, 4); c.sweep_Q = (4, 6, 8, 10)
        c.bp_Q = (4, 6, 8, 10); c.bp_inits = 200; c.cplx_Q = (4, 6, 8, 10, 12, 14)
        c.noise_p = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05); c.noise_n = 200
        c.max_raw_rows = 0
    return c

CFG = make_cfg(QUICK_MODE)
print("MODE:", "QUICK" if CFG.quick else "FULL")
print(json.dumps(asdict(CFG), indent=1, default=str)[:3500])

# %% [markdown]
# ## Cell 3 — Storage (Drive), logging, checkpointing, registries, small utilities

# %%
def _setup_root():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        return Path("/content/drive/MyDrive/fivqnn_results")
    except Exception as e:
        print("Drive not mounted (", type(e).__name__, ") -> using local folder.")
        return Path("/content/fivqnn_results") if os.path.exists("/content") else Path("./fivqnn_results")

ROOT = _setup_root()
for sub in ("figures", "tables", "ckpt_quick", "ckpt_full", "preds", "data"):
    (ROOT / sub).mkdir(parents=True, exist_ok=True)
CKPT = ROOT / ("ckpt_quick" if CFG.quick else "ckpt_full")
DATA_DIR = ROOT / "data"
LOG_PATH = ROOT / "logs.txt"
T_START = time.time()

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    try:
        with open(LOG_PATH, "a") as f: f.write(f"[{time.strftime('%H:%M:%S')}] {s}\n")
    except Exception:
        pass

def jsonable(o):
    if isinstance(o, np.floating): o = float(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    return o

class Store:
    """Tiny JSON key/value store on Drive: one file per (experiment, dataset, seed, variant)."""
    def __init__(self, root: Path): self.root = root; root.mkdir(parents=True, exist_ok=True)
    def _p(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.=,\-]+", "_", key)[:90]
        return self.root / f"{safe}_{hashlib.md5(key.encode()).hexdigest()[:10]}.json"
    def get(self, key):
        p = self._p(key)
        if p.exists():
            try:
                with open(p) as f: return json.load(f)
            except Exception: return None
        return None
    def put(self, key, obj):
        p = self._p(key); tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f: json.dump(jsonable(obj), f)
        os.replace(tmp, p)

STORE = Store(CKPT)

def cached(key: str, fn):
    if CFG.resume:
        v = STORE.get(key)
        if v is not None: return v
    v = fn()
    STORE.put(key, v)
    return STORE.get(key) if v is not None else v

def rng_for(*parts) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(str(p).encode()) & 0xFFFFFFFF for p in parts])

STATUS, NOT_COMPUTED, DESIGN = {}, {}, {}
def not_computed(name: str, reason: str):
    NOT_COMPUTED[name] = reason
    log(f"NOT COMPUTED [{name}]: {reason}")

def run_experiment(name: str, fn, *a, **k):
    t0 = time.time()
    try:
        fn(*a, **k)
        STATUS[name] = f"ran ({time.time()-t0:.0f}s)"
    except Exception as e:
        STATUS[name] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
        log(f"!! {name} failed:\n" + traceback.format_exc()[-1500:])

DESIGN.update({
    "encoding": "angle encoding RY(x_j) on |0>, one feature per qubit; x scaled to [0,pi] via train-only 1st/99th percentile min-max (clipped) after train-only PCA to Q components",
    "layers_VQNN": "per layer: RY on every qubit (+RZ if gateset=full) then CNOT ring (all-pairs if entangle=full)",
    "layers_IVQNN": "layer 1: QCNN-style conv (brick of 2-qubit blocks RY,RY,CNOT,RY,RY with weight sharing) on all Q qubits; pooling Q->Q' via CRY(discarded->nearest kept) then discarded qubits are traced out; layers 2..L: conv blocks on the Q' kept qubits",
    "pooling_equivalence": "pure-state simulation: tracing out qubits after pooling == ignoring them when measuring a kept-qubit observable; implemented by measuring <Z> on kept qubit only (marginal probabilities = partial trace)",
    "observable": "Pauli-Z on qubit 0 (VQNN) / first kept qubit (IVQNN)",
    "output_mapping": "p=sigmoid(head_gain*s*f+b) with learnable scalars s (init 1), b (init 0), head_gain=4; BCE loss (option head=linear: p=(1+f)/2)",
    "param_init": "circuit angles ~ N(0, 0.5^2); first federated round starts from this global model",
    "swarm_init": "default gauss_global: all particles = received global model + N(0, 0.3^2); ablation: random, global_plus_random",
    "pso_minibatching": "fitness on ONE random local mini-batch drawn at the start of each local PSO run (all particles & all iterations share it, so personal/global bests are comparable); resample_every>0 re-evaluates pbest on a new batch",
    "pso_velocity": "per-dimension r1,r2~U[0,1]; clamp |v|<=vmax=0.5; initial velocity small uniform; fitness F=1/(L+eps); returned local model = swarm best g*",
    "partitioning": "IID: random split with unequal sizes (Dirichlet(5) proportions, min share); non-IID: per-class Dirichlet(alpha) label skew (sizes unequal), min size enforced",
    "fedavg": "weights n_c/N over all clients each round (full participation); asserted to sum to 1",
    "feature_reduction": "train-only one-hot (rare levels merged), log1p of skewed non-negative columns, StandardScaler, PCA to Q comps",
    "conv_weight_sharing": "True by default (QCNN-style); ablation in E3",
    "adam_gradient": "exact backprop through the statevector simulator (equivalent to parameter-shift); evaluation counts reported both as forward passes and as parameter-shift-equivalent circuit executions (2*#circuit params per sample per step)",
    "spsa_substitute": "SPSA used as the extra gradient-free optimiser; COBYLA not implemented",
    "evaluation_threshold": "0.5 on predicted probability; ECE uses 15 equal-width bins on the positive-class probability",
})
print("ROOT:", ROOT, "| CKPT:", CKPT)

# %% [markdown]
# ## Cell 4 — Dataset loaders (loader chain per dataset, every source verified at runtime)
# KDDCUP99: sklearn `fetch_kddcup99` → URL(s) in config → local/Drive file → manual upload.
# UNSW-NB15: local/Drive → official URLs (if you set them) → Kaggle (`kagglehub`, Colab Secrets `KAGGLE_USERNAME`/`KAGGLE_KEY`) → Hugging-Face mirrors (schema verified) → manual upload.
# Nothing is assumed: after each download the schema is checked; sizes are compared with the expected official sizes (warning only).

# %%
import fnmatch
KDD_COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
 "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
 "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
 "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
 "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
 "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
DOWNLOAD_ROOTS = []
SOURCES = {}
UPLOADED = set()
DATASET_NAMES = ("kddcup99", "unsw_nb15")

def _search(patterns, roots=None):
    roots = roots or ([str(DATA_DIR)] + DOWNLOAD_ROOTS)
    out = []
    for r in roots:
        for dp, _, fs in os.walk(r):
            for f in fs:
                if any(fnmatch.fnmatch(f.lower(), p.lower()) for p in patterns): out.append(os.path.join(dp, f))
    return sorted(set(out))

def _read_any(path, **kw):
    if path.lower().endswith(".parquet"): return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False, **kw)

def _decode(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else v)
    return df

def _manual_upload(prompt):
    if not CFG.allow_upload: return []
    try:
        from google.colab import files
        print(">>> MANUAL UPLOAD:", prompt, "(cancel the dialog to skip)")
        up = files.upload()
        paths = []
        for name, data in up.items():
            p = DATA_DIR / name
            with open(p, "wb") as f: f.write(data)
            paths.append(str(p)); UPLOADED.add(str(p))
        return paths
    except Exception as e:
        log("manual upload unavailable:", type(e).__name__)
        return []

def load_kdd():
    errs = []
    try:
        from sklearn.datasets import fetch_kddcup99
        b = fetch_kddcup99(percent10=CFG.kdd_percent10, as_frame=True)
        df = _decode(b.frame.copy())
        lab = [c for c in df.columns if str(c).lower() in ("labels", "label")]
        df = df.rename(columns={lab[0]: "label"})
        return df, "sklearn.datasets.fetch_kddcup99(percent10=%s)" % CFG.kdd_percent10
    except Exception as e:
        errs.append(f"sklearn: {type(e).__name__}: {str(e)[:120]}")
    for url in CFG.kdd_urls:
        try:
            df = pd.read_csv(url, header=None, names=KDD_COLS)
            return df, url
        except Exception as e:
            errs.append(f"{url}: {type(e).__name__}: {str(e)[:100]}")
    for p in _search(["kddcup*10_percent*", "kddcup.data*", "kddcup99*.csv"]) + _manual_upload("KDDCUP99 file (42 columns, no header)"):
        try:
            df = pd.read_csv(p, header=None, names=KDD_COLS)
            if df.shape[1] == 42: return df, p
        except Exception as e:
            errs.append(f"{p}: {type(e).__name__}")
    raise RuntimeError("all KDDCUP99 sources failed: " + " | ".join(errs))

def _unsw_valid(df):
    cols = {str(c).strip().lower() for c in df.columns}
    return {"proto", "service", "state", "label"} <= cols and df.shape[1] >= 40

def _unsw_from_roots():
    cands = _search(["*training-set*", "*testing-set*", "*unsw*train*", "*unsw*test*", "train*.csv", "test*.csv", "*train*.parquet", "*test*.parquet"])
    best = {"train": None, "test": None}
    for p in cands:
        pl = os.path.basename(p).lower()
        if not pl.endswith((".csv", ".parquet")): continue
        kind = "train" if "train" in pl else ("test" if "test" in pl else None)
        if kind is None: continue
        try: df = _read_any(p)
        except Exception: continue
        if not _unsw_valid(df): continue
        if best[kind] is None or len(df) > len(best[kind][1]): best[kind] = (p, df)
    return best["train"], best["test"]

def _kaggle_env():
    try:
        from google.colab import userdata
        for k in ("mojtabamatinkhah", "898f3b4c5ddd20bb7b93fd628649f84a"): 
            try: os.environ.setdefault(k, userdata.get(k))
            except Exception: pass
    except Exception: pass

def _dl_official():
    import requests
    for url in CFG.unsw_urls:
        try:
            r = requests.get(url, stream=True, timeout=60); r.raise_for_status()
            p = DATA_DIR / os.path.basename(url.split("?")[0])
            with open(p, "wb") as f:
                for chunk in r.iter_content(1 << 20): f.write(chunk)
            log("downloaded", url)
        except Exception as e:
            log("official url failed:", url, type(e).__name__)

def _dl_kaggle():
    _kaggle_env()
    import kagglehub
    for s in CFG.unsw_kaggle_slugs:
        try:
            p = kagglehub.dataset_download(s); DOWNLOAD_ROOTS.append(p); SOURCES.setdefault("unsw_kaggle", []).append(s)
            log("kaggle ok:", s, "->", p)
        except Exception as e:
            log("kaggle failed:", s, type(e).__name__, str(e)[:120])

def _dl_hf(need="partition"):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); n_dl = 0
    for q in ("UNSW-NB15", "UNSW_NB15", "unsw nb15"):
        try: res = list(api.list_datasets(search=q, limit=20))
        except Exception as e:
            log("HF search failed:", type(e).__name__); continue
        for d in res:
            try: files = api.list_repo_files(d.id, repo_type="dataset")
            except Exception: continue
            pick = []
            for f in files:
                fl = f.lower()
                if not fl.endswith((".csv", ".parquet")): continue
                if need == "partition" and ("train" in fl or "test" in fl): pick.append(f)
                if need == "raw" and re.search(r"unsw[-_]nb15_[1-4]\.csv", fl): pick.append(f)
            for f in pick[:6]:
                try:
                    p = hf_hub_download(d.id, f, repo_type="dataset"); DOWNLOAD_ROOTS.append(os.path.dirname(p)); n_dl += 1
                    SOURCES.setdefault("unsw_hf", []).append(f"{d.id}/{f}")
                except Exception as e:
                    log("HF download failed:", d.id, f, type(e).__name__)
            if need == "partition" and n_dl >= 2 and all(_unsw_from_roots()): return
    if n_dl == 0: log("no Hugging-Face candidate files downloaded")

def load_unsw():
    errs = []
    tr, te = _unsw_from_roots()
    for step, fn in (("official_urls", _dl_official), ("kaggle", _dl_kaggle), ("huggingface", _dl_hf)):
        if tr and te: break
        try: fn()
        except Exception as e: errs.append(f"{step}: {type(e).__name__}: {str(e)[:100]}")
        tr, te = _unsw_from_roots()
    if not (tr and te):
        _manual_upload("UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv")
        tr, te = _unsw_from_roots()
    if not (tr and te): raise RuntimeError("UNSW-NB15 partition files not found. " + " | ".join(errs))
    return tr[1], te[1], {"train_file": tr[0], "test_file": te[0]}

def to_binary(name, df):
    d = df.copy(); d.columns = [str(c).strip() for c in d.columns]
    if name == "kddcup99":
        y = (d["label"].astype(str).str.strip().str.rstrip(".") != "normal").astype(int).values
        attack = d["label"].astype(str).str.strip().str.rstrip(".")
        X = d.drop(columns=["label"]); cat = ["protocol_type", "service", "flag"]
    else:
        lc = {c.lower(): c for c in d.columns}
        y = pd.to_numeric(d[lc["label"]], errors="coerce").fillna(0).astype(int).values
        attack = d[lc["attack_cat"]].astype(str) if "attack_cat" in lc else None
        X = d.drop(columns=[lc[k] for k in ("label", "attack_cat", "id") if k in lc])
        X.columns = [c.lower() for c in X.columns]
        cat = [c for c in ("proto", "service", "state") if c in X.columns]
    for c in X.columns:
        if c in cat: X[c] = X[c].astype(str)
        else: X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.reset_index(drop=True), y, cat, (None if attack is None else attack.reset_index(drop=True))

def _dedup_mask(X, y):
    h = pd.util.hash_pandas_object(X.assign(_y=y), index=False).values
    return ~pd.Series(h).duplicated().values, int(pd.Series(h).duplicated().sum())

EXPECTED = {"kddcup99": 494021, "unsw_train": 175341, "unsw_test": 82332}
BASE = {}

def get_base(name):
    """Load + binarise + (optionally) de-duplicate one dataset. Returns dict or None (NOT COMPUTED)."""
    if name in BASE: return BASE[name]
    try:
        if name == "kddcup99":
            df, src = load_kdd(); SOURCES["kddcup99"] = src
            X, y, cat, att = to_binary(name, df)
            n0 = len(y); m, ndup = _dedup_mask(X, y)
            exp = EXPECTED["kddcup99"] if CFG.kdd_percent10 else 4898431
            if abs(n0 - exp) > 0.01 * exp: log(f"WARNING: KDD rows {n0} != expected ~{exp}")
            if CFG.dedup: X, y, att = X[m].reset_index(drop=True), y[m], att[m].reset_index(drop=True)
            stats = dict(source=src, rows_raw=n0, duplicates=ndup, dedup_applied=CFG.dedup, rows_used=len(y),
                         features=X.shape[1], attack_frac=float(y.mean()),
                         top_attack_cats=att.value_counts().head(3).to_dict())
            BASE[name] = dict(X=X, y=y, cat=cat, attack=att, stats=stats)
        else:
            dtr, dte, info = load_unsw(); SOURCES["unsw_files"] = info
            Xtr, ytr, cat, atr = to_binary(name, dtr); Xte, yte, _, ate = to_binary(name, dte)
            ntr0, nte0 = len(ytr), len(yte)
            for nm, n0, e in (("train", ntr0, EXPECTED["unsw_train"]), ("test", nte0, EXPECTED["unsw_test"])):
                if abs(n0 - e) > 0.01 * e: log(f"WARNING: UNSW {nm} rows {n0} != official {e} (partition may differ from the official one)")
            mtr, dtr_n = _dedup_mask(Xtr, ytr); mte, dte_n = _dedup_mask(Xte, yte)
            htr = set(pd.util.hash_pandas_object(Xtr.assign(_y=ytr), index=False).values)
            hte = pd.util.hash_pandas_object(Xte.assign(_y=yte), index=False).values
            overlap = int(np.isin(hte, list(htr)).sum())
            if CFG.dedup:
                Xtr, ytr = Xtr[mtr].reset_index(drop=True), ytr[mtr]
                Xte, yte = Xte[mte].reset_index(drop=True), yte[mte]
            stats = dict(source=info, rows_raw_train=ntr0, rows_raw_test=nte0, duplicates_train=dtr_n, duplicates_test=dte_n,
                         test_rows_identical_to_a_train_row=overlap, dedup_applied=CFG.dedup, rows_train=len(ytr), rows_test=len(yte),
                         features=Xtr.shape[1], attack_frac_train=float(ytr.mean()), attack_frac_test=float(yte.mean()),
                         partition_matches_official=(ntr0 == EXPECTED["unsw_train"] and nte0 == EXPECTED["unsw_test"]))
            BASE[name] = dict(X=Xtr, y=ytr, cat=cat, attack=atr, Xte=Xte, yte=yte, stats=stats)
        log(f"[{name}] stats:", json.dumps(jsonable(BASE[name]["stats"]), default=str)[:600])
        if name == "kddcup99":
            log("  known issue: KDDCUP99 is dominated by smurf/neptune and heavily duplicated; dedup flag =", CFG.dedup)
        return BASE[name]
    except Exception as e:
        not_computed(f"dataset:{name}", f"{type(e).__name__}: {str(e)[:300]}")
        BASE[name] = None
        return None

# %% [markdown]
# ## Cell 5 — Preprocessing and splits (everything fitted on the training split only)
# Dedup (before splitting) → stratified train/val/test → balanced train subsample → one-hot (train levels) → log1p (train skew) → StandardScaler → PCA(Q) → [0,π] angles.

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

class Prep:
    def fit(self, Xdf, cat):
        self.cat = cat
        num = [c for c in Xdf.columns if c not in cat]
        A = Xdf[num]
        self.num = [c for c in num if A[c].nunique(dropna=True) > 1]
        self.dropped_const = [c for c in num if c not in self.num]
        self.med = A[self.num].median().fillna(0)
        Ai = A[self.num].fillna(self.med)
        self.logc = []
        if CFG.log1p:
            sk = Ai.skew()
            self.logc = [c for c in self.num if Ai[c].min() >= 0 and sk[c] > CFG.skew_thr]
        self.ohe = None
        if cat:
            try: self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=CFG.ohe_min_freq)
            except TypeError: self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            self.ohe.fit(Xdf[cat])
        self.sc = StandardScaler().fit(self._mat(Xdf))
        Z = self.sc.transform(self._mat(Xdf))
        self.pca = PCA(n_components=min(CFG.q_max, Z.shape[1]), random_state=0).fit(Z)
        return self
    def _mat(self, Xdf):
        A = Xdf[self.num].fillna(self.med).astype(float).copy()
        for c in self.logc: A[c] = np.log1p(A[c].clip(lower=0))
        M = A.values
        if self.ohe is not None: M = np.hstack([M, self.ohe.transform(Xdf[self.cat])])
        return M
    def transform(self, Xdf):
        Z = self.sc.transform(self._mat(Xdf))
        return np.clip(Z, -10, 10).astype(np.float32)

@dataclass
class Split:
    name: str; seed: int
    Xtr: np.ndarray; ytr: np.ndarray; Xva: np.ndarray; yva: np.ndarray; Xte: np.ndarray; yte: np.ndarray
    prep: Prep; info: dict
    _qc: dict = field(default_factory=dict)
    def q(self, Q: int):
        """Q-dimensional angle features in [0,pi] (PCA + percentile min-max fitted on train only)."""
        if Q in self._qc: return self._qc[Q]
        if Q > self.prep.pca.n_components_: raise ValueError(f"Q={Q} > available PCA components {self.prep.pca.n_components_}")
        Z = [self.prep.pca.transform(a)[:, :Q] for a in (self.Xtr, self.Xva, self.Xte)]
        lo, hi = np.percentile(Z[0], 1, axis=0), np.percentile(Z[0], 99, axis=0)
        hi = np.where(hi - lo < 1e-9, lo + 1.0, hi)
        f = lambda M: (np.clip((M - lo) / (hi - lo), 0, 1) * np.pi).astype(np.float32)
        self._qc[Q] = (f(Z[0]), f(Z[1]), f(Z[2]))
        return self._qc[Q]

def _strat_sub(y, n_max, rng, balanced):
    pos = [np.where(y == k)[0] for k in (0, 1)]
    if balanced:
        m = min(n_max // 2, len(pos[0]), len(pos[1]))
        sel = np.concatenate([rng.choice(p, m, replace=False) for p in pos])
    elif len(y) > n_max:
        sel, _ = train_test_split(np.arange(len(y)), train_size=n_max, stratify=y, random_state=int(rng.integers(1 << 30)))
    else:
        sel = np.arange(len(y))
    return np.sort(sel)

SPLITS = {}
def get_split(name, seed) -> Split:
    key = (name, seed, CFG.dedup, CFG.n_train_max, CFG.n_test_max, CFG.balance_train, CFG.log1p, CFG.q_max)
    if key in SPLITS: return SPLITS[key]
    base = get_base(name)
    if base is None: raise RuntimeError(f"dataset {name} unavailable")
    rng = rng_for("split", name, seed)
    X, y = base["X"], base["y"]; idx = np.arange(len(y))
    if name == "kddcup99":
        i_trv, i_te = train_test_split(idx, test_size=CFG.kdd_test_frac, stratify=y, random_state=seed)
        i_tr, i_va = train_test_split(i_trv, test_size=CFG.val_frac, stratify=y[i_trv], random_state=seed)
        Xte_df, yte = X.iloc[i_te].reset_index(drop=True), y[i_te]
    else:
        i_tr, i_va = train_test_split(idx, test_size=CFG.val_frac, stratify=y, random_state=seed)
        Xte_df, yte = base["Xte"], base["yte"]
    i_tr = i_tr[_strat_sub(y[i_tr], CFG.n_train_max, rng, CFG.balance_train)]
    i_va = i_va[_strat_sub(y[i_va], CFG.n_val_max, rng, False)]
    j_te = _strat_sub(yte, CFG.n_test_max, rng, False)
    Xte_df, yte = Xte_df.iloc[j_te].reset_index(drop=True), yte[j_te]
    assert len(np.intersect1d(i_tr, i_va)) == 0, "train/val overlap"
    if name == "kddcup99": assert len(np.intersect1d(np.concatenate([i_tr, i_va]), i_te[j_te])) == 0, "train/test overlap"
    Xtr_df, Xva_df = X.iloc[i_tr].reset_index(drop=True), X.iloc[i_va].reset_index(drop=True)
    ytr, yva = y[i_tr], y[i_va]
    prep = Prep().fit(Xtr_df, base["cat"])                      # TRAIN ONLY
    htr = set(pd.util.hash_pandas_object(Xtr_df, index=False).values)
    n_ident = int(np.isin(pd.util.hash_pandas_object(Xte_df, index=False).values, list(htr)).sum())
    if CFG.dedup and name == "kddcup99": assert n_ident == 0, "identical rows in train and test despite dedup"
    if n_ident: log(f"  note [{name} s{seed}]: {n_ident} test rows have an identical feature row in train (dedup={CFG.dedup})")
    S = Split(name, seed, prep.transform(Xtr_df), ytr, prep.transform(Xva_df), yva, prep.transform(Xte_df), yte, prep,
              dict(n_train=len(ytr), n_val=len(yva), n_test=len(yte), d_full=int(prep.sc.n_features_in_),
                   train_pos=float(ytr.mean()), val_pos=float(yva.mean()), test_pos=float(yte.mean()),
                   dropped_constant_cols=len(prep.dropped_const), log1p_cols=len(prep.logc), test_rows_identical_to_train=n_ident,
                   pca_var_explained_Qmax=float(prep.pca.explained_variance_ratio_.sum())))
    assert set(np.unique(ytr)) <= {0, 1} and set(np.unique(yte)) <= {0, 1}
    SPLITS[key] = S
    return S

def dataset_statistics():
    rows = []
    for ds in CFG.datasets:
        b = get_base(ds)
        if b is None: continue
        S = get_split(ds, CFG.seeds[0])
        rows.append(dict(dataset=ds, **{k: (v if not isinstance(v, dict) else json.dumps(jsonable(v), default=str)) for k, v in b["stats"].items() if k != "source"},
                         **{f"split_{k}": v for k, v in S.info.items()}))
    return pd.DataFrame(rows)

AVAILABLE = []
for _ds in CFG.datasets:
    if get_base(_ds) is not None: AVAILABLE.append(_ds)
log("Datasets available:", AVAILABLE)
DS_STATS = dataset_statistics() if AVAILABLE else pd.DataFrame()
if len(DS_STATS): display(DS_STATS.T) if "display" in globals() else print(DS_STATS.T)

MODE_STR = "QUICK" if CFG.quick else "FULL"
def data_source(ds):
    """REAL (downloaded by the loader chain) | MANUAL_UPLOAD (user-supplied file) | SYNTHETIC_FALLBACK (never produced by this notebook; flagged if a source string says so)."""
    s = str(SOURCES.get("kddcup99" if ds == "kddcup99" else "unsw_files", "")).lower()
    if "synthetic" in s: return "SYNTHETIC_FALLBACK"
    if any(u.lower() in s for u in UPLOADED): return "MANUAL_UPLOAD"
    return "REAL"
def OVERALL_SOURCE():
    return "+".join(f"{d}:{data_source(d)}" for d in AVAILABLE) or "NONE"
def print_run_banner():
    print("=" * 78); print(f"MODE: {MODE_STR} | seeds: {list(CFG.seeds)} (secondary experiments: {list(CFG.seeds_small)})")
    for d in CFG.datasets: print(f"  dataset {d}: {'NOT LOADED' if d not in AVAILABLE else data_source(d)}  <- {SOURCES.get('kddcup99' if d == 'kddcup99' else 'unsw_files', 'n/a')}")
    print("library versions:\n  " + ENV_TXT.replace("\n", "\n  "))
    if any(data_source(d) == "SYNTHETIC_FALLBACK" for d in AVAILABLE):
        print("#" * 78); print("#  RESULTS FROM SYNTHETIC DATA MUST NOT BE USED IN THE PAPER"); print("#" * 78)
    if CFG.quick: print("QUICK_MODE smoke test: small subsamples / few rounds. Do NOT quote these numbers; run FULL mode for the paper.")
    print("=" * 78)
print_run_banner()

# %% [markdown]
# ## Cell 6 — Circuit builder, batched statevector simulator (NumPy or PyTorch backend), gate statistics, density-matrix noise simulator
# * One **op list** (`Circ.ops`) is the single source of truth: the fast simulator, the dense NumPy reference and the PennyLane circuit are all generated from it.
# * Qubit 0 is the most-significant bit (PennyLane convention). Encoding = RY(x_j) on |0> (a product state, built directly).
# * Pooling = controlled-RY from each discarded qubit onto its nearest kept qubit, after which the discarded qubits are traced out
#   (pure-state simulation: measuring a kept-qubit observable uses the marginal distribution, i.e. the partial trace).

# %%
@dataclass
class Circ:
    kind: str; Q: int; L: int; Qp: int; gateset: str; entangle: str; share: bool
    ops: list; n: int; kept: list; obs: int

class _B:
    def __init__(s): s.ops = []; s.n = 0
    def p(s): s.n += 1; return s.n - 1
    def add(s, name, w, p=-1): s.ops.append((name, tuple(w), p))

def _rot(b, qs, gs):
    for q in qs:
        b.add("RY", (q,), b.p())
        if gs == "full": b.add("RZ", (q,), b.p())

def _ent_all(b, A, entangle):
    if len(A) < 2: return
    if entangle == "full":
        for i in range(len(A)):
            for j in range(i + 1, len(A)): b.add("CNOT", (A[i], A[j]))
    elif len(A) == 2: b.add("CNOT", (A[0], A[1]))
    else:
        for i in range(len(A)): b.add("CNOT", (A[i], A[(i + 1) % len(A)]))

def _conv_layer(b, A, gs, entangle, share):
    if entangle == "full" or len(A) < 2:
        _rot(b, A, gs); _ent_all(b, A, entangle); return
    shared = {}
    def get(key):
        if not share: return b.p()
        if key not in shared: shared[key] = b.p()
        return shared[key]
    for start in (0, 1):
        for i in range(start, len(A) - 1, 2):
            a, c = A[i], A[i + 1]
            b.add("RY", (a,), get("a1")); b.add("RY", (c,), get("b1"))
            if gs == "full": b.add("RZ", (a,), get("az")); b.add("RZ", (c,), get("bz"))
            b.add("CNOT", (a, c))
            b.add("RY", (a,), get("a2")); b.add("RY", (c,), get("b2"))

def build_circuit(kind, Q, L, Qp=None, gateset="real", entangle="ring", share=True) -> Circ:
    b = _B(); W = list(range(Q))
    if kind == "vqnn":
        for _ in range(L):
            _rot(b, W, gateset); _ent_all(b, W, entangle)
        kept, obs, Qp = W, 0, Q
    elif kind == "ivqnn":
        assert Qp is not None and 1 <= Qp < Q, "need 1 <= Q' < Q"
        K = sorted(set(int(round(v)) for v in np.linspace(0, Q - 1, Qp))); assert len(K) == Qp
        D = [q for q in W if q not in K]
        _conv_layer(b, W, gateset, entangle, share)
        sp = None
        for d in D:
            k = min(K, key=lambda z: (abs(z - d), z))
            if share:
                if sp is None: sp = (b.p(), b.p() if gateset == "full" else -1)
                p1, p2 = sp
            else:
                p1, p2 = b.p(), (b.p() if gateset == "full" else -1)
            b.add("CRY", (d, k), p1)
            if gateset == "full": b.add("CRZ", (d, k), p2)
        for _ in range(L - 1): _conv_layer(b, K, gateset, entangle, share)
        kept, obs = K, K[0]
    else:
        raise ValueError(kind)
    return Circ(kind, Q, L, Qp, gateset, entangle, share, b.ops, b.n, kept, obs)

def circ_stats(c: Circ) -> dict:
    n1 = sum(1 for o in c.ops if len(o[1]) == 1); free = [0] * c.Q
    for _, w, _ in c.ops:
        d = max(free[q] for q in w) + 1
        for q in w: free[q] = d
    return dict(n_circuit_params=c.n, n_gates=len(c.ops), n_1q=n1, n_2q=len(c.ops) - n1, depth=max(free) + 1, enc_gates=c.Q)

def sim_flops(c: Circ) -> float:
    """Rough classical-simulation FLOP count per sample (documented estimate): 1q real gate 3*2^Q, complex 12*2^Q/2, CNOT 2^Q, controlled rot 4*2^Q, encoding Q*2^Q, readout 3*2^Q."""
    N = 2 ** c.Q; f = c.Q * N + 3 * N
    for name, w, _ in c.ops:
        f += {"RY": 3 * N, "RZ": 6 * N, "CNOT": N, "CRY": 4 * N, "CRZ": 7 * N}[name]
    return float(f)

# ---------------------------------------------------------------- generic simulator primitives (xp = numpy or torch)
_PERM, _MASK = {}, {}
def _perm(nax, c, t, xp):
    key = (nax, c, t, xp.__name__)
    if key not in _PERM:
        idx = np.arange(2 ** nax); cb = (idx >> (nax - 1 - c)) & 1
        p = np.where(cb == 1, idx ^ (1 << (nax - 1 - t)), idx)
        _PERM[key] = p if xp is np else torch.as_tensor(p, dtype=torch.long)
    return _PERM[key]

def _cmask(nax, c, xp):
    key = (nax, c, xp.__name__)
    if key not in _MASK:
        m = ((np.arange(2 ** nax) >> (nax - 1 - c)) & 1).astype(bool)
        _MASK[key] = m if xp is np else torch.as_tensor(m)
    return _MASK[key]

def _ap1(st, g, q, nax, xp):
    P, n, _ = st.shape; A, B = 2 ** q, 2 ** (nax - q - 1)
    s = st.reshape(P, n, A, 2, B); s0, s1 = s[:, :, :, 0, :], s[:, :, :, 1, :]
    return xp.stack([g[0] * s0 + g[1] * s1, g[2] * s0 + g[3] * s1], 3).reshape(P, n, -1)

def _apd(st, e0, e1, q, nax, xp):
    P, n, _ = st.shape; A, B = 2 ** q, 2 ** (nax - q - 1)
    s = st.reshape(P, n, A, 2, B)
    return xp.stack([e0 * s[:, :, :, 0, :], e1 * s[:, :, :, 1, :]], 3).reshape(P, n, -1)

def _apply_op(st, name, w, th, nax, xp, conj=False):
    if name == "CNOT": return st[..., _perm(nax, w[0], w[1], xp)]
    h = th * 0.5; c = xp.cos(h)[:, None, None, None]; s = xp.sin(h)[:, None, None, None]; tgt = w[-1]
    if name in ("RY", "CRY"): out = _ap1(st, (c, -s, s, c), tgt, nax, xp)
    else:
        e0, e1 = (c + 1j * s, c - 1j * s) if conj else (c - 1j * s, c + 1j * s)
        out = _apd(st, e0, e1, tgt, nax, xp)
    if name in ("CRY", "CRZ"): out = xp.where(_cmask(nax, w[0], xp), out, st)
    return out

def _product_state(X, Q):
    X = np.asarray(X, dtype=np.float64); n = X.shape[0]; psi = np.ones((n, 1))
    for j in range(Q):
        v = np.stack([np.cos(X[:, j] / 2), np.sin(X[:, j] / 2)], 1)
        psi = (psi[:, :, None] * v[:, None, :]).reshape(n, -1)
    return psi

class Sim:
    """Batched statevector simulator. expval(theta (P,n_circ), X (n,Q)) -> (P,n) of <Z_obs>. xp = numpy or torch (autograd works with torch)."""
    def __init__(self, circ: Circ, xp=np, double=False):
        self.c, self.xp, self.cplx, self.double = circ, xp, circ.gateset != "real", double
        if xp is np: self.fdt, self.cdt = (np.float64, np.complex128) if double else (np.float32, np.complex64)
        else: self.fdt, self.cdt = (torch.float64, torch.complex128) if double else (torch.float32, torch.complex64)
        z = 1 - 2 * ((np.arange(2 ** circ.Q) >> (circ.Q - 1 - circ.obs)) & 1)
        self.z = self._cx(z.astype(np.float64), False)
    def _cx(self, a, cplx):
        dt = self.cdt if cplx else self.fdt
        return a.astype(dt) if self.xp is np else torch.as_tensor(a, dtype=dt)
    def expval(self, theta, X):
        xp = self.xp; Q = self.c.Q; X = np.asarray(X)
        theta = theta.astype(self.fdt) if xp is np else theta.to(self.fdt)
        P, n = theta.shape[0], X.shape[0]
        psi = self._cx(_product_state(X, Q), self.cplx)
        one = self._cx(np.ones((P, 1, 1)), self.cplx)
        st = psi[None] * one
        for name, w, p in self.c.ops:
            st = _apply_op(st, name, w, theta[:, p] if p >= 0 else None, Q, xp)
        pr = abs(st) ** 2 if self.cplx else st * st
        return (pr * self.z).sum(-1)

def _depol(st, q, Q, lam):
    P, n, _ = st.shape; a, b_ = 2 ** q, 2 ** (Q - q - 1)
    r = st.reshape(P, n, a, 2, b_, a, 2, b_)
    tr = r[:, :, :, 0, :, :, 0, :] + r[:, :, :, 1, :, :, 1, :]
    I = np.eye(2).reshape(1, 1, 1, 2, 1, 1, 2, 1)
    new = (1 - lam) * r + lam * 0.5 * (tr[:, :, :, None, :, :, None, :] * I)
    return new.reshape(P, n, -1)

def density_expval(c: Circ, theta, X, p_dep: float):
    """Density-matrix simulation with single-qubit depolarising noise (prob p_dep) after every gate on each qubit it touches. Small Q only."""
    Q = c.Q; theta = np.asarray(theta, dtype=np.float64)[None]; cplx = c.gateset != "real"
    psi = _product_state(X, Q); n = len(psi)
    st = (psi[:, :, None] * psi[:, None, :]).reshape(1, n, -1).astype(np.complex128 if cplx else np.float64)
    lam = 4.0 * p_dep / 3.0
    for name, w, p in c.ops:
        th = theta[:, p] if p >= 0 else None
        st = _apply_op(st, name, w, th, 2 * Q, np, conj=False)
        st = _apply_op(st, name, tuple(q + Q for q in w), th, 2 * Q, np, conj=True)
        if lam > 0:
            for q in w: st = _depol(st, q, Q, lam)
    diag = st[..., np.arange(2 ** Q) * (2 ** Q + 1)]
    z = 1 - 2 * ((np.arange(2 ** Q) >> (Q - 1 - c.obs)) & 1)
    return np.real((diag * z).sum(-1))[0]

# ---------------------------------------------------------------- dense reference + PennyLane (validation only)
def dense_expval(c: Circ, theta, x) -> float:
    Q = c.Q; I2 = np.eye(2)
    def emb(g, q):
        out = np.array([[1.0]])
        for j in range(Q): out = np.kron(out, g if j == q else I2)
        return out
    ry = lambda t: np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
    rz = lambda t: np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
    X_ = np.array([[0, 1], [1, 0]]); P0, P1 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
    psi = np.array([1.0])
    for j in range(Q): psi = np.kron(psi, [np.cos(x[j] / 2), np.sin(x[j] / 2)])
    psi = psi.astype(complex)
    for name, w, p in c.ops:
        if name == "RY": U = emb(ry(theta[p]), w[0])
        elif name == "RZ": U = emb(rz(theta[p]), w[0])
        elif name == "CNOT": U = emb(P0, w[0]) + emb(P1, w[0]) @ emb(X_, w[1])
        elif name == "CRY": U = emb(P0, w[0]) + emb(P1, w[0]) @ emb(ry(theta[p]), w[1])
        elif name == "CRZ": U = emb(P0, w[0]) + emb(P1, w[0]) @ emb(rz(theta[p]), w[1])
        psi = U @ psi
    return float(np.real(psi.conj() @ emb(np.diag([1.0, -1.0]), c.obs) @ psi))

def pl_expval(c: Circ, theta, X):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=c.Q)
    @qml.qnode(dev)
    def f(x, th):
        for j in range(c.Q): qml.RY(x[j], wires=j)
        for name, w, p in c.ops:
            if name == "RY": qml.RY(th[p], wires=w[0])
            elif name == "RZ": qml.RZ(th[p], wires=w[0])
            elif name == "CNOT": qml.CNOT(wires=list(w))
            elif name == "CRY": qml.CRY(th[p], wires=list(w))
            elif name == "CRZ": qml.CRZ(th[p], wires=list(w))
        return qml.expval(qml.PauliZ(c.obs))
    return np.array([float(f(x, theta)) for x in X])

# %% [markdown]
# ## Cell 7 — Self-tests: simulator vs dense reference vs PennyLane (max abs error < 1e-6), torch/NumPy agreement, noise-sim sanity

# %%
def validate_simulator(n_circ=10):
    rng = np.random.default_rng(123); worst_dense = worst_pl = worst_torch = 0.0; pl_ok = None; rows = []
    for i in range(n_circ):
        kind = ["vqnn", "ivqnn"][i % 2]; Q = int(rng.choice([3, 4, 5, 6])); L = int(rng.integers(1, 4))
        gs = ["real", "full"][(i // 2) % 2]; ent = ["ring", "full"][(i // 4) % 2]; sh = bool(i % 3)
        c = build_circuit(kind, Q, L, max(1, Q // 2) if kind == "ivqnn" else None, gs, ent, sh)
        th = rng.uniform(-np.pi, np.pi, (1, c.n)); X = rng.uniform(0, np.pi, (3, Q))
        ref = np.array([dense_expval(c, th[0], x) for x in X])
        got = Sim(c, np, double=True).expval(th, X)[0]
        e_d = float(np.abs(got - ref).max()); worst_dense = max(worst_dense, e_d)
        e_t = None
        if HAS_TORCH:
            gt = Sim(c, torch, double=True).expval(torch.as_tensor(th), X)[0].numpy()
            e_t = float(np.abs(gt - ref).max()); worst_torch = max(worst_torch, e_t)
        e_p = None
        try:
            pl = pl_expval(c, th[0], X); e_p = float(np.abs(pl - ref).max()); worst_pl = max(worst_pl, e_p); pl_ok = True
        except Exception as e:
            if pl_ok is None: pl_ok = False; log("PennyLane check skipped:", type(e).__name__, str(e)[:100])
        rows.append((kind, Q, L, gs, ent, sh, c.n, e_d, e_t, e_p))
    assert worst_dense < 1e-6, f"simulator != dense reference (max err {worst_dense})"
    if HAS_TORCH: assert worst_torch < 1e-6, f"torch backend != reference ({worst_torch})"
    if pl_ok: assert worst_pl < 1e-6, f"simulator != PennyLane ({worst_pl})"
    pl_txt = f"{worst_pl:.2e}" if pl_ok else "NOT RUN (PennyLane unavailable)"
    tc_txt = f"{worst_torch:.2e}" if HAS_TORCH else "n/a"
    log(f"SIM VALIDATION PASSED: max|err| vs dense = {worst_dense:.2e}; vs torch-backend = {tc_txt}; vs PennyLane = {pl_txt}")
    if not pl_ok: not_computed("pennylane_validation", "PennyLane import/execution failed; simulator validated against an independent dense-matrix NumPy reference only")
    # float32 working precision vs float64
    c = build_circuit("ivqnn", 8, 3, 4, "real", "ring", True); th = rng.uniform(-3, 3, (4, c.n)); X = rng.uniform(0, np.pi, (16, 8))
    e32 = float(np.abs(Sim(c, np).expval(th, X) - Sim(c, np, True).expval(th, X)).max())
    # noise-sim sanity
    c = build_circuit("ivqnn", 4, 2, 2, "real", "ring", True); th = rng.uniform(-3, 3, c.n); X = rng.uniform(0, np.pi, (5, 4))
    d0 = density_expval(c, th, X, 0.0); sv = Sim(c, np, True).expval(th[None], X)[0]
    assert np.abs(d0 - sv).max() < 1e-9, "density sim (p=0) != statevector"
    assert np.abs(density_expval(c, th, X, 0.3)).max() <= 1 + 1e-9, "noisy expectation out of [-1,1]"
    log(f"float32-vs-float64 max err (Q=8) = {e32:.2e}; density-matrix sim at p=0 matches statevector.")
    return rows, dict(max_err_dense=worst_dense, max_err_torch=worst_torch if HAS_TORCH else None, max_err_pennylane=worst_pl if pl_ok else None, float32_err=e32)

SIM_ROWS, SIM_VALID = validate_simulator()
print(pd.DataFrame(SIM_ROWS, columns=["kind", "Q", "L", "gateset", "entangle", "share", "n_params", "err_dense", "err_torch", "err_pennylane"]))

# circuit-specification table (pasted into the paper)
def circuit_spec_table(Qs=(4, 6, 8, 10), Ls=(1, 2, 3, 4)):
    rows = []
    for Q in Qs:
        for L in Ls:
            for kind in ("vqnn", "ivqnn"):
                c = build_circuit(kind, Q, L, Q // 2 if kind == "ivqnn" else None, CFG.gateset, CFG.entangle, CFG.conv_share)
                rows.append(dict(model=kind.upper(), Q=Q, L=L, Qp=c.Qp, **circ_stats(c), n_params_with_head=c.n + (2 if CFG.head == "affine" else 0)))
    return pd.DataFrame(rows)
CIRCUIT_TABLE = circuit_spec_table()
print(CIRCUIT_TABLE[(CIRCUIT_TABLE.Q == 8) & (CIRCUIT_TABLE.L == 3)])

# %% [markdown]
# ## Cell 8 — Models (quantum, small MLP), metrics (acc/prec/rec/F1/MCC/AUC/PR-AUC/Brier/ECE), aggregation and paired significance tests

# %%
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, roc_curve, precision_recall_curve)

def _toxp(a, xp):
    return a if xp is np else torch.as_tensor(a)

class QModel:
    """Variational quantum classifier/regressor: p = head(<Z>). Parameter vector = [circuit angles, (scale s, bias b)]."""
    quantum = True
    def __init__(self, circ: Circ, cfg: Cfg, task="bce"):
        self.circ, self.head, self.gain, self.task, self.init_std = circ, cfg.head, cfg.head_gain, task, cfg.init_std
        self.n_circ = circ.n; self.n_params = circ.n + (2 if cfg.head == "affine" else 0)
        self.dim = 2 ** circ.Q; self.Q = circ.Q; self._sims = {}
        self.ps_factor = 2 * circ.n            # parameter-shift-equivalent executions per sample per gradient
        self.flops_fwd = sim_flops(circ)
    def _sim(self, xp):
        k = xp.__name__
        if k not in self._sims: self._sims[k] = Sim(self.circ, xp)
        return self._sims[k]
    def init(self, rng):
        th = rng.normal(0, self.init_std, self.n_circ)
        if self.head == "affine": th = np.concatenate([th, [1.0, 0.0]]) + np.concatenate([np.zeros(self.n_circ), 0.05 * rng.standard_normal(2)])
        return th.astype(np.float32)
    def probs(self, theta, X, xp):
        f = self._sim(xp).expval(theta[:, :self.n_circ], X)
        if self.head == "affine":
            z = self.gain * theta[:, self.n_circ:self.n_circ + 1] * f + theta[:, self.n_circ + 1:self.n_circ + 2]
            return 1 / (1 + xp.exp(-z))
        return (1 + f) / 2
    def loss(self, theta, X, y, xp):
        p = xp.clip(self.probs(theta, X, xp), 1e-7, 1 - 1e-7); y = _toxp(y, xp)
        if self.task == "mse": return ((p - y) ** 2).mean(-1)
        return (-(y * xp.log(p) + (1 - y) * xp.log(1 - p))).mean(-1)

class MLPModel:
    """Small tanh MLP with a flat parameter vector (same interface as QModel) — used for the classical FedAvg baseline."""
    quantum = False
    def __init__(self, d, hidden=(32, 16), task="bce"):
        self.sizes = [d, *hidden, 1]; self.task = task
        self.n_params = sum(a * b + b for a, b in zip(self.sizes[:-1], self.sizes[1:])); self.n_circ = 0
        self.dim = 1; self.Q = 0; self.ps_factor = 1
        self.flops_fwd = float(sum(2 * a * b for a, b in zip(self.sizes[:-1], self.sizes[1:])))
    def init(self, rng):
        parts = []
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            parts += [rng.normal(0, 1 / math.sqrt(a), a * b), np.zeros(b)]
        return np.concatenate(parts).astype(np.float32)
    def probs(self, theta, X, xp):
        P = theta.shape[0]; h = _toxp(np.asarray(X, dtype=np.float32), xp)[None]; o = 0
        for i, (a, b) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            W = theta[:, o:o + a * b].reshape(P, a, b); o += a * b
            bb = theta[:, o:o + b].reshape(P, 1, b); o += b
            h = h @ W + bb
            if i < len(self.sizes) - 2: h = xp.tanh(h)
        z = h[..., 0]
        return 1 / (1 + xp.exp(-z))
    def loss(self, theta, X, y, xp):
        p = xp.clip(self.probs(theta, X, xp), 1e-7, 1 - 1e-7); y = _toxp(y, xp)
        if self.task == "mse": return ((p - y) ** 2).mean(-1)
        return (-(y * xp.log(p) + (1 - y) * xp.log(1 - p))).mean(-1)

BACKEND = "numpy"
def eval_losses(model, pos, X, y, backend=None):
    """Loss for each of P parameter vectors on (X,y) -> (P,) float64 (no autograd)."""
    backend = backend or BACKEND; P, n = pos.shape[0], len(X)
    ch = max(1, int(4e6 // (n * model.dim))); out = []
    for i in range(0, P, ch):
        sub = pos[i:i + ch]
        if backend == "torch" and HAS_TORCH:
            with torch.no_grad(): l = model.loss(torch.as_tensor(sub), X, y, torch).numpy()
        else: l = model.loss(np.asarray(sub, dtype=np.float32), X, np.asarray(y, dtype=np.float32), np)
        out.append(np.asarray(l, dtype=np.float64))
    return np.concatenate(out)

def predict_proba(model, theta, X, chunk=2048):
    out = []
    for i in range(0, len(X), chunk):
        out.append(np.asarray(model.probs(np.asarray(theta, dtype=np.float32)[None], X[i:i + chunk], np))[0])
    return np.concatenate(out).astype(np.float64)

# ---------------------------------------------------------------- metrics
def ece_score(y, p, bins=15):
    y = np.asarray(y, float); p = np.asarray(p, float); e = 0.0
    idx = np.minimum((p * bins).astype(int), bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def binary_metrics(y, p, thr=0.5) -> dict:
    y = np.asarray(y).astype(int); p = np.asarray(p, float); yh = (p >= thr).astype(int)
    two = len(np.unique(y)) == 2
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0), recall=recall_score(y, yh, zero_division=0),
                f1=f1_score(y, yh, zero_division=0), f1_macro=f1_score(y, yh, average="macro", zero_division=0), mcc=matthews_corrcoef(y, yh),
                roc_auc=roc_auc_score(y, p) if two else np.nan, pr_auc=average_precision_score(y, p) if two else np.nan,
                mse=float(np.mean((p - y) ** 2)), ece=ece_score(y, p))

METRICS = ["accuracy", "precision", "recall", "f1", "f1_macro", "mcc", "roc_auc", "pr_auc", "mse", "ece"]
HIGHER_BETTER = {m: (m not in ("mse", "ece")) for m in METRICS}
COST_COLS = ["n_params", "n_circuit_params", "evals_fwd", "evals_ps", "train_time", "flops_train", "flops_infer"]

def tcrit(n): return st.t.ppf(0.975, n - 1) if n > 1 else np.nan

def aggregate(df, by, cols):
    g = df.groupby(by)[cols]
    mean, std, n = g.mean(), g.std(ddof=1), g.count()
    ci = std.mul(n.apply(lambda c: c.map(lambda k: tcrit(k) / math.sqrt(k) if k > 1 else np.nan)))
    out = mean.add_suffix("_mean").join(std.add_suffix("_std")).join(ci.add_suffix("_ci95")).join(n[[cols[0]]].rename(columns={cols[0]: "n_seeds"}))
    return out.reset_index()

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

def paired_tests(df, ds, metric, proposed, others):
    """Paired tests across seeds: Wilcoxon signed-rank + paired t-test, Cohen's dz, Holm correction over `others`.
    NOTE: with n=5 seeds the smallest exact two-sided Wilcoxon p is 0.0625, so it can never be < 0.05; the t-test is reported as well."""
    sub = df[df.dataset == ds]; a = sub[sub.model == proposed].set_index("seed")[metric]; rows = []
    for o in others:
        b = sub[sub.model == o].set_index("seed")[metric]; common = a.index.intersection(b.index)
        if len(common) < 3: continue
        d = (a[common] - b[common]).values.astype(float)
        try: pw = st.wilcoxon(a[common], b[common]).pvalue if np.any(d != 0) else 1.0
        except Exception: pw = np.nan
        try: pt = st.ttest_rel(a[common], b[common]).pvalue
        except Exception: pt = np.nan
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
        rows.append(dict(dataset=ds, metric=metric, proposed=proposed, baseline=o, n=len(common), mean_diff=d.mean(), p_wilcoxon=pw, p_ttest=pt, cohen_dz=dz))
    if rows:
        for key in ("p_wilcoxon", "p_ttest"):
            ps = np.array([r[key] for r in rows], float); ok = ~np.isnan(ps)
            adj = np.full(len(ps), np.nan)
            if ok.any(): adj[ok] = holm(ps[ok])
            for r, v in zip(rows, adj): r[key + "_holm"] = v
    return rows

# %% [markdown]
# ## Cell 9 — Optimisers: local PSO (paper eq. 6), Adam (autograd), SPSA; evaluation counting
# * **PSO** — velocity/position updates exactly as in the paper (per-dimension r1,r2~U[0,1]), fitness F=1/(L+eps), swarm best g* returned.
# * Curves are lists of (evals_fwd, monitor_loss, evals_ps).
# * **Evaluation counts** are reported two ways: `evals_fwd` (sample-level forward passes actually executed) and `evals_ps`
#   (hardware-equivalent circuit executions; gradient methods pay 2×#circuit-params per sample per step via parameter-shift).

# %%
def init_swarm(mode, model, center, P, sigma, rng):
    n = model.n_params
    if mode == "gauss_global": pos = center[None] + sigma * rng.standard_normal((P, n))
    elif mode == "random": pos = np.stack([model.init(rng) for _ in range(P)])
    elif mode == "global_plus_random": pos = np.stack([center] + [model.init(rng) for _ in range(P - 1)])
    else: raise ValueError(mode)
    return pos.astype(np.float32)

def _draw(X, y, batch, rng):
    if not batch or batch >= len(X): return X, y
    i = rng.choice(len(X), batch, replace=False); return X[i], y[i]

def train_pso(model, X, y, center, rng, cfg: Cfg, P=None, T=None, w=None, monitor=None, mon_every=10, record=False):
    P = P or cfg.P; T = T or cfg.T; w = cfg.w if w is None else w; n = model.n_params; vmax = cfg.vmax
    pos = init_swarm(cfg.swarm_init, model, center, P, cfg.init_sigma, rng)
    vel = (rng.uniform(-1, 1, (P, n)) * 0.1 * vmax).astype(np.float32)
    xb, yb = _draw(X, y, cfg.pso_batch, rng); evals = 0
    Lq = eval_losses(model, pos, xb, yb); evals += P * len(xb)
    pbest, pbf = pos.copy(), 1.0 / (Lq + cfg.eps); k = int(np.argmax(pbf)); gbest, gbf = pbest[k].copy(), pbf[k]
    hist = [pos.copy()] if record else None; curve = []
    if monitor is not None: curve.append((0, float(eval_losses(model, gbest[None], *monitor)[0]), 0))
    for t in range(T):
        if cfg.resample_every and t > 0 and t % cfg.resample_every == 0:
            xb, yb = _draw(X, y, cfg.pso_batch, rng)
            pbf = 1.0 / (eval_losses(model, pbest, xb, yb) + cfg.eps); evals += P * len(xb)
            k = int(np.argmax(pbf)); gbest, gbf = pbest[k].copy(), pbf[k]
        r1 = rng.random((P, n), dtype=np.float32); r2 = rng.random((P, n), dtype=np.float32)
        vel = np.clip(w * vel + cfg.phi1 * r1 * (pbest - pos) + cfg.phi2 * r2 * (gbest[None] - pos), -vmax, vmax).astype(np.float32)
        pos = (pos + vel).astype(np.float32)
        Lq = eval_losses(model, pos, xb, yb); evals += P * len(xb); f = 1.0 / (Lq + cfg.eps)
        imp = f > pbf; pbest[imp] = pos[imp]; pbf[imp] = f[imp]
        k = int(np.argmax(pbf))
        if pbf[k] > gbf: gbf, gbest = pbf[k], pbest[k].copy()
        if record: hist.append(pos.copy())
        if monitor is not None and ((t + 1) % mon_every == 0 or t == T - 1):
            curve.append((evals, float(eval_losses(model, gbest[None], *monitor)[0]), evals))
    return gbest, dict(evals_fwd=evals, evals_ps=evals, curve=curve, hist=hist, final_batch_loss=float(1 / gbf - cfg.eps))

def pso_contraction(hist, gstar):
    """Empirical eta: D_t = mean_i ||p_i^t - g*||^2 with g* (final swarm best) as the proxy for Theta*."""
    D = np.array([np.mean(np.sum((h - gstar[None]) ** 2, axis=1)) for h in hist]); T = len(D) - 1
    ratios = D[1:] / np.maximum(D[:-1], 1e-30)
    eta_geo = 1 - (max(D[-1], 1e-30) / max(D[0], 1e-30)) ** (1 / max(T, 1))
    return dict(eta_geo=float(eta_geo), ratio_median=float(np.median(ratios)), violation_frac=float(np.mean(ratios >= 1.0)), D0=float(D[0]), DT=float(D[-1]))

def train_adam(model, theta0, X, y, steps, lr, batch, rng, monitor=None, mon_every=10):
    if not HAS_TORCH: raise RuntimeError("PyTorch required for Adam training")
    th = torch.tensor(np.asarray(theta0, dtype=np.float32), requires_grad=True); opt = torch.optim.Adam([th], lr=lr)
    evals = 0; curve = []
    if monitor is not None: curve.append((0, float(eval_losses(model, th.detach().numpy()[None], *monitor)[0]), 0))
    for s in range(steps):
        xb, yb = _draw(X, y, batch, rng)
        loss = model.loss(th[None], xb, torch.as_tensor(np.asarray(yb, dtype=np.float32)), torch)[0]
        opt.zero_grad(); loss.backward(); opt.step(); evals += len(xb)
        if monitor is not None and ((s + 1) % mon_every == 0 or s == steps - 1):
            curve.append((evals, float(eval_losses(model, th.detach().numpy()[None], *monitor)[0]), evals * model.ps_factor))
    th_np = th.detach().numpy().copy()
    return th_np, dict(evals_fwd=evals, evals_ps=evals * max(model.ps_factor, 1), curve=curve)

def train_spsa(model, theta0, X, y, steps, a, c, batch, rng, monitor=None, mon_every=10):
    th = np.asarray(theta0, dtype=np.float64).copy(); A = 0.1 * steps; evals = 0; curve = []
    for k in range(steps):
        ak, ck = a / (k + 1 + A) ** 0.602, c / (k + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], th.shape); xb, yb = _draw(X, y, batch, rng)
        l = eval_losses(model, np.stack([th + ck * delta, th - ck * delta]).astype(np.float32), xb, yb); evals += 2 * len(xb)
        th = th - ak * (l[0] - l[1]) / (2 * ck) * (1.0 / delta)
        if monitor is not None and ((k + 1) % mon_every == 0 or k == steps - 1):
            curve.append((evals, float(eval_losses(model, th.astype(np.float32)[None], *monitor)[0]), evals))
    return th.astype(np.float32), dict(evals_fwd=evals, evals_ps=evals, curve=curve)

def local_train(kind, model, center, X, y, cfg: Cfg, rng, hp, record=False, monitor=None):
    """One local training call (used by centralised runs with a long budget and by FedAvg clients with a per-round budget)."""
    if kind == "pso": return train_pso(model, X, y, center, rng, cfg, w=hp.get("pso_w", cfg.w), monitor=monitor, record=record)
    if kind == "adam": return train_adam(model, center, X, y, cfg.local_adam_steps, hp.get("adam_lr", cfg.adam_lr), cfg.adam_batch, rng, monitor)
    if kind == "spsa": return train_spsa(model, center, X, y, max(1, cfg.P * cfg.T // 2), hp.get("spsa_a", cfg.spsa_a), cfg.spsa_c, cfg.adam_batch, rng, monitor)
    raise ValueError(kind)

# %% [markdown]
# ## Cell 10 — Federated simulation: partitioning (IID / Dirichlet label-skew, unequal sizes), weighted FedAvg (n_c/N), per-round logging

# %%
def partition_clients(y, C, scheme, alpha, rng, min_size=30):
    n = len(y); idx = np.arange(n); min_size = min(min_size, max(2, n // (4 * C)))
    if scheme == "iid":
        for _ in range(100):
            p = rng.dirichlet(5.0 * np.ones(C)); p = np.maximum(p, 0.3 / C); p /= p.sum()
            sizes = np.floor(p * n).astype(int); sizes[-1] = n - sizes[:-1].sum()
            if sizes.min() >= min_size: break
        perm = rng.permutation(idx); cuts = np.cumsum(sizes)[:-1]
        return [np.sort(a) for a in np.split(perm, cuts)]
    for attempt in range(300):
        parts = [[] for _ in range(C)]
        for k in np.unique(y):
            ik = rng.permutation(idx[y == k]); pk = rng.dirichlet(alpha * np.ones(C))
            cuts = (np.cumsum(pk) * len(ik)).astype(int)[:-1]
            for c, chunk in enumerate(np.split(ik, cuts)): parts[c] += chunk.tolist()
        if min(len(p) for p in parts) >= min_size: return [np.sort(np.array(p)) for p in parts]
    # fallback: move samples from the largest client so every client reaches min_size (logged)
    log(f"  partition: Dirichlet(alpha={alpha}) could not reach min_size={min_size}; rebalancing minimally")
    parts = [list(p) for p in parts]
    for c in range(C):
        while len(parts[c]) < min_size:
            big = int(np.argmax([len(p) for p in parts])); parts[c].append(parts[big].pop())
    return [np.sort(np.array(p)) for p in parts]

def fedavg_weights(sizes):
    w = np.asarray(sizes, float) / np.sum(sizes); assert abs(w.sum() - 1.0) < 1e-9; return w

def run_federated(model, clients, theta0, kind, cfg: Cfg, hp, rng, val, record_eta=False):
    """clients: list of (X_c, y_c). Returns final global theta and a log with per-round global loss/accuracy and costs."""
    sizes = [len(c[1]) for c in clients]; wts = fedavg_weights(sizes)
    sub = []
    for Xc, yc in clients:
        i = rng.choice(len(yc), min(cfg.eval_cap, len(yc)), replace=False); sub.append((Xc[i], yc[i].astype(np.float32)))
    Xv, yv = val[0][:2000], val[1][:2000].astype(np.float32)
    def gloss(th): return float(sum(w * eval_losses(model, th[None], Xs, ys)[0] for w, (Xs, ys) in zip(wts, sub)))
    def vstats(th):
        p = predict_proba(model, th, Xv); return float(np.mean((p >= 0.5) == (yv >= 0.5)))
    theta = np.asarray(theta0, dtype=np.float32).copy()
    logs = [dict(round=0, global_loss=gloss(theta), val_acc=vstats(theta), time=0.0, evals_fwd=0, evals_ps=0)]
    tot_f = tot_p = 0; tot_t = 0.0; eta_rows = []
    for r in range(1, cfg.R + 1):
        t0 = time.time(); locs = []
        for ci, (Xc, yc) in enumerate(clients):
            th_c, info = local_train(kind, model, theta, Xc, yc.astype(np.float32), cfg, rng, hp, record=record_eta)
            locs.append(th_c); tot_f += info["evals_fwd"]; tot_p += info["evals_ps"]
            if record_eta and info.get("hist") is not None: eta_rows.append(dict(round=r, client=ci, **pso_contraction(info["hist"], th_c)))
        theta = np.tensordot(wts, np.stack(locs), 1).astype(np.float32)        # FedAvg: sum_c (n_c/N) Theta_c
        dt = time.time() - t0; tot_t += dt
        logs.append(dict(round=r, global_loss=gloss(theta), val_acc=vstats(theta), time=dt, evals_fwd=tot_f, evals_ps=tot_p))
    return theta, dict(rounds=logs, eta=eta_rows, train_time=tot_t, evals_fwd=tot_f, evals_ps=tot_p, client_sizes=sizes, weights=wts.tolist())

# ---- unit-style tests
def _test_partitions_fedavg():
    rng = np.random.default_rng(0); y = (rng.random(2000) < 0.4).astype(int)
    for scheme, al in (("iid", None), ("dirichlet", 0.1), ("dirichlet", 1.0)):
        parts = partition_clients(y, 5, scheme, al, rng)
        allidx = np.concatenate(parts); assert len(allidx) == len(y) and len(np.unique(allidx)) == len(y), "partition not a partition"
        w = fedavg_weights([len(p) for p in parts]); assert abs(w.sum() - 1) < 1e-12
    th = [np.ones(3), 3 * np.ones(3)]; w = fedavg_weights([1, 3]); assert np.allclose(np.tensordot(w, np.stack(th), 1), 2.5)
    log("partition / FedAvg unit tests passed")
_test_partitions_fedavg()

# %% [markdown]
# ## Cell 11 — Run helpers: tuning (equal budget for every method), quantum/classical/federated runners with per-run checkpointing

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
try:
    import xgboost as _xgb; HAS_XGB = True
except Exception:
    HAS_XGB = False

F32 = np.float32
def _fp(cfg: Cfg, hp: dict) -> str:
    keys = ["Q", "L", "Qp", "gateset", "entangle", "conv_share", "head", "head_gain", "init_std", "P", "T", "w", "phi1", "phi2", "vmax", "pso_batch",
            "swarm_init", "init_sigma", "resample_every", "adam_batch", "local_adam_steps", "spsa_c", "C", "R", "partition", "alpha", "n_train_max",
            "n_val_max", "n_test_max", "balance_train", "dedup", "log1p", "eval_cap"]
    d = {k: getattr(cfg, k) for k in keys}; d["hp"] = {k: hp.get(k) for k in ("adam_lr", "pso_w", "spsa_a", "fmlp_lr")}
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]

def model_label(arch, opt, fed, over):
    nm = ("F-" if fed else "") + arch.upper() + "-" + {"pso": "PSO", "adam": "Adam", "spsa": "SPSA"}[opt]
    suf = ",".join(f"{k}={v}" for k, v in sorted(over.items()))
    return nm + (f"[{suf}]" if suf else "")

def save_preds(ds, label, seed, y, p):
    safe = re.sub(r"[^A-Za-z0-9_.=,\-\[\]]+", "_", label)
    np.savez_compressed(ROOT / "preds" / f"{ds}__{safe}__{'q' if CFG.quick else 'f'}__s{seed}.npz", y=np.asarray(y, dtype=np.int8), p=np.asarray(p, dtype=np.float32))

def load_preds(ds, label, seed):
    safe = re.sub(r"[^A-Za-z0-9_.=,\-\[\]]+", "_", label)
    f = ROOT / "preds" / f"{ds}__{safe}__{'q' if CFG.quick else 'f'}__s{seed}.npz"
    if not f.exists(): return None
    z = np.load(f); return z["y"], z["p"]

def _val_loss(model, th, S, Xva):
    return float(eval_losses(model, np.asarray(th, dtype=F32)[None], Xva[:1000], S.yva[:1000].astype(F32))[0])

def get_hp(ds):
    """Hyper-parameter tuning on the VALIDATION split of seed CFG.seeds[0]: exactly 3 candidates per method, same training budget definitions."""
    def _do():
        S = get_split(ds, CFG.seeds[0]); Xtr, Xva, _ = S.q(CFG.Q); ytr = S.ytr.astype(F32)
        circ = build_circuit("ivqnn", CFG.Q, CFG.L, CFG.Qp, CFG.gateset, CFG.entangle, CFG.conv_share); model = QModel(circ, CFG)
        th0 = model.init(rng_for("tune_init", ds)); hp, tlog = {}, []
        def pick(name, grid, fn):
            vals = []
            for g in grid:
                try: vals.append(fn(g))
                except Exception as e: vals.append(np.inf); log(f"  tune {name}={g} failed: {type(e).__name__}: {str(e)[:80]}")
                tlog.append(dict(method=name, value=str(g), val_loss=float(vals[-1])))
            return grid[int(np.argmin(vals))]
        hp["adam_lr"] = pick("adam_lr", CFG.grid_adam_lr, lambda lr: _val_loss(model, train_adam(model, th0, Xtr, ytr, CFG.tune_adam_steps, lr, CFG.adam_batch, rng_for("tune", "adam", lr, ds))[0], S, Xva)) if HAS_TORCH else CFG.adam_lr
        hp["pso_w"] = pick("pso_w", CFG.grid_pso_w, lambda w: _val_loss(model, train_pso(model, Xtr, ytr, th0, rng_for("tune", "pso", w, ds), CFG, T=CFG.tune_pso_T, w=w)[0], S, Xva))
        hp["spsa_a"] = pick("spsa_a", CFG.grid_spsa_a, lambda a: _val_loss(model, train_spsa(model, th0, Xtr, ytr, max(1, CFG.P * CFG.tune_pso_T // 2), a, CFG.spsa_c, CFG.adam_batch, rng_for("tune", "spsa", a, ds))[0], S, Xva))
        for v in ("Q", "full"):
            Xa, Xb = (S.q(CFG.Q)[0], S.q(CFG.Q)[1]) if v == "Q" else (S.Xtr, S.Xva)
            hp[f"LR-{v}"] = pick(f"LR-{v}_C", CFG.grid_lr_C, lambda C: log_loss(S.yva, LogisticRegression(C=C, max_iter=300).fit(Xa, S.ytr).predict_proba(Xb)[:, 1], labels=[0, 1]))
            hp[f"MLP-{v}"] = list(pick(f"MLP-{v}_hidden", CFG.grid_mlp, lambda h: log_loss(S.yva, MLPClassifier(hidden_layer_sizes=h, max_iter=150, random_state=0).fit(Xa, S.ytr).predict_proba(Xb)[:, 1], labels=[0, 1])))
        if HAS_TORCH:
            fm = MLPModel(CFG.Q); f0 = fm.init(rng_for("tune_fmlp", ds))
            hp["fmlp_lr"] = pick("fmlp_lr", CFG.grid_adam_lr, lambda lr: _val_loss(fm, train_adam(fm, f0, Xtr, ytr, CFG.tune_adam_steps, lr, CFG.adam_batch, rng_for("tune", "fmlp", lr, ds))[0], S, Xva))
        else: hp["fmlp_lr"] = CFG.adam_lr
        hp["tuning_log"] = tlog
        return hp
    return cached(f"HP|{ds}|{CFG.Q}|{CFG.P}|{CFG.tune_pso_T}|{CFG.tune_adam_steps}|{CFG.n_train_max}|{CFG.pso_batch}", _do)

def _fill_cost(res, model, circ, opt, info, tt, fed):
    res.update(n_params=model.n_params, n_circuit_params=model.n_circ, evals_fwd=info["evals_fwd"], evals_ps=info["evals_ps"], train_time=tt,
               flops_infer=model.flops_fwd, flops_train=info["evals_fwd"] * model.flops_fwd * (3 if opt == "adam" else 1))
    if circ is not None: res.update({f"circ_{k}": v for k, v in circ_stats(circ).items()})

def qrun(ds, seed, arch, opt, fed, over=None):
    """Train + test one quantum model (centralised or federated). Cached per (dataset, seed, architecture, optimiser, federated, overrides, config fingerprint)."""
    over = dict(over or {}); cfg = replace(CFG, **over); hp = get_hp(ds); label = model_label(arch, opt, fed, over)
    key = f"Q|{ds}|{label}|s{seed}|{_fp(cfg, hp)}"
    def _do():
        S = get_split(ds, seed); Xtr, Xva, Xte = S.q(cfg.Q); ytr = S.ytr.astype(F32)
        circ = build_circuit(arch, cfg.Q, cfg.L, cfg.Qp if arch == "ivqnn" else None, cfg.gateset, cfg.entangle, cfg.conv_share); model = QModel(circ, cfg)
        th0 = model.init(rng_for("init", ds, seed, arch)); rng = rng_for("train", ds, seed, label); mon = (Xtr[:512], ytr[:512])
        extra = {}; t0 = time.time()
        if not fed:
            if opt == "pso":
                cc = replace(cfg, resample_every=cfg.resample_every or cfg.T)
                th, info = train_pso(model, Xtr, ytr, th0, rng, cc, T=cfg.R * cfg.T, w=hp["pso_w"], monitor=mon, record=True)
                extra["eta_central"] = pso_contraction(info["hist"], th); info.pop("hist")
            elif opt == "adam": th, info = train_adam(model, th0, Xtr, ytr, cfg.R * cfg.local_adam_steps, hp["adam_lr"], cfg.adam_batch, rng, mon)
            else: th, info = train_spsa(model, th0, Xtr, ytr, cfg.R * max(1, cfg.P * cfg.T // 2), hp["spsa_a"], cfg.spsa_c, cfg.adam_batch, rng, mon)
            tt = time.time() - t0; extra["curve"] = info["curve"]
        else:
            parts = partition_clients(S.ytr, cfg.C, cfg.partition, cfg.alpha, rng_for("part", ds, seed, cfg.C, cfg.partition, cfg.alpha))
            clients = [(Xtr[p], ytr[p]) for p in parts]
            th, info = run_federated(model, clients, th0, opt, cfg, hp, rng, (Xva, S.yva), record_eta=(opt == "pso"))
            tt = info["train_time"]; extra.update(rounds=info["rounds"], eta=info["eta"], client_sizes=info["client_sizes"],
                                                  client_pos_frac=[float(S.ytr[p].mean()) for p in parts])
        p = predict_proba(model, th, Xte); save_preds(ds, label, seed, S.yte, p)
        res = dict(dataset=ds, seed=seed, arch=arch, opt=opt, fed=bool(fed), model=label, over=over, **binary_metrics(S.yte, p))
        _fill_cost(res, model, circ, opt, info, tt, fed); res["val_acc"] = float(np.mean((predict_proba(model, th, Xva) >= 0.5) == S.yva))
        extra["theta"] = th; res["extra"] = extra
        return res
    return cached(key, _do)

def crun(ds, seed, name):
    """Classical baselines: LR/MLP on the Q matched angle features ('-Q') or all preprocessed features ('-full'); RF / XGB on full features."""
    hp = get_hp(ds); key = f"C|{ds}|{name}|s{seed}|{CFG.n_train_max}|{hp.get(name)}|{CFG.Q}"
    def _do():
        S = get_split(ds, seed); Q = CFG.Q
        Xtr, Xva, Xte = S.q(Q) if name.endswith("-Q") else (S.Xtr, S.Xva, S.Xte)
        base = name.split("-")[0]
        if base == "LR": clf = LogisticRegression(C=hp[name], max_iter=500, random_state=seed)
        elif base == "MLP": clf = MLPClassifier(hidden_layer_sizes=tuple(hp[name]), max_iter=200, random_state=seed)
        elif base == "RF": clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
        elif base == "XGB": clf = _xgb.XGBClassifier(n_estimators=200, max_depth=6, n_jobs=-1, random_state=seed, eval_metric="logloss")
        t0 = time.time(); clf.fit(Xtr, S.ytr); tt = time.time() - t0
        p = clf.predict_proba(Xte)[:, 1]; save_preds(ds, name, seed, S.yte, p)
        d = Xtr.shape[1]; npar = nit = ffl = np.nan
        if base == "LR": npar = d + 1; ffl = 2.0 * d; nit = float(np.max(clf.n_iter_))
        elif base == "MLP":
            sizes = [d, *hp[name], 1]; npar = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:])); ffl = float(sum(2 * a * b for a, b in zip(sizes[:-1], sizes[1:]))); nit = float(clf.n_iter_)
        elif base == "RF": npar = float(sum(t.tree_.node_count for t in clf.estimators_))
        ev = nit * len(S.ytr) if nit == nit else np.nan
        res = dict(dataset=ds, seed=seed, arch="classical", opt=base, fed=False, model=name, over={}, **binary_metrics(S.yte, p),
                   n_params=npar, n_circuit_params=0, evals_fwd=ev, evals_ps=ev, train_time=tt, flops_infer=ffl,
                   flops_train=(ev * ffl * 3 if ev == ev and ffl == ffl else np.nan), val_acc=float(clf.score(Xva, S.yva)), extra={})
        return res
    return cached(key, _do)

def fmlp_run(ds, seed):
    """Classical federated baseline: FedAvg with a small tanh-MLP on the same Q matched features, local Adam."""
    hp = get_hp(ds); cfg = CFG; key = f"FM|{ds}|s{seed}|{_fp(cfg, hp)}"
    def _do():
        S = get_split(ds, seed); Xtr, Xva, Xte = S.q(cfg.Q); ytr = S.ytr.astype(F32); model = MLPModel(cfg.Q); th0 = model.init(rng_for("init", ds, seed, "fmlp"))
        parts = partition_clients(S.ytr, cfg.C, cfg.partition, cfg.alpha, rng_for("part", ds, seed, cfg.C, cfg.partition, cfg.alpha))
        th, info = run_federated(model, [(Xtr[p], ytr[p]) for p in parts], th0, "adam", cfg, dict(adam_lr=hp["fmlp_lr"]), rng_for("train", ds, seed, "fmlp"), (Xva, S.yva))
        p = predict_proba(model, th, Xte); save_preds(ds, "F-MLP-Adam", seed, S.yte, p)
        res = dict(dataset=ds, seed=seed, arch="classical", opt="adam", fed=True, model="F-MLP-Adam", over={}, **binary_metrics(S.yte, p))
        _fill_cost(res, model, None, "adam", info, info["train_time"], True); res["val_acc"] = float(np.mean((predict_proba(model, th, Xva) >= 0.5) == S.yva))
        res["extra"] = dict(rounds=info["rounds"]); return res
    return cached(key, _do)

def safe(fn, *a, **k):
    try: return fn(*a, **k)
    except Exception as e:
        tag = f"{getattr(fn, '__name__', 'run')}{a[:4]}"
        not_computed(tag, f"{type(e).__name__}: {str(e)[:200]}"); return None

def results_df(rows):
    flat = []
    for r in rows:
        if r is None: continue
        d = {k: v for k, v in r.items() if k not in ("extra", "over")}; d["over"] = json.dumps(r.get("over", {}), sort_keys=True); flat.append(d)
    df = pd.DataFrame(flat)
    for c in METRICS + COST_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

RESULTS = {}      # experiment name -> DataFrame
def save_df(name, df):
    df = df.copy()
    df["data_source"] = df["dataset"].map(lambda d: data_source(d) if d in DATASET_NAMES else OVERALL_SOURCE()) if "dataset" in df.columns else OVERALL_SOURCE()
    df["mode"] = MODE_STR
    RESULTS[name] = df; df.to_csv(ROOT / f"{name}.csv", index=False); log(f"saved {name}.csv ({len(df)} rows)")

# %% [markdown]
# ## Cell 12 — Calibration (timing + backend choice) and runtime budget plan
# Measures the cost of one PSO iteration and one Adam step, picks the faster **validated** simulator backend (NumPy vs PyTorch; both are checked to agree),
# prints an estimated runtime per experiment, and — in FULL mode — shrinks secondary experiments if the estimate exceeds the time budget (every change is logged and exported).

# %%
T_IT, T_ADAM = 0.05, 0.05
def calibrate():
    global BACKEND, T_IT, T_ADAM
    circ = build_circuit("ivqnn", CFG.Q, CFG.L, CFG.Qp, CFG.gateset, CFG.entangle, CFG.conv_share); model = QModel(circ, CFG); rng = np.random.default_rng(0)
    pos = (model.init(rng)[None] + 0.3 * rng.standard_normal((CFG.P, model.n_params))).astype(F32)
    X = rng.uniform(0, np.pi, (CFG.pso_batch, CFG.Q)).astype(F32); y = (rng.random(CFG.pso_batch) < 0.5).astype(F32)
    times, outs = {}, {}
    for be in ["numpy"] + (["torch"] if HAS_TORCH else []):
        try:
            outs[be] = eval_losses(model, pos, X, y, be); t0 = time.time()
            for _ in range(3): eval_losses(model, pos, X, y, be)
            times[be] = (time.time() - t0) / 3
        except Exception as e: log("backend", be, "failed:", type(e).__name__, str(e)[:100])
    choice = "numpy"
    if CFG.pso_backend in ("numpy", "torch") and CFG.pso_backend in times: choice = CFG.pso_backend
    elif len(times) == 2 and np.allclose(outs["numpy"], outs["torch"], atol=1e-4): choice = min(times, key=times.get)
    BACKEND = choice; T_IT = times[choice]
    if HAS_TORCH:
        try:
            th0 = model.init(rng); Xt = np.repeat(X, 1, 0); t0 = time.time(); train_adam(model, th0, Xt, y, 3, 0.05, CFG.adam_batch, rng); T_ADAM = (time.time() - t0) / 3
        except Exception as e: log("adam timing failed:", e)
    else: T_ADAM = T_IT * 3
    log(f"calibration: backend={BACKEND} times={ {k: round(v, 4) for k, v in times.items()} } | 1 PSO iteration (P={CFG.P}, B={CFG.pso_batch}) = {T_IT:.3f}s | 1 Adam step = {T_ADAM:.3f}s")
calibrate()

def _gates(arch, Q, L): return len(build_circuit(arch, Q, L, Q // 2 if arch == "ivqnn" else None, CFG.gateset, CFG.entangle, CFG.conv_share).ops)
def est_pso(its, P=None, Q=None, L=None, arch="ivqnn"):
    P, Q, L = P or CFG.P, Q or CFG.Q, L or CFG.L
    return its * T_IT * (P / CFG.P) * 2 ** (Q - CFG.Q) * _gates(arch, Q, L) / _gates("ivqnn", CFG.Q, CFG.L)
def est_adam(steps, Q=None, L=None, arch="ivqnn"):
    Q, L = Q or CFG.Q, L or CFG.L
    return steps * T_ADAM * 2 ** (Q - CFG.Q) * _gates(arch, Q, L) / _gates("ivqnn", CFG.Q, CFG.L)

def estimates():
    R, C, T, P = CFG.R, CFG.C, CFG.T, CFG.P; n_ds = max(1, len(AVAILABLE)); sec = [d for d in CFG.sec_datasets if d in AVAILABLE] or AVAILABLE[:1]
    E = {}
    one = (est_pso(R * C * T) + est_pso(R * T, arch="ivqnn") + est_pso(R * T, arch="vqnn")
           + est_adam(R * CFG.local_adam_steps) * 2 + est_adam(R * C * CFG.local_adam_steps) * 2.5 + (10 if CFG.quick else 90))
    E["E1"] = one * len(CFG.seeds) * n_ds
    nfed = len(CFG.alpha_list) + len([c for c in CFG.fed_C_list if c != CFG.C])
    E["E2"] = len(CFG.seeds_small) * len(sec) * (sum(est_pso(R * c * T) for c in CFG.fed_C_list if c != CFG.C) + len(CFG.alpha_list) * (est_pso(R * C * T) + est_adam(R * C * CFG.local_adam_steps)))
    e3 = est_pso(R * C * T, arch="vqnn") + est_pso(R * C * (P * T // 2) * 2 / P) + 2 * est_pso(R * C * T)
    e3 += sum(est_pso(R * T, P=p) for p in CFG.sweep_P) + sum(est_pso(R * t) for t in CFG.sweep_T)
    e3 += sum(est_pso(R * T, L=l, arch=a) for l in CFG.sweep_L for a in ("ivqnn", "vqnn")) + sum(est_pso(R * T, Q=q, arch=a) for q in CFG.sweep_Q for a in ("ivqnn", "vqnn")) + est_pso(R * T)
    E["E3"] = e3 * len(CFG.seeds_small) * len(sec)
    E["E4"] = sum(2 * est_pso(CFG.bp_inits * 3 / P * CFG.bp_batch / CFG.pso_batch, Q=q) for q in CFG.bp_Q) * 1.0 + 20
    E["E5"] = 120 if CFG.quick else 480
    E["E6"] = 5; E["E7"] = 20 if CFG.quick else 60
    E["E8"] = len(CFG.seeds_small) * (est_pso(R * T) + est_pso(R * C * T) + 20)
    E["E9"] = 60 if CFG.quick else 240
    return E

def plan_budget():
    budget = CFG.time_budget_h * 3600 if not CFG.quick else 20 * 60
    changes = []
    steps = [("E2/E3/E9 use 2 seeds", lambda: setattr(CFG, "seeds_small", tuple(CFG.seeds_small[:2]))),
             ("E2/E3/E9 restricted to the first dataset", lambda: setattr(CFG, "sec_datasets", tuple(CFG.sec_datasets[:1]))),
             ("E3 sweeps trimmed (P,T -> 3 values; Q sweep without Q=10)", lambda: (setattr(CFG, "sweep_P", tuple(CFG.sweep_P[:3])), setattr(CFG, "sweep_T", tuple(CFG.sweep_T[:3])), setattr(CFG, "sweep_Q", tuple(q for q in CFG.sweep_Q if q <= 8)))),
             ("E1 uses 3 seeds", lambda: setattr(CFG, "seeds", tuple(CFG.seeds[:3])))]
    for _ in range(len(steps) + 1):
        E = estimates(); tot = sum(E.values())
        if tot <= budget or not steps: break
        nm, fn = steps.pop(0); fn(); changes.append(nm)
    E = estimates(); tot = sum(E.values())
    print(pd.Series({k: f"{v / 60:.1f} min" for k, v in E.items()}).to_string()); log(f"ESTIMATED TOTAL: {tot / 60:.0f} min (budget {budget / 60:.0f} min). Rough, from calibration timings.")
    if changes: log("BUDGET PLAN reduced the workload:", "; ".join(changes))
    if tot > budget: log("WARNING: estimate still above budget; consider QUICK_MODE or fewer seeds in Cell 2.")
    DESIGN["budget_degradations"] = changes; return E
EST = plan_budget()

# %% [markdown]
# ## Cell 13 — E1: main comparison (all models, both datasets, all seeds; test set touched once per model/seed)
# Models: LR, MLP (+RF/XGB reference), VQNN-Adam, IVQNN-Adam, IVQNN-PSO, VQNN-PSO (centralised) and F-VQNN-Adam, F-IVQNN-Adam, **F-IVQNN-PSO (proposed)**, F-MLP (classical FedAvg).

# %%
PROPOSED = "F-IVQNN-PSO"
E1_Q = [("vqnn", "adam", False), ("ivqnn", "adam", False), ("ivqnn", "pso", False), ("vqnn", "pso", False),
        ("vqnn", "adam", True), ("ivqnn", "adam", True), ("ivqnn", "pso", True)]
CLASSICAL = ["LR-Q", "MLP-Q", "LR-full", "MLP-full", "RF-full"] + (["XGB-full"] if HAS_XGB and CFG.use_xgboost else [])

def exp_E1():
    rows = []
    for ds in AVAILABLE:
        for seed in tqdm(CFG.seeds, desc=f"E1 {ds}"):
            for nm in CLASSICAL: rows.append(safe(crun, ds, seed, nm))
            for arch, opt, fed in E1_Q: rows.append(safe(qrun, ds, seed, arch, opt, fed))
            rows.append(safe(fmlp_run, ds, seed))
    df = results_df(rows); save_df("E1_main", df)
    summ = aggregate(df, ["dataset", "model"], METRICS + COST_COLS); save_df("E1_summary", summ)
    sig = []
    for ds in AVAILABLE:
        others = [m for m in df[df.dataset == ds].model.unique() if m != PROPOSED]
        for met in ("accuracy", "recall", "precision", "f1", "mse", "mcc"): sig += paired_tests(df, ds, met, PROPOSED, others)
    save_df("E1_significance", pd.DataFrame(sig))
    show = summ[["dataset", "model"] + [f"{m}_mean" for m in ("accuracy", "recall", "precision", "f1", "mse", "ece")] + ["n_seeds"]]
    print(show.round(4).to_string(index=False))
    if CFG.seeds and len(CFG.seeds) < 6: log("NOTE: with n<=5 seeds the exact two-sided Wilcoxon p-value cannot go below 0.0625; use the paired t-test columns / effect sizes.")
run_experiment("E1", exp_E1)

# %% [markdown]
# ## Cell 14 — E2: federated analysis (IID vs Dirichlet non-IID, number of clients, rounds, centralised-vs-federated gap)

# %%
def _e2_configs():
    cfgs = [("iid", {})]
    cfgs += [(f"dirichlet a={a}", dict(partition="dirichlet", alpha=a)) for a in CFG.alpha_list]
    cfgs += [(f"iid C={c}", dict(C=c)) for c in CFG.fed_C_list if c != CFG.C]
    return cfgs

def exp_E2():
    rows, curves = [], []
    for ds in [d for d in CFG.sec_datasets if d in AVAILABLE]:
        for seed in tqdm(CFG.seeds_small, desc=f"E2 {ds}"):
            for cname, over in _e2_configs():
                for opt in (("pso", "adam") if (over.get("partition") == "dirichlet" or not over) else ("pso",)):
                    r = safe(qrun, ds, seed, "ivqnn", opt, True, over)
                    if r is None: continue
                    rr = results_df([r]).iloc[0].to_dict(); rr["config"] = cname; rr["optimizer"] = opt; rows.append(rr)
                    for q in r["extra"]["rounds"]: curves.append(dict(dataset=ds, seed=seed, config=cname, optimizer=opt, **q))
                    if r["extra"].get("client_pos_frac"): rr["client_pos_frac"] = json.dumps(np.round(r["extra"]["client_pos_frac"], 3).tolist())
    df = pd.DataFrame(rows); save_df("E2_federated", df); save_df("E2_round_curves", pd.DataFrame(curves))
    if len(df):
        print(aggregate(df, ["dataset", "config", "optimizer"], ["accuracy", "f1", "mse", "train_time"]).round(4).to_string(index=False))
        e1 = RESULTS.get("E1_main")
        if e1 is not None:   # centralised vs federated gap (IID, default C), same seeds
            gap = []
            for ds in df.dataset.unique():
                for seed in CFG.seeds_small:
                    a = e1[(e1.dataset == ds) & (e1.seed == seed)].set_index("model")
                    if "IVQNN-PSO" in a.index and PROPOSED in a.index:
                        gap.append(dict(dataset=ds, seed=seed, **{f"gap_{m}": a.loc["IVQNN-PSO", m] - a.loc[PROPOSED, m] for m in ("accuracy", "f1", "mse")}))
            save_df("E2_central_vs_fed_gap", pd.DataFrame(gap))
run_experiment("E2", exp_E2)

# %% [markdown]
# ## Cell 15 — E3: ablations (pooling on/off, PSO vs Adam vs SPSA, swarm size P, iterations T, swarm initialisation, layers L, qubits Q, conv weight sharing)
# P/T/L/Q/sharing sweeps use the centralised trainer to keep runtime low (T_total = R·T); initialisation, pooling and optimiser ablations are federated.

# %%
def _e3_specs():
    S = []   # (group, setting, arch, opt, fed, over)
    S += [("pooling", "IVQNN (pool)", "ivqnn", "pso", True, {}), ("pooling", "VQNN (no pool)", "vqnn", "pso", True, {})]
    S += [("optimizer", "PSO", "ivqnn", "pso", True, {}), ("optimizer", "Adam", "ivqnn", "adam", True, {}), ("optimizer", "SPSA", "ivqnn", "spsa", True, {})]
    S += [("init", m, "ivqnn", "pso", True, dict(swarm_init=m)) for m in ("gauss_global", "random", "global_plus_random")]
    S += [("P", str(p), "ivqnn", "pso", False, dict(P=p)) for p in CFG.sweep_P]
    S += [("T", str(t), "ivqnn", "pso", False, dict(T=t)) for t in CFG.sweep_T]
    for L in CFG.sweep_L: S += [("L", f"{a} L={L}", a, "pso", False, dict(L=L)) for a in ("ivqnn", "vqnn")]
    for Q in CFG.sweep_Q: S += [("Q", f"{a} Q={Q}", a, "pso", False, dict(Q=Q, Qp=Q // 2)) for a in ("ivqnn", "vqnn")]
    S += [("conv_share", "shared", "ivqnn", "pso", False, {}), ("conv_share", "not shared", "ivqnn", "pso", False, dict(conv_share=False))]
    return S

def exp_E3():
    rows = []
    for ds in [d for d in CFG.sec_datasets if d in AVAILABLE]:
        for seed in tqdm(CFG.seeds_small, desc=f"E3 {ds}"):
            for grp, setting, arch, opt, fed, over in _e3_specs():
                r = safe(qrun, ds, seed, arch, opt, fed, over)
                if r is None: continue
                rr = results_df([r]).iloc[0].to_dict(); rr.update(group=grp, setting=setting); rows.append(rr)
    df = pd.DataFrame(rows); save_df("E3_ablations", df)
    if len(df):
        g = aggregate(df, ["dataset", "group", "setting"], ["accuracy", "f1", "mse", "n_params", "evals_fwd", "train_time"]); save_df("E3_summary", g)
        print(g[["dataset", "group", "setting", "accuracy_mean", "accuracy_std", "f1_mean", "mse_mean", "n_params_mean", "train_time_mean"]].round(4).to_string(index=False))
run_experiment("E3", exp_E3)

# %% [markdown]
# ## Cell 16 — E4: optimisation behaviour
# (a) variance of the loss and of the gradient at random initialisation vs Q (VQNN vs IVQNN; exact autograd gradients, head-free loss p=(1+f)/2).
# (b) convergence (monitor loss vs evaluations) for PSO vs Adam, from the E1 centralised runs — evaluations counted both as forward passes and as parameter-shift-equivalent circuit executions.

# %%
def exp_E4a():
    if not HAS_TORCH: return not_computed("E4a_barren_plateau", "PyTorch unavailable (autograd gradients needed)")
    ds = AVAILABLE[0]; S = get_split(ds, CFG.seeds_small[0]); rows = []
    for Q in CFG.bp_Q:
        for arch in ("vqnn", "ivqnn"):
            def _do():
                X = S.q(Q)[0][:CFG.bp_batch]; y = S.ytr[:CFG.bp_batch].astype(F32)
                c = build_circuit(arch, Q, CFG.L, Q // 2 if arch == "ivqnn" else None, CFG.gateset, CFG.entangle, CFG.conv_share)
                m = QModel(c, replace(CFG, head="linear")); rng = rng_for("bp", Q, arch)
                TH = rng.uniform(-np.pi, np.pi, (CFG.bp_inits, c.n)).astype(F32); ch = max(1, int(3e6 // (len(X) * 2 ** Q))); Ls, Gs = [], []
                for i in range(0, len(TH), ch):
                    th = torch.tensor(TH[i:i + ch], requires_grad=True); l = m.loss(th, X, y, torch); l.sum().backward()
                    Ls.append(l.detach().numpy()); Gs.append(th.grad.numpy())
                L_, G_ = np.concatenate(Ls), np.concatenate(Gs)
                return dict(dataset=ds, arch=arch.upper(), Q=Q, n_params=c.n, n_inits=len(L_), var_loss=float(L_.var(ddof=1)),
                            var_grad_mean=float(G_.var(0, ddof=1).mean()), var_grad_first=float(G_[:, 0].var(ddof=1)), mean_abs_grad=float(np.abs(G_).mean()))
            rows.append(cached(f"BP|{ds}|{arch}|{Q}|{CFG.L}|{CFG.bp_inits}|{CFG.bp_batch}|{CFG.conv_share}", _do))
    df = pd.DataFrame(rows); fits = []
    for arch, g in df.groupby("arch"):
        if len(g) >= 3:
            for col in ("var_grad_mean", "var_loss"):
                sl = np.polyfit(g.Q, np.log2(g[col]), 1); fits.append(dict(arch=arch, quantity=col, log2_slope_per_qubit=float(sl[0])))
    save_df("E4a_barren_plateau", df); save_df("E4a_slopes", pd.DataFrame(fits)); print(df.round(6).to_string(index=False)); print(pd.DataFrame(fits).round(3).to_string(index=False))
    log("E4a note: PSO is gradient-free but is NOT immune to barren plateaus (loss variance concentrates too); interpret the loss-variance column.")

def exp_E4b():
    e1 = RESULTS.get("E1_main")
    if e1 is None: return not_computed("E4b_convergence", "E1 results missing")
    rows = []
    for ds in AVAILABLE:
        for seed in CFG.seeds:
            for arch in ("ivqnn", "vqnn"):
                cur = {}
                for opt in ("adam", "pso"):
                    r = STORE.get(f"Q|{ds}|{model_label(arch, opt, False, {})}|s{seed}|{_fp(CFG, get_hp(ds))}")
                    if r and r["extra"].get("curve"): cur[opt] = np.array(r["extra"]["curve"], float)
                if len(cur) < 2: continue
                target = max(cur["adam"][-1, 1], cur["pso"][-1, 1])     # loss both optimisers reach by the end of their budget
                for opt, c in cur.items():
                    hit = np.where(c[:, 1] <= target + 1e-12)[0]; k = hit[0] if len(hit) else None
                    rows.append(dict(dataset=ds, seed=seed, arch=arch.upper(), optimizer=opt, final_monitor_loss=c[-1, 1], target_loss=target,
                                     evals_fwd_to_target=c[k, 0] if k is not None else np.nan, evals_ps_to_target=c[k, 2] if k is not None else np.nan))
    df = pd.DataFrame(rows); save_df("E4b_convergence", df)
    if len(df): print(aggregate(df, ["dataset", "arch", "optimizer"], ["final_monitor_loss", "evals_fwd_to_target", "evals_ps_to_target"]).round(4).to_string(index=False))
run_experiment("E4a", exp_E4a); run_experiment("E4b", exp_E4b)

# %% [markdown]
# ## Cell 17 — E5: complexity verification (gate counts, measured simulation time, scaling exponents, 0.618 threshold, per-round wall-clock)

# %%
def _fit_exp(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float)); ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2: return np.nan, np.nan
    sl, ic = np.polyfit(x[ok], y[ok], 1); pred = sl * x[ok] + ic
    r2 = 1 - np.sum((y[ok] - pred) ** 2) / max(np.sum((y[ok] - y[ok].mean()) ** 2), 1e-30); return float(sl), float(r2)

def _tsim(c, P, n, rep=3):
    rng = np.random.default_rng(0); sim = Sim(c, np); X = rng.uniform(0, np.pi, (n, c.Q)); th = rng.normal(size=(P, c.n)).astype(F32)
    sim.expval(th, X); t0 = time.time()
    for _ in range(rep): sim.expval(th, X)
    return (time.time() - t0) / rep

def exp_E5():
    rows = []
    for ent in ("ring", "full"):                                         # connectivity decides the gate-count scaling
        for Q in CFG.cplx_Q:
            for arch in ("vqnn", "ivqnn"):
                c = build_circuit(arch, Q, CFG.L, Q // 2 if arch == "ivqnn" else None, CFG.gateset, ent, CFG.conv_share)
                rows.append(dict(sweep="Q", entangle=ent, arch=arch.upper(), Q=Q, L=CFG.L, Qp=c.Qp, **circ_stats(c), t_P1_n64=_tsim(c, 1, 64) if ent == "ring" else np.nan,
                                 t_P20_n64=_tsim(c, 20, 64) if (ent == "ring" and Q <= 10) else np.nan))
        for L in CFG.cplx_L:
            for arch in ("vqnn", "ivqnn"):
                c = build_circuit(arch, 8, L, 4 if arch == "ivqnn" else None, CFG.gateset, ent, CFG.conv_share)
                rows.append(dict(sweep="L", entangle=ent, arch=arch.upper(), Q=8, L=L, Qp=c.Qp, **circ_stats(c), t_P1_n64=_tsim(c, 1, 64) if ent == "ring" else np.nan, t_P20_n64=np.nan))
        for Qp in CFG.cplx_Qp:
            c = build_circuit("ivqnn", 10, CFG.L, Qp, CFG.gateset, ent, CFG.conv_share)
            rows.append(dict(sweep="Qp", entangle=ent, arch="IVQNN", Q=10, L=CFG.L, Qp=Qp, **circ_stats(c), t_P1_n64=_tsim(c, 1, 64) if ent == "ring" else np.nan, t_P20_n64=np.nan))
    df = pd.DataFrame(rows); save_df("E5_complexity", df)
    ex = []
    for ent in ("ring", "full"):
        for arch in ("VQNN", "IVQNN"):
            g = df[(df.sweep == "Q") & (df.entangle == ent) & (df.arch == arch)]
            sl, r2 = _fit_exp(g.Q, g.n_gates); ex.append(dict(quantity=f"gates vs Q ({ent}, {arch})", exponent=sl, r2=r2, expected="paper: 2 (O(LQ^2)) — only for dense connectivity"))
            sl, r2 = _fit_exp(g.Q, g.n_2q.clip(lower=1)); ex.append(dict(quantity=f"2q-gates vs Q ({ent}, {arch})", exponent=sl, r2=r2, expected=""))
            g = df[(df.sweep == "L") & (df.entangle == ent) & (df.arch == arch)]; sl, r2 = _fit_exp(g.L, g.n_gates); ex.append(dict(quantity=f"gates vs L ({ent}, {arch})", exponent=sl, r2=r2, expected="1"))
    for arch in ("VQNN", "IVQNN"):
        g = df[(df.sweep == "Q") & (df.entangle == "ring") & (df.arch == arch) & (df.Q >= 8)]
        if len(g) >= 2:
            ex.append(dict(quantity=f"log2(time) slope per qubit, P=1,n=64, Q>=8 ({arch})", exponent=float(np.polyfit(g.Q, np.log2(g.t_P1_n64), 1)[0]), r2=np.nan, expected="~1 (+log2 Q/Q gate-count growth): O(2^Q)"))
        g = df[(df.sweep == "L") & (df.entangle == "ring") & (df.arch == arch)]; sl, r2 = _fit_exp(g.L, g.t_P1_n64); ex.append(dict(quantity=f"time vs L ({arch})", exponent=sl, r2=r2, expected="1"))
    g = df[(df.sweep == "Qp") & (df.entangle == "ring")]; sl, r2 = _fit_exp(g.Qp, g.n_gates); ex.append(dict(quantity="IVQNN gates vs Q' (Q=10, ring)", exponent=sl, r2=r2, expected="paper: between 1 and 2"))
    sl, r2 = _fit_exp(g.Qp, g.t_P1_n64); ex.append(dict(quantity="IVQNN sim time vs Q' (Q=10, ring)", exponent=sl, r2=r2, expected=""))
    # formula + threshold check (pure arithmetic on the paper's expressions)
    x = np.linspace(0.01, 1, 100000); f = x ** 2 + x - 1; root = float(x[np.argmin(np.abs(f))])
    ex.append(dict(quantity="numeric root of (Q'/Q)^2+(Q'/Q)=1", exponent=root, r2=np.nan, expected=f"(sqrt5-1)/2={(math.sqrt(5)-1)/2:.4f}"))
    ratio = []
    for Q in CFG.cplx_Q:
        for r_ in (0.25, 0.5, 0.75):
            Qp_ = max(1, int(round(r_ * Q))); ratio.append(dict(Q=Q, Qp=Qp_, formula_ratio=(Qp_ ** 2 + Q * Qp_) / Q ** 2))
    save_df("E5_formula_ratio", pd.DataFrame(ratio))
    # per-round wall-clock of federated PSO vs C, P, T, n_c (full-batch fitness so that n_c matters)
    base = dict(C=3, P=10, T=5, nc=150) if CFG.quick else dict(C=4, P=10, T=10, nc=400)
    grids = dict(C=(2, 4, 8) if not CFG.quick else (2, 3, 4), P=(5, 10, 20) if not CFG.quick else (5, 10, 20), T=(5, 10, 20) if not CFG.quick else (3, 6, 12), nc=(200, 400, 800) if not CFG.quick else (100, 200, 400))
    rrows = []; c = build_circuit("ivqnn", CFG.Q, CFG.L, CFG.Qp, CFG.gateset, CFG.entangle, CFG.conv_share); m = QModel(c, CFG); r_ = np.random.default_rng(0)
    for k, vals in grids.items():
        for v in vals:
            p = {**base, k: v}; cfg = replace(CFG, R=1, C=p["C"], P=p["P"], T=p["T"], pso_batch=0, eval_cap=50)
            cl = [(r_.uniform(0, np.pi, (p["nc"], CFG.Q)).astype(F32), (r_.random(p["nc"]) < .5).astype(F32)) for _ in range(p["C"])]
            _, info = run_federated(m, cl, m.init(r_), "pso", cfg, {}, r_, (cl[0][0], cl[0][1]))
            rrows.append(dict(factor=k, value=v, round_time=info["rounds"][1]["time"], evals_fwd=info["evals_fwd"]))
    rd = pd.DataFrame(rrows); save_df("E5_round_time", rd)
    for k in grids:
        g = rd[rd.factor == k]; sl, r2 = _fit_exp(g.value, g.round_time); ex.append(dict(quantity=f"round time vs {k}", exponent=sl, r2=r2, expected="paper: 1 (linear in C, P, T, n_c)"))
        sl, r2 = _fit_exp(g.value, g.evals_fwd); ex.append(dict(quantity=f"evaluations vs {k}", exponent=sl, r2=r2, expected="1"))
    save_df("E5_exponents", pd.DataFrame(ex)); print(pd.DataFrame(ex).round(3).to_string(index=False))
run_experiment("E5", exp_E5)

# %% [markdown]
# ## Cell 18 — E6: empirical estimates of η (PSO contraction) and γ, δ (federated loss fit) — estimates of ASSUMED inequalities, not proofs
# η: D_t = mean_i‖p_i^t − g*‖² with g* = final swarm best (proxy for Θ*; biased towards contraction because g* is in the swarm).
# γ, δ: least-squares fit of L(r) = (1−γ)^r L(0) + δ to the global loss per round (per seed), plus δ_env = smallest δ for which the *inequality* holds at every round for the fitted γ.

# %%
def fit_gamma_delta(losses):
    l = np.asarray(losses, float); r = np.arange(len(l)); L0 = l[0]
    f = lambda r, g, d: (1 - g) ** r * L0 + d
    try: popt, _ = curve_fit(f, r, l, p0=[0.3, max(l[-1], 1e-3)], bounds=([1e-6, 0], [1 - 1e-6, max(l.max(), 1e-3)]), maxfev=20000)
    except Exception: return None
    pred = f(r, *popt); r2 = 1 - np.sum((l - pred) ** 2) / max(np.sum((l - l.mean()) ** 2), 1e-30)
    return dict(gamma=float(popt[0]), delta=float(popt[1]), r2=float(r2), delta_env=float(np.max(l - (1 - popt[0]) ** r * L0)), L0=float(L0), L_final=float(l[-1]))

def exp_E6():
    rows, erows = [], []
    for ds in AVAILABLE:
        for seed in CFG.seeds:
            for opt in ("pso", "adam"):
                r = STORE.get(f"Q|{ds}|{model_label('ivqnn', opt, True, {})}|s{seed}|{_fp(CFG, get_hp(ds))}")
                if not r: continue
                fit = fit_gamma_delta([q["global_loss"] for q in r["extra"]["rounds"]])
                if fit: rows.append(dict(dataset=ds, seed=seed, optimizer=opt, **fit))
                if opt == "pso":
                    e = pd.DataFrame(r["extra"]["eta"])
                    if len(e): erows.append(dict(dataset=ds, seed=seed, scope="federated local PSO (per round, per client)", eta_geo_mean=e.eta_geo.mean(), eta_geo_min=e.eta_geo.min(),
                                                  ratio_median=e.ratio_median.mean(), violation_frac=e.violation_frac.mean(), frac_runs_eta_in_01=float(((e.eta_geo > 0) & (e.eta_geo < 1)).mean())))
                    rc = STORE.get(f"Q|{ds}|{model_label('ivqnn', 'pso', False, {})}|s{seed}|{_fp(CFG, get_hp(ds))}")
                    if rc and rc["extra"].get("eta_central"):
                        ec = rc["extra"]["eta_central"]; erows.append(dict(dataset=ds, seed=seed, scope="centralised PSO (whole run)", eta_geo_mean=ec["eta_geo"], eta_geo_min=ec["eta_geo"],
                                                                         ratio_median=ec["ratio_median"], violation_frac=ec["violation_frac"], frac_runs_eta_in_01=float(0 < ec["eta_geo"] < 1)))
    g = pd.DataFrame(rows); e = pd.DataFrame(erows); save_df("E6_gamma_delta", g); save_df("E6_eta", e)
    if len(g): print(aggregate(g, ["dataset", "optimizer"], ["gamma", "delta", "delta_env", "r2"]).round(4).to_string(index=False))
    if len(e): print(aggregate(e, ["dataset", "scope"], ["eta_geo_mean", "ratio_median", "violation_frac"]).round(4).to_string(index=False))
    if len(g) and (g.gamma < 1e-3).any(): log("E6 note: some runs show gamma ~ 0 (no measurable contraction of the global loss); the assumed geometric decay is then NOT supported there.")
run_experiment("E6", exp_E6)

# %% [markdown]
# ## Cell 19 — E7: efficiency metric Ξ = accuracy / cost under every cost definition (none is cherry-picked) + scaling checks of the O(1/d²) and O(1/(Q'L)) claims

# %%
def exp_E7():
    df = RESULTS.get("E1_main")
    if df is None: return not_computed("E7", "E1 results missing")
    costs = {"params": "n_params", "train_evals_fwd": "evals_fwd", "train_evals_paramshift": "evals_ps", "train_seconds": "train_time", "train_flops": "flops_train", "infer_flops": "flops_infer"}
    d = df.copy()
    for nm, col in costs.items(): d[f"xi_{nm}"] = d["accuracy"] / d[col].replace(0, np.nan)
    agg = aggregate(d, ["dataset", "model"], [f"xi_{k}" for k in costs]); save_df("E7_xi", agg)
    print(agg[["dataset", "model"] + [f"xi_{k}_mean" for k in costs]].to_string(index=False, float_format=lambda v: f"{v:.3e}"))
    # scaling: classical cost vs d (MLP with hidden width = d), quantum cost vs Q'*L
    ds = AVAILABLE[0]; S = get_split(ds, CFG.seeds[0]); rows = []
    for dd in [4, 8, 16, 32]:
        if dd > S.Xtr.shape[1]: continue
        t0 = time.time(); clf = MLPClassifier(hidden_layer_sizes=(dd,), max_iter=60, random_state=0).fit(S.Xtr[:, :dd], S.ytr); tt = time.time() - t0
        npar = dd * dd + dd + dd + 1; rows.append(dict(kind="MLP hidden=d", size=dd, params=npar, flops=2 * dd * dd + 2 * dd, seconds=tt))
    for Qp in (2, 3, 4, 5):
        for L in (1, 2, 3, 4):
            c = build_circuit("ivqnn", 2 * Qp, L, Qp, CFG.gateset, CFG.entangle, CFG.conv_share); rows.append(dict(kind="IVQNN Q=2Q'", size=Qp * L, params=c.n + 2, flops=sim_flops(c), seconds=_tsim(c, 1, 64), gates=len(c.ops)))
    sc = pd.DataFrame(rows); save_df("E7_scaling_inputs", sc); fits = []
    for kind, g in sc.groupby("kind"):
        for col in ("params", "flops", "seconds") + (("gates",) if "gates" in g and g.gates.notna().any() else ()):
            sl, r2 = _fit_exp(g["size"], g[col]); fits.append(dict(kind=kind, quantity=f"{col} vs size", exponent=sl, r2=r2, paper_claim="cost ~ d^2 (classical); ~ Q'L (quantum)"))
    save_df("E7_scaling_fits", pd.DataFrame(fits)); print(pd.DataFrame(fits).round(3).to_string(index=False))
run_experiment("E7", exp_E7)

# %% [markdown]
# ## Cell 20 — E8: situation-awareness forecasting  ŝ_{t+1} = f_Θ(s_{t-k},…,s_t)
# s_t = attack ratio per fixed time window computed from the raw UNSW-NB15 files (`Stime`, `Label`). Only windows with enough flows are kept and a
# training sample is built only from *consecutive* windows (no invented ordering). If the raw timestamped files are unavailable → `NOT COMPUTED`.
# Chronological 60/20/20 split. Baselines: persistence, moving average, truncated exponential smoothing (α on validation), ARIMA(2,0,1), MLP, LSTM; quantum: IVQNN-PSO (centralised) and F-IVQNN-PSO (C chronological blocks).

# %%
def load_series():
    pats = ["unsw*nb15_[1-4].csv"]; files = _search(pats)
    if not files:
        for fn in (_dl_kaggle, lambda: _dl_hf("raw")):
            try: fn()
            except Exception as e: log("raw-file fetch step failed:", type(e).__name__)
            files = _search(pats)
            if files: break
    files = [f for f in files if re.search(r"nb15_[1-4]\.csv$", f.lower())]
    if not files: return None, "raw UNSW-NB15 files UNSW-NB15_1..4.csv (with Stime/Ltime) not found via Drive/Kaggle/Hugging Face (the official train/test partition CSVs have no timestamps)"
    dfs = []
    for f in sorted(files):
        d = pd.read_csv(f, header=None, usecols=[28, 48], nrows=CFG.max_raw_rows or None, low_memory=False); d.columns = ["stime", "label"]; dfs.append(d)
    d = pd.concat(dfs); d["stime"] = pd.to_numeric(d["stime"], errors="coerce"); d["label"] = pd.to_numeric(d["label"], errors="coerce"); d = d.dropna()
    if len(d) == 0 or d.stime.median() < 1e9: return None, "column 28 does not look like an epoch timestamp (Stime); raw schema differs from the documented one"
    w = ((d.stime - d.stime.min()) // CFG.window_sec).astype(int)
    g = d.groupby(w).label.agg(["mean", "count"]); g = g[g["count"] >= CFG.min_win_count]
    return dict(win=g.index.values, s=g["mean"].values.astype(float), files=[os.path.basename(f) for f in files], n_flows=len(d)), None

def _lag_matrix(win, s, k):
    X, y = [], []
    for i in range(k, len(s) - 1):
        if win[i + 1] - win[i - k] == k + 1: X.append(s[i - k:i + 1]); y.append(s[i + 1])
    return np.array(X, float), np.array(y, float)

def exp_E8():
    ser, why = load_series()
    if ser is None: return not_computed("E8_forecasting", why)
    k = CFG.Q - 1; X, y = _lag_matrix(ser["win"], ser["s"], k)
    if len(y) < 60: return not_computed("E8_forecasting", f"only {len(y)} usable consecutive-window samples")
    n = len(y); i1, i2 = int(.6 * n), int(.8 * n); Xtr, ytr, Xva, yva, Xte, yte = X[:i1], y[:i1], X[i1:i2], y[i1:i2], X[i2:], y[i2:]
    log(f"E8: {len(ser['s'])} windows ({CFG.window_sec}s), {n} samples (k={k}); attack-ratio mean={ser['s'].mean():.3f}, var={ser['s'].var():.4f}; files={ser['files']}")
    mets = lambda p: dict(mse=float(np.mean((p - yte) ** 2)), mae=float(np.mean(np.abs(p - yte))))
    rows, preds = [], {"y_true": yte}
    def add(name, seed, p): rows.append(dict(model=name, seed=seed, **mets(np.clip(p, 0, 1)))); preds.setdefault(name, np.clip(p, 0, 1)) if seed == CFG.seeds_small[0] else None
    add("persistence", 0, Xte[:, -1]); add("moving-average", 0, Xte.mean(1))
    def es(Xa, a): wts = a * (1 - a) ** np.arange(Xa.shape[1])[::-1]; return (Xa * wts).sum(1) / wts.sum()
    a_best = min(np.linspace(0.05, 0.95, 19), key=lambda a: np.mean((es(Xva, a) - yva) ** 2)); add(f"exp-smoothing(a={a_best:.2f})", 0, es(Xte, a_best))
    try:
        from statsmodels.tsa.arima.model import ARIMA
        s_full = ser["s"]; res = ARIMA(s_full[:i1 + k + 1], order=(2, 0, 1)).fit(); fv = res.apply(s_full).fittedvalues
        tgt_idx = [i + 1 for i in range(k, len(s_full) - 1) if ser["win"][i + 1] - ser["win"][i - k] == k + 1]
        add("ARIMA(2,0,1)", 0, np.asarray(fv)[tgt_idx][i2:])
    except Exception as e: not_computed("E8_ARIMA", f"{type(e).__name__}: {str(e)[:100]}")
    from sklearn.neural_network import MLPRegressor
    for seed in CFG.seeds_small:
        add("MLP(16)", seed, MLPRegressor(hidden_layer_sizes=(16,), max_iter=800, random_state=seed).fit(Xtr, ytr).predict(Xte))
        if HAS_TORCH:
            torch.manual_seed(seed); net = torch.nn.LSTM(1, 16, batch_first=True); head = torch.nn.Linear(16, 1); opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=0.01)
            tt = lambda a: torch.as_tensor(a, dtype=torch.float32)[:, :, None]; fwd = lambda a: torch.sigmoid(head(net(tt(a))[0][:, -1])[:, 0])
            best, bs = 1e9, None
            for ep in range(150):
                opt.zero_grad(); loss = ((fwd(Xtr) - torch.as_tensor(ytr, dtype=torch.float32)) ** 2).mean(); loss.backward(); opt.step()
                with torch.no_grad(): vl = float(((fwd(Xva).numpy() - yva) ** 2).mean())
                if vl < best: best, bs = vl, ({k_: v.clone() for k_, v in net.state_dict().items()}, {k_: v.clone() for k_, v in head.state_dict().items()})
            net.load_state_dict(bs[0]); head.load_state_dict(bs[1])
            with torch.no_grad(): add("LSTM(16)", seed, fwd(Xte).numpy())
        # quantum forecasters (angles = ratio*pi; target ratio in [0,1]; head output used directly as the forecast)
        Atr, Ate = (Xtr * np.pi).astype(F32), (Xte * np.pi).astype(F32); cfg = CFG
        c = build_circuit("ivqnn", cfg.Q, cfg.L, cfg.Qp, cfg.gateset, cfg.entangle, cfg.conv_share); m = QModel(c, cfg, task="mse"); th0 = m.init(rng_for("e8init", seed))
        t0 = time.time(); th, _ = train_pso(m, Atr, ytr.astype(F32), th0, rng_for("e8c", seed), replace(cfg, resample_every=cfg.T), T=cfg.R * cfg.T)
        add("IVQNN-PSO (central)", seed, predict_proba(m, th, Ate))
        blocks = np.array_split(np.arange(len(ytr)), cfg.C); cl = [(Atr[b], ytr[b].astype(F32)) for b in blocks]
        th, _ = run_federated(m, cl, th0, "pso", cfg, {}, rng_for("e8f", seed), (Atr[-200:], (ytr[-200:] > ytr.mean()).astype(int)))
        add("F-IVQNN-PSO", seed, predict_proba(m, th, Ate))
    df = pd.DataFrame(rows); save_df("E8_forecasting", df); np.savez_compressed(ROOT / "preds" / "E8_forecast_preds.npz", **{k_: v for k_, v in preds.items()})
    print(aggregate(df, ["model"], ["mse", "mae"]).round(5).to_string(index=False)); log(f"E8 note: tolerance eps*={CFG.forecast_tol_mse} on test MSE was fixed in the config before running.")
run_experiment("E8", exp_E8)

# %% [markdown]
# ## Cell 21 — E9 (optional): depolarising-noise robustness (density-matrix simulation, Q=4, L=2)

# %%
def exp_E9():
    ds = [d for d in CFG.sec_datasets if d in AVAILABLE][0]; seed = CFG.seeds_small[0]; S = get_split(ds, seed); Q, L = 4, 2
    Xtr, _, Xte = S.q(Q); n = min(CFG.noise_n, len(Xte)); rows = []; cfg = CFG
    for arch in ("vqnn", "ivqnn"):
        c = build_circuit(arch, Q, L, 2 if arch == "ivqnn" else None, cfg.gateset, cfg.entangle, cfg.conv_share); m = QModel(c, cfg)
        th, _ = train_pso(m, Xtr, S.ytr.astype(F32), m.init(rng_for("e9", arch)), rng_for("e9t", arch), replace(cfg, P=20, resample_every=10), T=max(60, cfg.R * cfg.T))
        for p in CFG.noise_p:
            f = density_expval(c, th[:c.n], Xte[:n], p); pr = 1 / (1 + np.exp(-(cfg.head_gain * th[c.n] * f + th[c.n + 1]))); yy = S.yte[:n]
            if p == 0: assert np.abs(f - Sim(c, np, True).expval(th[None, :c.n].astype(np.float64), Xte[:n])[0]).max() < 1e-4
            rows.append(dict(dataset=ds, arch=arch.upper(), p_dep=p, n_gates=len(c.ops), **{k: v for k, v in binary_metrics(yy, pr).items() if k in ("accuracy", "f1", "mse", "roc_auc")}))
    df = pd.DataFrame(rows); save_df("E9_noise", df); print(df.round(4).to_string(index=False))
run_experiment("E9", exp_E9)

# %% [markdown]
# ## Cell 22 — E10: automatic claims check (rules fixed in advance in the code below)
# (i) proposed > every classical baseline on accuracy/recall/precision/F1 and lower MSE on both datasets · (ii) proposed > conventional QNN baselines · (iii) PSO converges faster AND more reliably than Adam (both evaluation accountings) ·
# (iv) pooling reduces cost (params, gates, simulation time, training time) · (v) mean ECE < 0.05 on every dataset · (vi) |central − federated| F1 gap ≤ 0.01 (FAIL if > 0.03) · (vii) forecast test MSE < tolerance.
# Rule for (i)/(ii): PASS = mean-better on every (dataset, baseline, metric); FAIL = worse in ≥ 50 % of them; otherwise INCONCLUSIVE.

# %%
def exp_E10():
    e1, sig = RESULTS.get("E1_main"), RESULTS.get("E1_significance"); out = []
    def tri(ok, bad): return "PASS" if ok else ("FAIL" if bad else "INCONCLUSIVE")
    def dominance(baselines, label):
        if e1 is None: return out.append(dict(claim=label, verdict="INCONCLUSIVE", support="E1 missing"))
        m = e1.groupby(["dataset", "model"])[["accuracy", "recall", "precision", "f1", "mse"]].mean(); tot = better = sigc = 0; worse_list = []
        for ds in AVAILABLE:
            for b in baselines:
                if (ds, b) not in m.index or (ds, PROPOSED) not in m.index: continue
                for met in ("accuracy", "recall", "precision", "f1", "mse"):
                    a_, b_ = m.loc[(ds, PROPOSED), met], m.loc[(ds, b), met]; ok = (a_ < b_) if met == "mse" else (a_ > b_); tot += 1; better += int(ok)
                    if not ok: worse_list.append(f"{ds}:{b}:{met}")
                    if sig is not None and len(sig):
                        r = sig[(sig.dataset == ds) & (sig.baseline == b) & (sig.metric == met)]
                        sigc += int(len(r) > 0 and r.p_ttest_holm.iloc[0] < 0.05 and ok)
        out.append(dict(claim=label, verdict=tri(tot > 0 and better == tot, tot > 0 and better < tot / 2) if tot else "INCONCLUSIVE",
                        support=f"better in {better}/{tot} (dataset,baseline,metric) cells; Holm-significant (paired t) in {sigc}; worse: {', '.join(worse_list[:6])}"))
    dominance(CLASSICAL + ["F-MLP-Adam"], "(i) F-IVQNN-PSO beats classical baselines (acc, rec, prec, F1, MSE) on all datasets")
    dominance(["VQNN-Adam", "F-VQNN-Adam", "VQNN-PSO"], "(ii) F-IVQNN-PSO beats conventional QNN baselines")
    cv = RESULTS.get("E4b_convergence")
    if cv is not None and len(cv):
        g = cv.groupby("optimizer"); ps, fw, sd = g.evals_ps_to_target.mean(), g.evals_fwd_to_target.mean(), g.final_monitor_loss.std()
        if {"pso", "adam"} <= set(ps.index):
            f_ps, f_fw, rel = ps["pso"] < ps["adam"], fw["pso"] < fw["adam"], sd["pso"] < sd["adam"]
            out.append(dict(claim="(iii) PSO converges faster and more reliably than gradient training", verdict=tri(f_ps and f_fw and rel, not (f_ps or f_fw or rel)),
                            support=f"evals-to-target (param-shift) PSO {ps['pso']:.0f} vs Adam {ps['adam']:.0f}; (fwd) {fw['pso']:.0f} vs {fw['adam']:.0f}; std final loss {sd['pso']:.4f} vs {sd['adam']:.4f}"))
    else: out.append(dict(claim="(iii) PSO converges faster/more reliably", verdict="INCONCLUSIVE", support="E4b missing"))
    if e1 is not None:
        m = e1.groupby("model")[["n_params", "circ_n_gates", "train_time"]].mean(); a, b = "IVQNN-PSO", "VQNN-PSO"
        if a in m.index and b in m.index:
            cA = build_circuit("ivqnn", CFG.Q, CFG.L, CFG.Qp, CFG.gateset, CFG.entangle, CFG.conv_share); cB = build_circuit("vqnn", CFG.Q, CFG.L, None, CFG.gateset, CFG.entangle, CFG.conv_share)
            tA, tB = _tsim(cA, CFG.P, CFG.pso_batch), _tsim(cB, CFG.P, CFG.pso_batch)
            flags = [m.loc[a, "n_params"] < m.loc[b, "n_params"], m.loc[a, "circ_n_gates"] < m.loc[b, "circ_n_gates"], tA < tB, m.loc[a, "train_time"] < m.loc[b, "train_time"]]
            out.append(dict(claim="(iv) pooling reduces cost", verdict=tri(all(flags), not any(flags)),
                            support=f"params {m.loc[a,'n_params']:.0f} vs {m.loc[b,'n_params']:.0f}; gates {m.loc[a,'circ_n_gates']:.0f} vs {m.loc[b,'circ_n_gates']:.0f}; sim time {tA:.3f}s vs {tB:.3f}s; train time {m.loc[a,'train_time']:.1f}s vs {m.loc[b,'train_time']:.1f}s (IVQNN vs VQNN)"))
    if e1 is not None:
        ec = e1[e1.model == PROPOSED].groupby("dataset").ece.mean()
        if len(ec): out.append(dict(claim=f"(v) calibration: mean ECE < {CFG.ece_thr}", verdict=tri((ec < CFG.ece_thr).all(), (ec >= CFG.ece_thr).all()), support="; ".join(f"{k}: ECE={v:.4f}" for k, v in ec.items())))
        gaps = []
        for ds in AVAILABLE:
            g = e1[e1.dataset == ds].groupby("model").f1.mean()
            if "IVQNN-PSO" in g.index and PROPOSED in g.index: gaps.append((ds, g["IVQNN-PSO"] - g[PROPOSED]))
        if gaps: out.append(dict(claim="(vi) federated approaches centralised performance (F1 gap, IID)", verdict=tri(all(v <= CFG.gap_pass for _, v in gaps), any(v > CFG.gap_fail for _, v in gaps)), support="; ".join(f"{d}: gap={v:+.4f}" for d, v in gaps)))
    fc = RESULTS.get("E8_forecasting")
    if fc is not None and len(fc):
        q = fc[fc.model.str.contains("IVQNN")].groupby("model").mse.mean()
        if len(q): out.append(dict(claim=f"(vii) forecasting MSE < {CFG.forecast_tol_mse}", verdict=tri((q < CFG.forecast_tol_mse).all(), (q >= CFG.forecast_tol_mse).all()),
                                   support="; ".join(f"{k}: MSE={v:.5f}" for k, v in q.items()) + f"; persistence MSE={fc[fc.model=='persistence'].mse.mean():.5f}"))
    else: out.append(dict(claim=f"(vii) forecasting MSE < {CFG.forecast_tol_mse}", verdict="INCONCLUSIVE", support="NOT COMPUTED: " + NOT_COMPUTED.get("E8_forecasting", "E8 did not run")))
    df = pd.DataFrame(out); save_df("E10_claims", df); print(df.to_string(index=False, max_colwidth=160))
run_experiment("E10", exp_E10)

# %% [markdown]
# ## Cell 23 — Figures (matplotlib only; PDF + 300-dpi PNG in `results/figures/`; Okabe–Ito colour-blind-safe palette; fonts ≥ 8 pt; sized for IEEE single (3.5 in) / double (7.16 in) column)
# F1 ROC · F2 PR · F3 reliability · F4 decision landscape · F5 convergence (+ fitted federated curve) · F6 barren plateaus · F7 complexity + Ξ · F8 non-IID / clients / ablations · F9 forecasting

# %%
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.prop_cycle": cycler(color=OKABE), "figure.dpi": 110, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.2})
SINGLE, DOUBLE = 3.5, 7.16
FIGDIR = ROOT / "figures"

def savefig(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIGDIR / f"{name}.png", dpi=300, bbox_inches="tight"); plt.show(); plt.close(fig)
    log(f"figure saved: {name}")

def load_results():
    for p in ROOT.glob("E*.csv"):
        if p.stem not in RESULTS: RESULTS[p.stem] = pd.read_csv(p)
load_results()
FIG_MODELS = ["LR-Q", "MLP-Q", "RF-full", "VQNN-Adam", "IVQNN-Adam", "IVQNN-PSO", "F-IVQNN-Adam", "F-IVQNN-PSO"]

def _mean_curve(ds, model, seeds, kind):
    grid = np.linspace(0, 1, 201); ys = []
    for s in seeds:
        pr = load_preds(ds, model, s)
        if pr is None: continue
        y, p = pr
        if len(np.unique(y)) < 2: continue
        if kind == "roc": fpr, tpr, _ = roc_curve(y, p); ys.append(np.interp(grid, fpr, tpr))
        else:
            pp, rc, _ = precision_recall_curve(y, p); o = np.argsort(rc); ys.append(np.interp(grid, rc[o], pp[o]))
    return (grid, np.mean(ys, 0), np.std(ys, 0)) if ys else None

def fig_curves(kind):
    fig, axs = plt.subplots(1, len(AVAILABLE), figsize=(DOUBLE, 3.0), squeeze=False)
    for ax, ds in zip(axs[0], AVAILABLE):
        for i, m in enumerate(FIG_MODELS):
            c = _mean_curve(ds, m, CFG.seeds, kind)
            if c is None: continue
            ax.plot(c[0], c[1], label=m, color=OKABE[i % 8], ls="-" if i < 5 else "--"); ax.fill_between(c[0], c[1] - c[2], c[1] + c[2], color=OKABE[i % 8], alpha=0.15, lw=0)
        ax.set_title(ds); ax.set_xlabel("FPR" if kind == "roc" else "Recall"); ax.set_ylabel("TPR" if kind == "roc" else "Precision")
        if kind == "roc": ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls=":")
        if kind == "roc": ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    axs[0][0].legend(loc="lower right" if kind == "roc" else "lower left", ncol=1, frameon=False)
    fig.tight_layout(); savefig(fig, "F1_roc" if kind == "roc" else "F2_pr")

def fig_reliability():
    models = ["MLP-Q", "VQNN-Adam", "F-IVQNN-Adam", "F-IVQNN-PSO"]; e1 = RESULTS.get("E1_main")
    fig, axs = plt.subplots(len(AVAILABLE), len(models), figsize=(DOUBLE, 1.8 * len(AVAILABLE) + 0.4), squeeze=False)
    for r, ds in enumerate(AVAILABLE):
        for c, m in enumerate(models):
            ax = axs[r][c]; ys, ps = [], []
            for s in CFG.seeds:
                pr = load_preds(ds, m, s)
                if pr is not None: ys.append(pr[0]); ps.append(pr[1])
            if not ys: ax.axis("off"); continue
            y, p = np.concatenate(ys), np.concatenate(ps); idx = np.minimum((p * 10).astype(int), 9)
            xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() > 5]; yy = [y[idx == b].mean() for b in range(10) if (idx == b).sum() > 5]
            ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.7); ax.plot(xs, yy, marker="o", ms=3, color=OKABE[c])
            ece = e1[(e1.dataset == ds) & (e1.model == m)].ece.mean() if e1 is not None else np.nan
            ax.set_title(f"{m}\nECE={ece:.3f}" if r == 0 else f"ECE={ece:.3f}", fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if c == 0: ax.set_ylabel(f"{ds}\nfraction positive")
            if r == len(AVAILABLE) - 1: ax.set_xlabel("mean predicted prob.")
    fig.tight_layout(); savefig(fig, "F3_reliability")

LAND_ROWS = []
def fig_landscape():
    ds = AVAILABLE[0]; seed = CFG.seeds[0]; S = get_split(ds, seed); Xtr, _, Xte = S.q(CFG.Q); hp = get_hp(ds)
    pca2 = PCA(2, random_state=0).fit(Xtr); Z = pca2.transform(Xte); mu = Xtr.mean(0)
    gx, gy = np.meshgrid(np.linspace(Z[:, 0].min(), Z[:, 0].max(), 70), np.linspace(Z[:, 1].min(), Z[:, 1].max(), 70))
    G = pca2.inverse_transform(np.c_[gx.ravel(), gy.ravel()]); G = np.clip(G, 0, np.pi).astype(F32)
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.5)); sel = np.random.default_rng(0).choice(len(Z), min(300, len(Z)), replace=False)
    for ax, name in zip(axs, ["VQNN-Adam", "IVQNN-Adam", "MLP-Q"]):
        if name == "MLP-Q": P_ = MLPClassifier(hidden_layer_sizes=tuple(hp["MLP-Q"]), max_iter=200, random_state=seed).fit(Xtr, S.ytr).predict_proba(G)[:, 1]
        else:
            arch = name.split("-")[0].lower(); r = STORE.get(f"Q|{ds}|{name}|s{seed}|{_fp(CFG, hp)}")
            if not r: ax.axis("off"); continue
            c = build_circuit(arch, CFG.Q, CFG.L, CFG.Qp if arch == "ivqnn" else None, CFG.gateset, CFG.entangle, CFG.conv_share)
            P_ = predict_proba(QModel(c, CFG), np.array(r["extra"]["theta"], dtype=F32), G)
        cs = ax.contourf(gx, gy, P_.reshape(gx.shape), levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.85); LAND_ROWS.append(pd.DataFrame(dict(dataset=ds, model=name, pc1=gx.ravel(), pc2=gy.ravel(), p_attack=P_)))
        ax.scatter(Z[sel, 0], Z[sel, 1], c=S.yte[sel], cmap="RdBu_r", edgecolor="k", s=6, lw=0.3); ax.set_title(name); ax.set_xlabel("PC1"); ax.grid(False)
    axs[0].set_ylabel("PC2"); fig.colorbar(cs, ax=axs, shrink=0.8, label="P(attack)")
    savefig(fig, "F4_decision_landscape")
    print("F4 caption caveat: grid points live in the 2-D PCA plane of the Q-dim angle features (fitted on train); the remaining Q-2 components are held at their mean (inverse-transform of the 2-D point), so this is a 2-D slice, not the full decision surface.")

def _interp_curve(curves, xs, col):
    out = []
    for c in curves:
        c = np.array(c, float); out.append(np.interp(xs, c[:, col], c[:, 1]))
    return np.mean(out, 0), np.std(out, 0)

def fig_convergence():
    ds = AVAILABLE[0]; hp = get_hp(ds); fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.5))
    for k, (col, lab) in enumerate([(2, "parameter-shift-equivalent evaluations"), (0, "forward evaluations")]):
        ax = axs[k]
        for i, (arch, opt) in enumerate([("ivqnn", "adam"), ("ivqnn", "pso"), ("vqnn", "adam"), ("vqnn", "pso")]):
            cur = [STORE.get(f"Q|{ds}|{model_label(arch, opt, False, {})}|s{s}|{_fp(CFG, hp)}") for s in CFG.seeds]; cur = [c["extra"]["curve"] for c in cur if c and c["extra"].get("curve")]
            if not cur: continue
            hi = min(np.array(c, float)[-1, col] for c in cur); xs = np.geomspace(max(1, np.array(cur[0], float)[1, col]), hi, 60); m, s = _interp_curve(cur, xs, col)
            ax.plot(xs, m, label=f"{arch.upper()}-{opt.upper() if opt == 'pso' else 'Adam'}", color=OKABE[i], ls="-" if arch == "ivqnn" else "--"); ax.fill_between(xs, m - s, m + s, color=OKABE[i], alpha=0.15, lw=0)
        ax.set_xscale("log"); ax.set_xlabel(lab); ax.set_ylabel("monitor loss (BCE)"); ax.set_title(f"{ds}: central training")
    axs[0].legend(frameon=False); ax = axs[2]; g = RESULTS.get("E6_gamma_delta")
    for i, opt in enumerate(("pso", "adam")):
        runs = [STORE.get(f"Q|{ds}|{model_label('ivqnn', opt, True, {})}|s{s}|{_fp(CFG, hp)}") for s in CFG.seeds]; runs = [r for r in runs if r]
        if not runs: continue
        L = np.mean([[q["global_loss"] for q in r["extra"]["rounds"]] for r in runs], 0); rr = np.arange(len(L)); ax.plot(rr, L, marker="o", ms=2.5, color=OKABE[i], label=f"F-IVQNN-{'PSO' if opt=='pso' else 'Adam'}")
        if g is not None and len(g):
            q = g[(g.dataset == ds) & (g.optimizer == opt)]
            if len(q): ax.plot(rr, (1 - q.gamma.mean()) ** rr * L[0] + q.delta.mean(), color=OKABE[i], ls=":", label=f"fit γ={q.gamma.mean():.2f}, δ={q.delta.mean():.2f}")
    ax.set_xlabel("federated round"); ax.set_ylabel("global loss  Σ n_c/N · L_c"); ax.legend(frameon=False, fontsize=6); ax.set_title("FedAvg + fitted (1−γ)^r L0 + δ")
    fig.tight_layout(); savefig(fig, "F5_convergence")

def fig_barren():
    d = RESULTS.get("E4a_barren_plateau")
    if d is None or not len(d): return log("F6 skipped (E4a not computed)")
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    for ax, col, lab in zip(axs, ("var_grad_mean", "var_loss"), ("Var[∂L/∂θ] (mean over params)", "Var[L]")):
        for i, a in enumerate(("VQNN", "IVQNN")):
            g = d[d.arch == a]; ax.semilogy(g.Q, g[col], marker="o", ms=3, label=a, color=OKABE[i])
        ax.set_xlabel("qubits Q"); ax.set_ylabel(lab); ax.legend(frameon=False)
    fig.tight_layout(); savefig(fig, "F6_barren_plateau")

def fig_complexity():
    d = RESULTS.get("E5_complexity"); xi = RESULTS.get("E7_xi")
    if d is None: return
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 5.2))
    ax = axs[0][0]
    for i, (ent, a) in enumerate([(e, a) for e in ("ring", "full") for a in ("VQNN", "IVQNN")]):
        g = d[(d.sweep == "Q") & (d.entangle == ent) & (d.arch == a)]; ax.plot(g.Q, g.n_gates, marker="o", ms=3, label=f"{a} ({ent})", color=OKABE[i])
    ax.set_xlabel("Q"); ax.set_ylabel("gate count"); ax.set_yscale("log"); ax.legend(frameon=False, fontsize=6)
    ax = axs[0][1]
    for i, a in enumerate(("VQNN", "IVQNN")):
        g = d[(d.sweep == "Q") & (d.entangle == "ring") & (d.arch == a)]; ax.semilogy(g.Q, g.t_P1_n64, marker="o", ms=3, label=f"{a}, P=1", color=OKABE[i])
        g2 = g.dropna(subset=["t_P20_n64"]); ax.semilogy(g2.Q, g2.t_P20_n64, marker="s", ms=3, ls="--", label=f"{a}, P=20", color=OKABE[i + 2])
    ax.set_xlabel("Q"); ax.set_ylabel("simulation time / forward (s)"); ax.legend(frameon=False, fontsize=6)
    ax = axs[1][0]
    for i, (col, lab) in enumerate([("n_gates", "gates"), ("t_P1_n64", "time (s)")]):
        for j, (sw, x) in enumerate([("L", "L"), ("Qp", "Qp")]):
            g = d[(d.sweep == sw) & (d.entangle == "ring") & (d.arch == "IVQNN")]
            if len(g): ax.plot(g[x] / g[x].max(), g[col] / g[col].max(), marker="o", ms=3, label=f"IVQNN {lab} vs {x}", color=OKABE[i * 2 + j])
    ax.set_xlabel("sweep value / max"); ax.set_ylabel("cost / max"); ax.legend(frameon=False, fontsize=6)
    ax = axs[1][1]
    if xi is not None and len(xi):
        ds = AVAILABLE[0]; g = xi[(xi.dataset == ds) & xi.model.isin(["LR-Q", "MLP-Q", "VQNN-Adam", "IVQNN-Adam", "F-IVQNN-PSO"])]
        for i, c in enumerate(["xi_params_mean", "xi_train_seconds_mean", "xi_infer_flops_mean"]):
            if c in g: ax.bar(np.arange(len(g)) + 0.27 * i, g[c], width=0.27, label=c.replace("xi_", "").replace("_mean", ""), color=OKABE[i])
        ax.set_xticks(np.arange(len(g)) + 0.27); ax.set_xticklabels(g.model, rotation=35, ha="right"); ax.set_yscale("log"); ax.set_ylabel("Ξ = accuracy / cost"); ax.legend(frameon=False, fontsize=6); ax.set_title(ds)
    fig.tight_layout(); savefig(fig, "F7_complexity_xi")

def fig_ablation():
    e2, e3, cv = RESULTS.get("E2_federated"), RESULTS.get("E3_ablations"), RESULTS.get("E2_round_curves")
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 5.2))
    if e2 is not None and len(e2):
        ds = e2.dataset.iloc[0]; g = aggregate(e2[(e2.dataset == ds) & (e2.optimizer == "pso")], ["config"], ["accuracy", "f1"]); ax = axs[0][0]
        ax.bar(g.config, g.accuracy_mean, yerr=g.accuracy_std.fillna(0), color=OKABE[0], capsize=2); ax.set_ylabel("test accuracy"); ax.set_title(f"F-IVQNN-PSO, {ds}"); ax.tick_params(axis="x", rotation=40)
        for t in ax.get_xticklabels(): t.set_ha("right")
    if cv is not None and len(cv):
        ds = cv.dataset.iloc[0]; ax = axs[0][1]
        for i, (cfgn, g) in enumerate(cv[(cv.dataset == ds) & (cv.optimizer == "pso")].groupby("config")):
            m = g.groupby("round").val_acc.mean(); ax.plot(m.index, m.values, label=cfgn, color=OKABE[i % 8])
        ax.set_xlabel("round"); ax.set_ylabel("validation accuracy"); ax.legend(frameon=False, fontsize=6)
    if e3 is not None and len(e3):
        ds = e3.dataset.iloc[0]; ax = axs[1][0]
        g = aggregate(e3[(e3.dataset == ds) & e3.group.isin(["pooling", "optimizer", "init", "conv_share"])], ["group", "setting"], ["accuracy"])
        lab = g.group + ": " + g.setting; ax.barh(lab, g.accuracy_mean, xerr=g.accuracy_std.fillna(0), color=OKABE[1], capsize=2); ax.set_xlabel("test accuracy"); ax.invert_yaxis()
        ax = axs[1][1]
        for i, grp in enumerate(("P", "T", "L", "Q")):
            g = aggregate(e3[(e3.dataset == ds) & (e3.group == grp) & (~e3.setting.str.contains("vqnn") | (grp in ("P", "T")))], ["group", "setting"], ["accuracy"])
            if len(g): xs = pd.to_numeric(g.setting.str.extract(r"(\d+)$")[0], errors="coerce"); ax.plot(np.arange(len(g)), g.accuracy_mean, marker="o", ms=3, label=f"{grp}: " + ",".join(map(str, xs.astype(int).tolist())), color=OKABE[i])
        ax.set_xlabel("sweep index"); ax.set_ylabel("test accuracy (IVQNN-PSO)"); ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(); savefig(fig, "F8_noniid_clients_ablation")

def fig_forecast():
    f = ROOT / "preds" / "E8_forecast_preds.npz"
    if not f.exists(): return log("F9 skipped (E8 not computed)")
    z = np.load(f); n = min(150, len(z["y_true"])); fig, ax = plt.subplots(figsize=(DOUBLE, 2.6)); ax.plot(z["y_true"][:n], color="k", lw=1.4, label="true $s_{t+1}$")
    for i, k in enumerate([k for k in z.files if k in ("persistence", "MLP(16)", "LSTM(16)", "IVQNN-PSO (central)", "F-IVQNN-PSO")]): ax.plot(z[k][:n], label=k, color=OKABE[i], lw=0.9, alpha=0.9)
    ax.set_xlabel("test step (windows)"); ax.set_ylabel("attack ratio"); ax.legend(frameon=False, ncol=3, fontsize=6); fig.tight_layout(); savefig(fig, "F9_forecast")

for _name, _fn in [("F1", lambda: fig_curves("roc")), ("F2", lambda: fig_curves("pr")), ("F3", fig_reliability), ("F4", fig_landscape), ("F5", fig_convergence),
                   ("F6", fig_barren), ("F7", fig_complexity), ("F8", fig_ablation), ("F9", fig_forecast)]:
    try: _fn()
    except Exception as e: log(f"figure {_name} failed: {type(e).__name__}: {str(e)[:150]}")

# %% [markdown]
# ## Cell 24 — LaTeX (booktabs, IEEEtran-friendly) and CSV tables: main results, ablations, complexity, Ξ, hyper-parameters, circuit specification, dataset statistics

# %%
TABDIR = ROOT / "tables"
def tex_table(header, rows, caption, label, name, colfmt=None):
    colfmt = colfmt or "l" + "c" * (len(header) - 1)
    L = [f"% data_source={OVERALL_SOURCE()} mode={MODE_STR}", "\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\footnotesize", f"\\begin{{tabular}}{{{colfmt}}}", "\\toprule",
         " & ".join(header) + " \\\\", "\\midrule"] + [" & ".join(r) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABDIR / f"{name}.tex").write_text("\n".join(L)); log(f"table saved: {name}.tex")
esc = lambda s: str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
def fmt(m, s, bold=False, d=4):
    if m is None or (isinstance(m, float) and math.isnan(m)): return "--"
    t = f"{m:.{d}f}" + ("" if s is None or (isinstance(s, float) and math.isnan(s)) else f" $\\pm$ {s:.{d}f}")
    return f"\\textbf{{{t}}}" if bold else t

def table_main():
    summ = RESULTS.get("E1_summary")
    if summ is None: return
    for tag, mets in (("main", ["accuracy", "recall", "precision", "f1", "mse"]), ("extra", ["mcc", "roc_auc", "pr_auc", "ece", "f1_macro"])):
        for ds in AVAILABLE:
            g = summ[summ.dataset == ds].reset_index(drop=True); rows = []
            best = {m: (g[f"{m}_mean"].idxmax() if HIGHER_BETTER[m] else g[f"{m}_mean"].idxmin()) for m in mets}
            for i, r in g.iterrows(): rows.append([esc(r.model)] + [fmt(r[f"{m}_mean"], r[f"{m}_std"], best[m] == i) for m in mets])
            tex_table(["Model"] + [m.upper().replace("_", " ") for m in mets], rows, f"{tag.capitalize()} test results on {esc(ds)} (mean $\\pm$ std over {int(g.n_seeds.max())} seeds; best in bold).", f"tab:{tag}_{ds}", f"T_{tag}_{ds}")
            g[["model"] + [f"{m}_{s}" for m in mets for s in ("mean", "std", "ci95")]].assign(dataset=ds, data_source=data_source(ds), mode=MODE_STR).to_csv(TABDIR / f"T_{tag}_{ds}.csv", index=False)
def table_generic(df, cols, name, caption, digits=4, rename=None):
    if df is None or not len(df): return
    d = df[cols].copy(); d.assign(data_source=OVERALL_SOURCE(), mode=MODE_STR).to_csv(TABDIR / f"{name}.csv", index=False)
    rows = [[esc(v) if isinstance(v, str) else (f"{v:.{digits}g}" if isinstance(v, (float, np.floating)) and v == v else ("--" if isinstance(v, float) else str(v))) for v in r] for r in d.itertuples(index=False)]
    tex_table([esc(c) for c in (rename or cols)], rows, caption, f"tab:{name}", name)
def table_hparams():
    keys = ["Q", "L", "Qp", "gateset", "entangle", "conv_share", "head", "head_gain", "init_std", "P", "T", "w", "phi1", "phi2", "vmax", "eps", "pso_batch", "swarm_init", "init_sigma", "adam_batch", "local_adam_steps", "C", "R", "partition", "alpha", "n_train_max", "n_test_max", "seeds"]
    rows = [(k, str(getattr(CFG, k))) for k in keys]
    for ds in AVAILABLE:
        hp = get_hp(ds); rows += [(f"tuned[{ds}] {k}", str(v)) for k, v in hp.items() if k != "tuning_log"]
    pd.DataFrame(rows, columns=["parameter", "value"]).to_csv(TABDIR / "T_hyperparameters.csv", index=False)
    tex_table(["Parameter", "Value"], [[esc(a), esc(b)] for a, b in rows], "Hyper-parameters (config and values selected on the validation split).", "tab:hparams", "T_hyperparameters", "ll")

table_main()
e3s = RESULTS.get("E3_summary")
if e3s is not None and len(e3s): table_generic(e3s, ["dataset", "group", "setting", "accuracy_mean", "accuracy_std", "f1_mean", "mse_mean", "n_params_mean", "train_time_mean"], "T_ablations", "Ablations (mean over seeds).", 4)
table_generic(RESULTS.get("E5_exponents"), ["quantity", "exponent", "r2", "expected"], "T_complexity", "Empirical scaling exponents versus the paper's complexity claims.", 3)
xi = RESULTS.get("E7_xi")
if xi is not None and len(xi): table_generic(xi, ["dataset", "model"] + [c for c in xi.columns if c.endswith("_mean") and c.startswith("xi_")], "T_xi", "Efficiency $\\Xi$ = accuracy / cost under each cost definition.", 3)
table_hparams()
table_generic(CIRCUIT_TABLE[CIRCUIT_TABLE.L.isin([1, 3])], ["model", "Q", "L", "Qp", "n_circuit_params", "n_params_with_head", "n_gates", "n_1q", "n_2q", "depth"], "T_circuits", "Circuit specification (counts exclude the Q encoding rotations).", 4)
if len(DS_STATS): table_generic(DS_STATS, [c for c in DS_STATS.columns if c in ("dataset", "rows_raw", "duplicates", "rows_used", "rows_raw_train", "rows_raw_test", "duplicates_train", "duplicates_test", "features", "split_n_train", "split_n_val", "split_n_test", "split_train_pos", "split_test_pos", "split_d_full")], "T_datasets", "Dataset statistics.", 4)
CIRCUIT_TABLE.to_csv(TABDIR / "circuit_table_full.csv", index=False)

# %% [markdown]
# ## Cell 25 — Export: Drive folder, results.zip (browser download) and the PASTE_BACK block

# %%
def r4(o):
    if isinstance(o, (float, np.floating)): return None if (o != o or math.isinf(o)) else round(float(o), 4)
    if isinstance(o, dict): return {str(k): r4(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [r4(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    return o

def ms_str(m, s): return None if m != m else (f"{m:.4f}±{s:.4f}" if s == s else f"{m:.4f}")

def build_paste(level):
    mets = [["accuracy", "precision", "recall", "f1", "f1_macro", "mcc", "roc_auc", "pr_auc", "mse", "ece"], ["accuracy", "recall", "precision", "f1", "mcc", "roc_auc", "mse", "ece"], ["accuracy", "f1", "mse"], ["accuracy", "f1"]][min(level, 3)]
    P = dict(mode="QUICK" if CFG.quick else "FULL", config={k: v for k, v in asdict(CFG).items() if k not in ("kdd_urls", "unsw_urls", "unsw_kaggle_slugs", "grid_mlp")}, sources=SOURCES,
             environment=ENV_TXT.replace("\n", "; ")[:600], simulator_validation=SIM_VALID)
    P["dataset_statistics"] = {r["dataset"]: {k: v for k, v in r.items() if k != "dataset" and k.split("_")[0] in ("rows", "duplicates", "features", "split", "dedup", "attack", "test", "partition")} for r in DS_STATS.to_dict("records")} if len(DS_STATS) else {}
    s = RESULTS.get("E1_summary"); P["main_results"] = {}
    if s is not None:
        for ds, g in s.groupby("dataset"): P["main_results"][ds] = {r.model: {m: ms_str(r[f"{m}_mean"], r[f"{m}_std"]) for m in mets} for _, r in g.iterrows()}
    sg = RESULTS.get("E1_significance")
    if sg is not None and len(sg):
        sg = sg[sg.metric.isin(["accuracy", "f1", "mse"] if level < 2 else ["f1"])]
        P["significance_vs_proposed(ds,metric,baseline,mean_diff,p_t_holm,p_wilcoxon_holm,dz)"] = [[r.dataset, r.metric, r.baseline, r.mean_diff, r.p_ttest_holm, r.p_wilcoxon_holm, r.cohen_dz] for r in sg.itertuples()]
        P["significance_note"] = "exact two-sided Wilcoxon p >= 0.0625 when n=5 seeds; Holm over baselines"
    a = RESULTS.get("E3_summary")
    if a is not None and level < 3: P["ablation_highlights(ds,group,setting,acc,f1,mse,params)"] = [[r.dataset, r.group, r.setting, r.accuracy_mean, r.f1_mean, r.mse_mean, r.n_params_mean] for r in a.itertuples()]
    g6, e6 = RESULTS.get("E6_gamma_delta"), RESULTS.get("E6_eta")
    if g6 is not None and len(g6): P["gamma_delta_fit"] = {f"{k[0]}|{k[1]}": {c: ms_str(v[f"{c}_mean"], v[f"{c}_ci95"]) for c in ("gamma", "delta", "delta_env", "r2")} for k, v in aggregate(g6, ["dataset", "optimizer"], ["gamma", "delta", "delta_env", "r2"]).set_index(["dataset", "optimizer"]).iterrows()}
    if e6 is not None and len(e6): P["eta_pso"] = {f"{k[0]}|{k[1]}": {c: ms_str(v[f"{c}_mean"], v[f"{c}_ci95"]) for c in ("eta_geo_mean", "violation_frac")} for k, v in aggregate(e6, ["dataset", "scope"], ["eta_geo_mean", "violation_frac"]).set_index(["dataset", "scope"]).iterrows()}
    ex = RESULTS.get("E5_exponents")
    if ex is not None: P["complexity_exponents(name,exp,r2)"] = [[r.quantity, r.exponent, r.r2] for r in ex.itertuples()]
    if xi is not None and len(xi): P["xi_means(ds,model,params,evals_fwd,evals_ps,seconds,train_flops,infer_flops)"] = [[r.dataset, r.model] + [f"{getattr(r, 'xi_' + c + '_mean'):.3g}" for c in ("params", "train_evals_fwd", "train_evals_paramshift", "train_seconds", "train_flops", "infer_flops")] for r in xi.itertuples()] if level < 2 else "see Xi csv"
    bp = RESULTS.get("E4a_barren_plateau")
    if bp is not None and len(bp): P["barren_plateau(arch,Q,var_grad,var_loss)"] = [[r.arch, r.Q, r.var_grad_mean, r.var_loss] for r in bp.itertuples()]
    fc = RESULTS.get("E8_forecasting")
    if fc is not None and len(fc): P["forecasting_test(model,mse,mae)"] = [[k[0] if isinstance(k, tuple) else k, v.mse_mean, v.mae_mean] for k, v in aggregate(fc, ["model"], ["mse", "mae"]).set_index("model").iterrows()]; P["forecast_tolerance_mse"] = CFG.forecast_tol_mse
    nz = RESULTS.get("E9_noise")
    if nz is not None and len(nz) and level < 2: P["noise(arch,p,acc,f1)"] = [[r.arch, r.p_dep, r.accuracy, r.f1] for r in nz.itertuples()]
    cl = RESULTS.get("E10_claims")
    if cl is not None: P["claims_check"] = [[r.claim[:60], r.verdict, str(r.support)[:(260 if level < 2 else 120)]] for r in cl.itertuples()]
    P["runtime_s"] = round(time.time() - T_START); P["experiment_status"] = STATUS; P["NOT_COMPUTED"] = NOT_COMPUTED
    P["design_choices_left_open_by_paper"] = DESIGN if level < 3 else {k: str(v)[:90] for k, v in DESIGN.items()}
    return r4(P)

def tok(s): return len(s) / 2.8      # conservative chars-per-token (numbers tokenise poorly)
def export_all():
    (ROOT / "config.json").write_text(json.dumps(jsonable(asdict(CFG)), indent=1, default=str)); (ROOT / "environment.txt").write_text(ENV_TXT)
    (ROOT / "design_choices.json").write_text(json.dumps(jsonable(DESIGN), indent=1))
    log("config/environment/design files written; the final zip + PASTE_BACK are produced in Cell 25b")
export_all()

print("\n=== CHECKLIST ===")
for k in ["E1", "E2", "E3", "E4a", "E4b", "E5", "E6", "E7", "E8", "E9", "E10"]: print(f"{k:5s} {STATUS.get(k, 'not run / skipped')}")
print("NOT COMPUTED:", json.dumps(NOT_COMPUTED, indent=1)[:2500])

# %% [markdown]
# ## Cell 25b — Paper-ready export: tidy curve CSVs (pgfplots), figure/table aliases with the manuscript's names, final `PASTE_BACK` block (fixed top-level keys) and results.zip
# Every CSV carries `data_source` and `mode`. Figures: `fig_*.pdf/png`; tables: `tab_*.tex`; curves: `curves_*.csv`, `decision_landscape_grid.csv`.

# %%
def wcsv(df, name):
    df = df.copy(); df["data_source"] = df["dataset"].map(lambda d: data_source(d) if d in DATASET_NAMES else OVERALL_SOURCE()) if "dataset" in df.columns else OVERALL_SOURCE()
    df["mode"] = MODE_STR; df.to_csv(ROOT / name, index=False); log(f"saved {name} ({len(df)} rows)"); return df

def _std_curve(curves, col, n=60):
    cur = [np.array(c, float) for c in curves if c is not None and len(c) > 2]
    if not cur: return None
    lo = max(1.0, min(c[1, col] for c in cur)); hi = min(c[-1, col] for c in cur)
    if hi <= lo: return None
    xs = np.geomspace(lo, hi, n); ys = np.array([np.interp(xs, c[:, col], c[:, 1]) for c in cur]); return xs, ys.mean(0), ys.std(0)

def make_curve_csvs():
    # ROC / PR
    rows = []
    for ds in AVAILABLE:
        for m in FIG_MODELS:
            for kind in ("roc", "pr"):
                c = _mean_curve(ds, m, CFG.seeds, kind)
                if c is None: continue
                rows += [dict(dataset=ds, model=m, curve=kind, x=x, y_mean=a, y_std=b) for x, a, b in zip(c[0][::2], c[1][::2], c[2][::2])]
    wcsv(pd.DataFrame(rows), "curves_roc_pr.csv")
    # calibration (10 equal-width bins, pooled over seeds)
    rows = []
    for ds in AVAILABLE:
        for m in ["MLP-Q", "VQNN-Adam", "F-IVQNN-Adam", "F-IVQNN-PSO"]:
            pr = [load_preds(ds, m, s) for s in CFG.seeds]; pr = [p for p in pr if p is not None]
            if not pr: continue
            y, p = np.concatenate([a[0] for a in pr]), np.concatenate([a[1] for a in pr]); idx = np.minimum((p * 10).astype(int), 9)
            for b in range(10):
                k = idx == b
                if k.sum(): rows.append(dict(dataset=ds, model=m, bin=b, bin_center=(b + .5) / 10, mean_pred=p[k].mean(), frac_pos=y[k].mean(), count=int(k.sum())))
    CAL = wcsv(pd.DataFrame(rows), "curves_calibration.csv")
    # convergence PSO vs Adam (central runs)
    rows = []
    for ds in AVAILABLE:
        hp = get_hp(ds)
        for arch in ("ivqnn", "vqnn"):
            for opt in ("adam", "pso"):
                cur = [STORE.get(f"Q|{ds}|{model_label(arch, opt, False, {})}|s{s}|{_fp(CFG, hp)}") for s in CFG.seeds]; cur = [c["extra"].get("curve") for c in cur if c]
                for col, acc in ((2, "param_shift"), (0, "forward")):
                    r = _std_curve(cur, col)
                    if r: rows += [dict(dataset=ds, arch=arch.upper(), optimizer=opt, accounting=acc, evals=x, loss_mean=a, loss_std=b) for x, a, b in zip(*r)]
    wcsv(pd.DataFrame(rows), "curves_convergence_pso_vs_adam.csv")
    # federated rounds (E1 default runs + E2 configurations)
    rows = []
    for ds in AVAILABLE:
        hp = get_hp(ds)
        for opt in ("pso", "adam"):
            runs = [STORE.get(f"Q|{ds}|{model_label('ivqnn', opt, True, {})}|s{s}|{_fp(CFG, hp)}") for s in CFG.seeds]; runs = [r for r in runs if r]
            if not runs: continue
            for rd in range(len(runs[0]["extra"]["rounds"])):
                gl = [r["extra"]["rounds"][rd]["global_loss"] for r in runs]; va = [r["extra"]["rounds"][rd]["val_acc"] for r in runs]
                rows.append(dict(dataset=ds, config="iid_default", optimizer=opt, round=rd, global_loss_mean=np.mean(gl), global_loss_std=np.std(gl), val_acc_mean=np.mean(va), val_acc_std=np.std(va)))
    e2c = RESULTS.get("E2_round_curves")
    if e2c is not None and len(e2c):
        g = e2c.groupby(["dataset", "config", "optimizer", "round"]).agg(global_loss_mean=("global_loss", "mean"), global_loss_std=("global_loss", "std"), val_acc_mean=("val_acc", "mean"), val_acc_std=("val_acc", "std")).reset_index()
        rows += g.to_dict("records")
    FED = wcsv(pd.DataFrame(rows), "curves_federated_rounds.csv")
    # barren plateau
    bp = RESULTS.get("E4a_barren_plateau")
    if bp is not None and len(bp): wcsv(bp[["dataset", "arch", "Q", "n_params", "n_inits", "var_loss", "var_grad_mean", "var_grad_first", "mean_abs_grad"]], "curves_barren_plateau.csv")
    else: wcsv(pd.DataFrame([dict(status="NOT COMPUTED", reason=NOT_COMPUTED.get("E4a_barren_plateau", "E4a did not run"))]), "curves_barren_plateau.csv")
    # scaling
    sc = RESULTS.get("E5_complexity")
    if sc is not None: wcsv(sc[["sweep", "entangle", "arch", "Q", "L", "Qp", "n_circuit_params", "n_gates", "n_1q", "n_2q", "depth", "t_P1_n64", "t_P20_n64"]], "curves_scaling.csv")
    # forecast
    f = ROOT / "preds" / "E8_forecast_preds.npz"
    if f.exists():
        z = np.load(f); n = min(200, len(z["y_true"])); wcsv(pd.DataFrame({"step": np.arange(n), **{k: z[k][:n] for k in z.files}}), "curves_forecast.csv")
    else: wcsv(pd.DataFrame([dict(status="NOT COMPUTED", reason=NOT_COMPUTED.get("E8_forecasting", "E8 did not run"))]), "curves_forecast.csv")
    # decision landscape grid
    if LAND_ROWS:
        g = pd.concat(LAND_ROWS); g["note"] = "2-D PCA plane of the Q-dim angle features; other components held at the train-PCA mean (slice, not the full surface)"; wcsv(g, "decision_landscape_grid.csv")
    else: wcsv(pd.DataFrame([dict(status="NOT COMPUTED", reason="F4 did not run")]), "decision_landscape_grid.csv")
    return CAL, FED

def fig_roc_pr():
    fig, axs = plt.subplots(2, len(AVAILABLE), figsize=(DOUBLE, 5.2), squeeze=False)
    for j, ds in enumerate(AVAILABLE):
        for i, m in enumerate(FIG_MODELS):
            for r, kind in enumerate(("roc", "pr")):
                c = _mean_curve(ds, m, CFG.seeds, kind)
                if c is None: continue
                axs[r][j].plot(c[0], c[1], label=m if (r == 0 and j == 0) else None, color=OKABE[i % 8], ls="-" if i < 5 else "--"); axs[r][j].fill_between(c[0], c[1] - c[2], c[1] + c[2], color=OKABE[i % 8], alpha=0.12, lw=0)
        axs[0][j].set_title(ds); axs[0][j].set_xlabel("FPR"); axs[0][j].set_ylabel("TPR"); axs[1][j].set_xlabel("Recall"); axs[1][j].set_ylabel("Precision")
    axs[0][0].legend(frameon=False, fontsize=6, loc="lower right"); fig.tight_layout(); savefig(fig, "fig_roc_pr")

def fig_fed_rounds(FED):
    if FED is None or not len(FED): return
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.6)); ds = FED.dataset.iloc[0]; g = FED[(FED.dataset == ds) & (FED.optimizer == "pso")]
    for i, (cfgn, q) in enumerate(g.groupby("config")):
        axs[0].plot(q["round"], q.global_loss_mean, label=cfgn, color=OKABE[i % 8]); axs[1].plot(q["round"], q.val_acc_mean, color=OKABE[i % 8])
    a = FED[(FED.dataset == ds) & (FED.optimizer == "adam") & (FED.config == "iid_default")]
    if len(a): axs[0].plot(a["round"], a.global_loss_mean, color="k", ls="--", label="Adam (iid)"); axs[1].plot(a["round"], a.val_acc_mean, color="k", ls="--")
    axs[0].set_xlabel("round"); axs[0].set_ylabel("global loss"); axs[1].set_xlabel("round"); axs[1].set_ylabel("validation accuracy"); axs[0].legend(frameon=False, fontsize=6); axs[0].set_title(f"F-IVQNN-PSO, {ds}")
    fig.tight_layout(); savefig(fig, "fig_fed_rounds")

def alias_outputs():
    for a, b in [("F3_reliability", "fig_calibration"), ("F4_decision_landscape", "fig_decision_landscape"), ("F5_convergence", "fig_convergence"), ("F6_barren_plateau", "fig_barren_plateau"), ("F7_complexity_xi", "fig_scaling"), ("F9_forecast", "fig_forecast")]:
        for ext in ("pdf", "png"):
            if (FIGDIR / f"{a}.{ext}").exists(): shutil.copy(FIGDIR / f"{a}.{ext}", FIGDIR / f"{b}.{ext}")
    mains = [TABDIR / f"T_main_{d}.tex" for d in AVAILABLE if (TABDIR / f"T_main_{d}.tex").exists()]
    if mains: (TABDIR / "tab_main_results.tex").write_text("\n\n".join(p.read_text() for p in mains))
    for a, b in [("T_ablations", "tab_ablation"), ("T_complexity", "tab_complexity"), ("T_xi", "tab_efficiency"), ("T_hyperparameters", "tab_hyperparameters"), ("T_circuits", "tab_circuit_spec"), ("T_datasets", "tab_datasets")]:
        if (TABDIR / f"{a}.tex").exists(): shutil.copy(TABDIR / f"{a}.tex", TABDIR / f"{b}.tex")
        else: (TABDIR / f"{b}.tex").write_text(f"% data_source={OVERALL_SOURCE()} mode={MODE_STR}\n% NOT COMPUTED: source experiment produced no table\n")

def _ms3(r, m): return [r4(r[f"{m}_mean"]), r4(r[f"{m}_std"]), r4(r[f"{m}_ci95"])]

def limitations():
    L = []
    L.append(f"mode={MODE_STR}; training subsample <= {CFG.n_train_max} (balanced={CFG.balance_train}), val <= {CFG.n_val_max}, test <= {CFG.n_test_max} stratified rows; dedup={CFG.dedup}")
    L.append(f"seeds={len(CFG.seeds)} (E1) / {len(CFG.seeds_small)} (E2,E3,E8,E9); Wilcoxon cannot reach p<0.0625 with 5 seeds; hyper-parameters tuned on seed {CFG.seeds[0]} validation split only (3 candidates per method)")
    L.append(f"classical statevector simulation only (Q<={max(list(CFG.sweep_Q) + list(CFG.cplx_Q))}, 2^Q memory); gateset={CFG.gateset} (no RZ) unless changed; noise only in E9 (density matrix, Q=4)")
    L.append("PSO fitness uses one local mini-batch per round (noisy); Adam uses exact backprop through the simulator, so parameter-shift evaluation counts are analytic equivalents, not executed")
    L.append("PCA to Q components discards variance: " + "; ".join(f"{d}: {get_split(d, CFG.seeds[0]).info['pca_var_explained_Qmax']:.3f} (first q_max comps)" for d in AVAILABLE))
    if not SIM_VALID.get("max_err_pennylane"): L.append("PennyLane cross-check NOT run/available; simulator validated against a dense NumPy reference only")
    if not HAS_TORCH: L.append("PyTorch unavailable: Adam, E4a gradients, LSTM were not run")
    L.append(f"simulator backend used for PSO: {BACKEND}")
    L += [f"{k}: {v}" for k, v in STATUS.items() if str(v).startswith("FAILED")]
    for d in AVAILABLE:
        st_ = BASE[d]["stats"]
        if d == "unsw_nb15" and not st_.get("partition_matches_official", True): L.append("UNSW-NB15 partition sizes differ from the official 175,341/82,332 split")
        if st_.get("test_rows_identical_to_a_train_row"): L.append(f"{d}: {st_['test_rows_identical_to_a_train_row']} test rows identical to a train row (official partition kept)")
    cl = RESULTS.get("E10_claims")
    if cl is not None: L += [f"claim not supported as stated: {r.claim[:70]} -> {r.verdict}" for r in cl.itertuples() if r.verdict != "PASS"]
    return L

def build_paste2(level):
    mets = [["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc", "mse", "ece"], ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "mse", "ece"], ["accuracy", "recall", "precision", "f1", "mse", "ece"], ["accuracy", "f1", "mse", "ece"]][min(level, 3)]
    PB = {}
    PB["run_info"] = dict(mode=MODE_STR, seeds=list(CFG.seeds), seeds_secondary=list(CFG.seeds_small), data_source={d: data_source(d) for d in CFG.datasets if d in AVAILABLE}, datasets_not_loaded=[d for d in CFG.datasets if d not in AVAILABLE],
                          runtime_s=round(time.time() - T_START), simulator_backend=BACKEND, libraries=ENV_TXT.replace("\n", "; "), budget_planner_changes=DESIGN.get("budget_degradations", []), simulator_validation=SIM_VALID,
                          synthetic_data_warning=("RESULTS FROM SYNTHETIC DATA MUST NOT BE USED IN THE PAPER" if any(data_source(d) == "SYNTHETIC_FALLBACK" for d in AVAILABLE) else None))
    PB["datasets"] = {}
    for d in AVAILABLE:
        b = BASE[d]; S = get_split(d, CFG.seeds[0])
        PB["datasets"][d] = dict(rows_after_cleaning=(len(b["y"]) if d == "kddcup99" else [len(b["y"]), len(b["yte"])]), dedup=CFG.dedup, raw_stats={k: v for k, v in b["stats"].items() if k != "source"},
                                 class_balance=dict(train_attack=S.info["train_pos"], val_attack=S.info["val_pos"], test_attack=S.info["test_pos"]), features_raw=int(b["X"].shape[1]), features_after_encoding=S.info["d_full"], features_after_pca=CFG.Q,
                                 split_sizes=dict(train=S.info["n_train"], val=S.info["n_val"], test=S.info["n_test"]), subsampling=dict(n_train_max=CFG.n_train_max, n_val_max=CFG.n_val_max, n_test_max=CFG.n_test_max, balanced_train=CFG.balance_train))
    c8 = build_circuit("ivqnn", CFG.Q, CFG.L, CFG.Qp, CFG.gateset, CFG.entangle, CFG.conv_share); ct = CIRCUIT_TABLE if level < 2 else (CIRCUIT_TABLE[CIRCUIT_TABLE.L.isin([1, 3])] if level < 4 else CIRCUIT_TABLE[(CIRCUIT_TABLE.L == 3) & CIRCUIT_TABLE.Q.isin([4, 8, 10])])
    PB["circuit_spec"] = dict(encoding=DESIGN["encoding"], feature_to_qubit=f"PCA component j -> qubit j (j=0..{CFG.Q-1}); RY(x_j), x_j in [0,pi]", gate_set=f"{CFG.gateset}: RY, CNOT, CRY (pooling)" + (", RZ, CRZ" if CFG.gateset == "full" else ""), entanglers=f"{CFG.entangle} (CNOT ring / brick conv blocks); conv weight sharing={CFG.conv_share}",
                              observable=DESIGN["observable"], observable_qubit=dict(VQNN=0, IVQNN=c8.obs), pooling=dict(definition=DESIGN["layers_IVQNN"], kept_qubits=c8.kept, traced_out_qubits=[q for q in range(CFG.Q) if q not in c8.kept], equivalence=DESIGN["pooling_equivalence"]),
                              simulation="classical statevector (float32; NumPy/PyTorch), validated vs dense reference" + (" and PennyLane" if SIM_VALID.get("max_err_pennylane") is not None else "; PennyLane check not run"), noise_model="none in E1-E8; E9: single-qubit depolarising after every gate on touched qubits, density matrix, Q=4, L=2, p in " + str(list(CFG.noise_p)),
                              counts_columns=["model", "Q", "L", "Qp", "circuit_params", "params_with_head", "gates", "n1q", "n2q", "depth"], counts=[[r.model, r.Q, r.L, r.Qp, r.n_circuit_params, r.n_params_with_head, r.n_gates, r.n_1q, r.n_2q, r.depth] for r in ct.itertuples()])
    pd_ = lambda a: int(CIRCUIT_TABLE[(CIRCUIT_TABLE.Q == CFG.Q) & (CIRCUIT_TABLE.L == CFG.L) & (CIRCUIT_TABLE.model == a)].n_params_with_head.iloc[0]) if ((CIRCUIT_TABLE.Q == CFG.Q) & (CIRCUIT_TABLE.L == CFG.L)).any() else None
    PB["pso_spec"] = dict(particle_dimension=dict(IVQNN=pd_("IVQNN"), VQNN=pd_("VQNN")), P=CFG.P, T=CFG.T, omega=CFG.w, omega_tuned_grid=list(CFG.grid_pso_w), phi1=CFG.phi1, phi2=CFG.phi2, velocity_clamp=CFG.vmax, fitness="F=1/(L+eps), eps=%g, L=mean BCE" % CFG.eps,
                          data_used=f"mini-batch of {CFG.pso_batch} local samples" if CFG.pso_batch else "full local data", swarm_init=DESIGN["swarm_init"], minibatching=DESIGN["pso_minibatching"], probability_mapping=DESIGN["output_mapping"], tuned_w={d: get_hp(d).get("pso_w") for d in AVAILABLE})
    ex_sizes = None
    for d in AVAILABLE:
        r = STORE.get(f"Q|{d}|F-IVQNN-PSO|s{CFG.seeds[0]}|{_fp(CFG, get_hp(d))}")
        if r: ex_sizes = {d: r["extra"].get("client_sizes")}; break
    PB["fed_spec"] = dict(C=CFG.C, R=CFG.R, C_values_tested=sorted(set(list(CFG.fed_C_list) + [CFG.C])), partitions=["iid (unequal sizes)"] + [f"dirichlet alpha={a}" for a in CFG.alpha_list], example_client_sizes=ex_sizes, client_sampling="all clients participate every round; weights n_c/N (asserted to sum to 1)", local_restart_rule=DESIGN["swarm_init"], local_adam_steps=CFG.local_adam_steps)
    s = RESULTS.get("E1_summary"); PB["main_results"] = {}
    if s is not None:
        for d, g in s.groupby("dataset"):
            if level < 3: PB["main_results"][d] = {r.model: {**{m: (_ms3(r, m) if level < 2 else _ms3(r, m)[:2]) for m in mets}, **{c: r4(r[f"{c}_mean"]) for c in (("n_params", "evals_fwd", "evals_ps", "train_time") if level < 2 else ("n_params", "train_time"))}} for _, r in g.iterrows()}
            else: PB["main_results"][d] = {r.model: [v for m in mets for v in _ms3(r, m)[:2]] + [r4(r["n_params_mean"]), r4(r["train_time_mean"])] for _, r in g.iterrows()}
        PB["main_results_format"] = ("metric: [mean, std" + (", ci95]" if level < 2 else "] (ci95 in results.zip tables)") + "; costs: mean over seeds") if level < 3 else "list per model = [" + ", ".join(f"{m}_mean, {m}_std" for m in mets) + ", n_params, train_time_s] (ci95, evals, other metrics in results.zip)"
    sg = RESULTS.get("E1_significance")
    PB["significance"] = dict(note="paired tests across seeds, Holm-corrected over baselines; with 5 seeds the exact two-sided Wilcoxon p cannot go below 0.0625", columns=["dataset", "metric", "baseline", "mean_diff", "p_wilcoxon_holm", "p_ttest_holm", "cohen_dz"],
                              rows=[[r.dataset, r.metric, r.baseline, r.mean_diff, r.p_wilcoxon_holm, r.p_ttest_holm, r.cohen_dz] for r in sg[sg.metric.isin(["accuracy", "f1", "mse"] if level < 2 else ["f1"])].itertuples()] if sg is not None and len(sg) else "NOT COMPUTED (needs >=3 paired seeds)")
    a3, a2 = RESULTS.get("E3_summary"), RESULTS.get("E2_federated")
    PB["ablations"] = dict(columns=["dataset", "group", "setting", "acc", "f1", "mse", "params"], E3=[[r.dataset, r.group, r.setting, r.accuracy_mean, r.f1_mean, r.mse_mean, r.n_params_mean] for r in (a3 if level < 3 else a3[a3.group.isin(["pooling", "optimizer", "init"])]).itertuples()] if a3 is not None else "NOT COMPUTED",
                           noniid_and_clients=[[r.dataset, r.config, r.optimizer, r.accuracy_mean, r.f1_mean] for r in aggregate(a2, ["dataset", "config", "optimizer"], ["accuracy", "f1"]).itertuples()] if a2 is not None and len(a2) else "NOT COMPUTED")
    g6, e6 = RESULTS.get("E6_gamma_delta"), RESULTS.get("E6_eta"); cv = {}
    if g6 is not None and len(g6):
        ag = aggregate(g6, ["dataset", "optimizer"], ["gamma", "delta", "delta_env", "r2"]); cv["gamma_delta"] = {f"{r.dataset}|{r.optimizer}": {c: [r4(r[f"{c}_mean"]), r4(r[f"{c}_ci95"])] for c in ("gamma", "delta", "delta_env", "r2")} for _, r in ag.iterrows()}
        cv["flag_gamma_not_positive"] = bool((g6.gamma < 1e-3).any())
    if e6 is not None and len(e6):
        ae = aggregate(e6, ["dataset", "scope"], ["eta_geo_mean", "violation_frac"]); cv["eta"] = {f"{r.dataset}|{r.scope}": {"eta": [r4(r.eta_geo_mean_mean), r4(r.eta_geo_mean_ci95)], "violation_frac": r4(r.violation_frac_mean)} for _, r in ae.iterrows()}
        cv["flag_pso_contraction_violated_in_some_iterations"] = bool((e6.violation_frac > 0).any()); cv["eta_note"] = "Theta* proxy = final swarm best (biased towards contraction)"
    PB["convergence"] = cv or "NOT COMPUTED"
    bp, sl = RESULTS.get("E4a_barren_plateau"), RESULTS.get("E4a_slopes")
    PB["barren_plateau"] = dict(columns=["arch", "Q", "var_grad_mean", "var_loss"], rows=[[r.arch, r.Q, r.var_grad_mean, r.var_loss] for r in bp.itertuples()], log2_slopes=[[r.arch, r.quantity, r.log2_slope_per_qubit] for r in sl.itertuples()] if sl is not None else None) if bp is not None and len(bp) else dict(status="NOT COMPUTED", reason=NOT_COMPUTED.get("E4a_barren_plateau", "E4a did not run"))
    ex = RESULTS.get("E5_exponents"); PB["complexity"] = dict(columns=["quantity", "exponent", "r2"], rows=[[r.quantity, r.exponent, r.r2] for r in ex.itertuples()], note="ring vs full(all-to-all) entanglers listed separately; threshold row = numeric root of x^2+x=1") if ex is not None else "NOT COMPUTED"
    xi = RESULTS.get("E7_xi"); defs = ("params", "train_evals_fwd", "train_evals_paramshift", "train_seconds", "train_flops", "infer_flops")
    PB["efficiency_Xi"] = dict(definition="Xi = accuracy / cost; all cost definitions reported, none selected", columns=["dataset", "model", *defs], rows=[[r.dataset, r.model] + [f"{getattr(r, 'xi_' + c + '_mean'):.3g}" for c in defs] for r in (xi if level < 3 else xi[xi.model.isin(["LR-Q", "MLP-Q", "VQNN-Adam", "IVQNN-Adam", "F-IVQNN-PSO"])]).itertuples()]) if xi is not None and len(xi) else "NOT COMPUTED"
    fc = RESULTS.get("E8_forecasting")
    PB["forecasting"] = dict(definition=f"s_t = attack ratio per {CFG.window_sec}s window (windows with >= {CFG.min_win_count} flows kept; samples only from consecutive windows); k={CFG.Q-1} lags; target s_(t+1); chronological 60/20/20 split", tolerance_mse=CFG.forecast_tol_mse,
                             baselines=["persistence", "moving average", "exp. smoothing", "ARIMA(2,0,1)", "MLP", "LSTM"], columns=["model", "mse_mean", "mae_mean"],
                             rows=[[r.model, r.mse_mean, r.mae_mean] for r in aggregate(fc, ["model"], ["mse", "mae"]).itertuples()] if fc is not None and len(fc) else f"NOT COMPUTED: {NOT_COMPUTED.get('E8_forecasting', 'E8 did not run')}")
    e1 = RESULTS.get("E1_main"); cal = {}
    if e1 is not None:
        for (d, m), v in e1.groupby(["dataset", "model"]).ece: cal.setdefault(d, {})[m] = [r4(v.mean()), r4(v.std())]
    PB["calibration"] = dict(ece_mean_std=cal, ece_threshold=CFG.ece_thr, bins="15 equal-width bins on P(attack)")
    if level < 2 and CAL_DF is not None and len(CAL_DF): PB["calibration"]["reliability_bins(ds|model -> [mean_pred,frac_pos])"] = {f"{k[0]}|{k[1]}": [[r4(a), r4(b)] for a, b in zip(v.mean_pred, v.frac_pos)] for k, v in CAL_DF.groupby(["dataset", "model"]) if k[1] in ("F-IVQNN-PSO", "MLP-Q")}
    cl = RESULTS.get("E10_claims"); PB["claims_check"] = [[r.claim[:70], r.verdict, str(r.support)[:(240 if level < 2 else (110 if level < 4 else (80 if level == 4 else 60)))]] for r in cl.itertuples()] if cl is not None else "NOT COMPUTED"
    PB["not_computed"] = NOT_COMPUTED
    PB["design_choices"] = DESIGN if level < 3 else {k: str(v)[:(80 if level == 3 else (50 if level == 4 else 30))] for k, v in DESIGN.items()}
    PB["limitations_observed"] = limitations() if level < 4 else [x[:(140 if level == 4 else 90)] for x in limitations()]
    return r4(PB)

def finalize():
    for lvl in range(6):
        txt = json.dumps(build_paste2(lvl), separators=(",", ":"), default=str)
        if tok(txt) <= 6000 or lvl == 5: break
    (ROOT / "paste_back.json").write_text(txt)
    zp = ROOT.parent / "results.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ROOT.rglob("*"):
            if f.is_file() and "ckpt_" not in str(f) and f.suffix != ".zip" and "/data/" not in str(f): z.write(f, f.relative_to(ROOT.parent))
    try:
        from google.colab import files; files.download(str(zp))
    except Exception as e: log("browser download unavailable here:", type(e).__name__, "| zip saved at", zp)
    print("=== PASTE_BACK_START ==="); print(txt); print("=== PASTE_BACK_END ===")
    print(f"(compaction level {lvl}; ~{tok(txt):.0f} tokens; mode={MODE_STR}; data_source={OVERALL_SOURCE()})")
    if any(data_source(d) == "SYNTHETIC_FALLBACK" for d in AVAILABLE): print("RESULTS FROM SYNTHETIC DATA MUST NOT BE USED IN THE PAPER")

CAL_DF = None
try:
    CAL_DF, FED_DF = make_curve_csvs()
    for _fn in (fig_roc_pr, lambda: fig_fed_rounds(FED_DF)):
        try: _fn()
        except Exception as e: log(f"figure failed: {type(e).__name__}: {str(e)[:120]}")
    alias_outputs()
except Exception as e:
    log("curve/figure export failed:", type(e).__name__, str(e)[:200]); log(traceback.format_exc()[-800:])
finalize()

# %% [markdown]
# ## Cell 26 — Re-running a single experiment
# 1. Run Cells 1–12 (config, loaders, simulator, helpers; finished runs are loaded from the Drive checkpoints, so this is fast).
# 2. Run `run_one("E3")` below (any of `E1 E2 E3 E4a E4b E5 E6 E7 E8 E9 E10`). To recompute from scratch, delete the matching files in `…/ckpt_quick|ckpt_full/` or set `CFG.resume = False`.
# 3. Re-run Cells 23–25 to refresh figures, tables and the PASTE_BACK block.

# %%
EXPERIMENTS = {"E1": exp_E1, "E2": exp_E2, "E3": exp_E3, "E4a": exp_E4a, "E4b": exp_E4b, "E5": exp_E5, "E6": exp_E6, "E7": exp_E7, "E8": exp_E8, "E9": exp_E9, "E10": exp_E10}
def run_one(name):
    load_results(); run_experiment(name, EXPERIMENTS[name]); print(name, STATUS.get(name))
# run_one("E3")
