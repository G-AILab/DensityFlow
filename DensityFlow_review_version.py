import copy
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import deque

from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import xgboost as xgb

from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint_normal


# -------------------- Global config & grids --------------------
DATASET = "spirals"  # circles  moons  spirals  checkerboard
DATASET_NOISE = 0.05
N_SAMPLES = 3000
DIAG = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 800
S_STEPS = 100
BATCH = 64
ACCUM_STEPS = 1
LR_G = 1e-3
LR_C = 1e-4
MIN_LR = 1e-6
WARMUP_EPOCHS = 20
EMA_DECAY = 0.999
CLAMP_X = 2.0  # Placeholder
LAMBDA_VALIDITY = 1.0  # Placeholder
LAMBDA_COST = 0.5  # Placeholder
lambda_nod = 0.1  # Placeholder
LAMBDA_VALIDITY_GRID = [1.0]
LAMBDA_COST_GRID = [0.2, 0.4, 0.6]
LAMBDA_NOD_GRID = [0.0, 0.1, 0.3]

C_OUT_DIM = 3
LAMBDA_NOISE = 1.0  # Placeholder
NOISE_RATIO = 0.2

PATIENCE = 1000  # Placeholder
MIN_DELTA = 1e-5

FT_EPOCHS = 20
FT_INNER_ITER = 3
INFERENCE_SAMPLES = 30
INFERENCE_NOISE = 0.05

ODE_METHOD = 'dopri5'
ODE_ATOL_TRAIN = 1e-3
ODE_RTOL_TRAIN = 1e-3
ODE_ATOL_TEST = 1e-4
ODE_RTOL_TEST = 1e-4

ALPHA_COST = 0.0  # Placeholder

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f"Using device: {device}")
print(f"Loading dataset: {DATASET}")


# -------------------- Data generation --------------------
def make_intertwined_spirals(n_samples=3000, noise=0.1, n_rotations=2.5, seed=42):
    np.random.seed(seed)
    n_per_class = n_samples // 2
    theta = np.sqrt(np.random.rand(n_per_class)) * n_rotations * 2 * np.pi
    r0 = theta / (n_rotations * 2 * np.pi) * 5
    x0 = r0 * np.cos(theta)
    y0 = r0 * np.sin(theta)
    theta1 = theta + np.pi
    r1 = theta / (n_rotations * 2 * np.pi) * 5
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    X0 = np.column_stack([x0, y0])
    X1 = np.column_stack([x1, y1])
    X0 += noise * np.random.randn(n_per_class, 2)
    X1 += noise * np.random.randn(n_per_class, 2)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    idx = np.random.permutation(n_samples)
    X = X[idx]
    y = y[idx]
    return X, y


def make_concentric_circles(n_samples=3000, noise=0.05, factor=0.5, seed=42):
    np.random.seed(seed)
    n_per_class = n_samples // 2
    angles_outer = 2 * np.pi * np.random.rand(n_per_class)
    radius_outer = 1.0 + noise * np.random.randn(n_per_class)
    x_outer = radius_outer * np.cos(angles_outer)
    y_outer = radius_outer * np.sin(angles_outer)
    angles_inner = 2 * np.pi * np.random.rand(n_per_class)
    radius_inner = factor + noise * np.random.randn(n_per_class)
    x_inner = radius_inner * np.cos(angles_inner)
    y_inner = radius_inner * np.sin(angles_inner)
    X0 = np.column_stack([x_outer, y_outer])
    X1 = np.column_stack([x_inner, y_inner])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    idx = np.random.permutation(n_samples)
    X = X[idx]
    y = y[idx]
    return X, y


def make_checkerboard(n_samples=3000, n_clusters=3, noise=0.1, seed=42):
    np.random.seed(seed)
    X_list = []
    y_list = []
    samples_per_cluster = n_samples // (n_clusters ** 2)
    for i in range(n_clusters):
        for j in range(n_clusters):
            label = (i + j) % 2
            center_x = (i + 0.5) / n_clusters * 6 - 3
            center_y = (j + 0.5) / n_clusters * 6 - 3
            cluster_x = center_x + noise * np.random.randn(samples_per_cluster)
            cluster_y = center_y + noise * np.random.randn(samples_per_cluster)
            X_list.append(np.column_stack([cluster_x, cluster_y]))
            y_list.append(np.full(samples_per_cluster, label))
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    remaining = n_samples - len(y)
    if remaining > 0:
        extra_x = np.random.uniform(-3, 3, (remaining, 2))
        extra_y = np.random.randint(0, 2, remaining)
        X = np.vstack([X, extra_x])
        y = np.hstack([y, extra_y])
    idx = np.random.permutation(len(y))
    X = X[idx]
    y = y[idx]
    return X, y

# -------------------- Dataset split: train / val / test --------------------
if DATASET == 'moons':
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.05, random_state=42)
elif DATASET == 'circles':
    X, y = make_concentric_circles(n_samples=N_SAMPLES, noise=0.05, seed=42)
elif DATASET == 'spirals':
    X, y = make_intertwined_spirals(n_samples=N_SAMPLES, noise=DATASET_NOISE, seed=42)
elif DATASET == 'checkerboard':
    X, y = make_checkerboard(n_samples=N_SAMPLES, noise=DATASET_NOISE, seed=42)
else:
    raise ValueError(f"Unknown dataset: {DATASET}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
y = y.astype(np.int64)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

if X_train_full.size > 0:
    CLAMP_X = 1.2 * float(np.max(np.abs(X_train_full)))

print(f"Train set: {len(X_train)} samples")
print(f"Val set:   {len(X_val)} samples")
print(f"Test set:  {len(X_test)} samples")

# -------------------- Baseline models --------------------
print("\nTraining kNN classifier with simple grid search...")
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}
knn_base = KNeighborsClassifier()
knn = GridSearchCV(knn_base, knn_param_grid, cv=3, n_jobs=10)
knn.fit(X_train_full, y_train_full)
knn_test_acc = knn.score(X_test, y_test)
print(f"kNN best params: {knn.best_params_}")
print(f"kNN test accuracy: {knn_test_acc:.3f}")

print("Training MLP classifier with simple grid search...")
mlp_param_grid = {
    'hidden_layer_sizes': [
        (64, 64),
        (128, 64),
        (128, 64, 64),
        (128, 64, 64, 128)
    ],
    'alpha': [1e-4, 1e-3, 1e-2]
}
mlp_base = MLPClassifier(
    max_iter=500,
    random_state=SEED,
    early_stopping=True
)
mlp = GridSearchCV(mlp_base, mlp_param_grid, cv=3, n_jobs=10)
mlp.fit(X_train_full, y_train_full)
mlp_test_acc = mlp.score(X_test, y_test)
print(f"MLP best params: {mlp.best_params_}")
print(f"MLP test accuracy: {mlp_test_acc:.3f}")

print("Training CatBoost classifier with simple grid search...")
cat_param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.03, 0.1],
    'l2_leaf_reg': [3, 5, 7]
}
cat_base = CatBoostClassifier(
    iterations=500,
    random_seed=SEED,
    verbose=0,
    early_stopping_rounds=50,
    thread_count=3
)
catboost = GridSearchCV(
    cat_base,
    cat_param_grid,
    cv=3,
    n_jobs=3
)
catboost.fit(X_train_full, y_train_full, eval_set=(X_test, y_test))
catboost_test_acc = catboost.score(X_test, y_test)
print(f"CatBoost best params: {catboost.best_params_}")
print(f"CatBoost test accuracy: {catboost_test_acc:.3f}")

print("Training Random Forest with simple grid search...")
rf_param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}
rf_base = RandomForestClassifier(random_state=SEED)
rf = GridSearchCV(rf_base, rf_param_grid, cv=3, n_jobs=10)
rf.fit(X_train_full, y_train_full)
rf_test_acc = rf.score(X_test, y_test)
print(f"Random Forest best params: {rf.best_params_}")
print(f"Random Forest test accuracy: {rf_test_acc:.3f}")

print("Training XGBoost with simple grid search...")
xgb_param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.03, 0.1]
}
xgb_base = xgb.XGBClassifier(
    random_state=SEED,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=2
)
xgboost = GridSearchCV(xgb_base, xgb_param_grid, cv=3, n_jobs=10)
xgboost.fit(X_train_full, y_train_full)
xgboost_test_acc = xgboost.score(X_test, y_test)
print(f"XGBoost best params: {xgboost.best_params_}")
print(f"XGBoost test accuracy: {xgboost_test_acc:.3f}")

print("Training SVM with simple grid search...")
param_grid = {
    'C': [1, 10, 100],
    'gamma': [0.1, 1, 10, 100]
}
svm = SVC(kernel='rbf', probability=True, random_state=SEED)
svm = GridSearchCV(svm, param_grid, cv=3, n_jobs=10)
svm.fit(X_train_full, y_train_full)
svm_test_acc = svm.score(X_test, y_test)
print(f"SVM best params: {svm.best_params_}")
print(f"SVM test accuracy: {svm_test_acc:.3f}")

print("Training TabNet...")
y_train_tabnet = y_train_full
y_test_tabnet = y_test
tabnet = TabNetClassifier(
    optimizer_fn=torch.optim.AdamW,
    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    scheduler_params={"mode": "min", "patience": 5, "factor": 0.5},
    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
    mask_type='sparsemax',
    device_name=device,
    seed=SEED
)
tabnet.fit(
    X_train=X_train_full, y_train=y_train_tabnet,
    eval_set=[(X_test, y_test_tabnet)],
    eval_name=['test'],
    max_epochs=500,
    patience=75,
    batch_size=128,
    virtual_batch_size=64,
    num_workers=0,
    drop_last=False,
    eval_metric=['accuracy']
)
y_pred_tabnet = tabnet.predict(X_test).ravel()
tabnet_test_acc = accuracy_score(y_test, y_pred_tabnet)
print(f"TabNet test accuracy: {tabnet_test_acc:.3f}")

mm_models = {
    "kNN": knn,
    "MLP": mlp,
    "SVM": svm,
    "CatBoost": catboost,
    "RandomForest": rf,
    "XGBoost": xgboost,
    "TabNet": tabnet
}


def make_loader(X_arr, y_arr, batch_size=BATCH, shuffle=True):
    X_t = torch.from_numpy(X_arr)
    y_t = torch.from_numpy(y_arr)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0, pin_memory=True)
    return loader


train_loader = make_loader(X_train, y_train, BATCH, True)
val_loader = make_loader(X_val, y_val, BATCH, True)
test_loader = make_loader(X_test, y_test, BATCH, True)

x_dim = X_train.shape[1]
y_dim = 2


# -------------------- Model definitions --------------------
class ClassifierNet(nn.Module):
    def __init__(self, x_dim=2, y_dim=3, hid=128):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hid = hid
        self.fc1 = nn.utils.spectral_norm(nn.Linear(x_dim, hid))
        self.ln1 = nn.LayerNorm(hid)
        self.fc2 = nn.utils.spectral_norm(nn.Linear(hid, hid))
        self.ln2 = nn.LayerNorm(hid)
        self.fc3 = nn.utils.spectral_norm(nn.Linear(hid, hid))
        self.ln3 = nn.LayerNorm(hid)
        self.fc4 = nn.Linear(hid, y_dim)
        nn.init.xavier_normal_(self.fc1.weight, gain=0.5)
        nn.init.xavier_normal_(self.fc2.weight, gain=0.5)
        nn.init.xavier_normal_(self.fc3.weight, gain=0.5)
        nn.init.xavier_normal_(self.fc4.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = F.silu(self.ln1(self.fc1(x)))
        h = F.silu(self.ln2(self.fc2(h)))
        h = F.silu(self.ln3(self.fc3(h)))
        return self.fc4(h)


class ClassifierNetLite(nn.Module):
    def __init__(self, x_dim=2, y_dim=3, hid=64):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hid = hid
        self.net = nn.Sequential(
            nn.Linear(x_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, y_dim)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class VelocityFieldNet(nn.Module):
    def __init__(self, x_dim, context_dim, hid):
        super().__init__()
        input_dim = x_dim + 1 + context_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, x_dim)
        )

    def forward(self, t, x_t, context):
        t_vec = t.expand(x_t.size(0), 1)
        inp = torch.cat([x_t, t_vec, context], dim=1)
        return self.net(inp)


class GeneratorNet(nn.Module):
    def __init__(self, x_dim=2, y_dim=2, hid=128):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hid = hid
        self.emb_dim = 16
        self.emb = nn.Embedding(y_dim, self.emb_dim)
        self.context_dim = x_dim + self.emb_dim
        self.v_net = VelocityFieldNet(x_dim=x_dim, context_dim=self.context_dim, hid=hid)

        class AugmentedODEFunc(nn.Module):
            def __init__(self, v_net, context):
                super().__init__()
                self.v_net = v_net
                self.context = context

            def forward(self, t, z_t):
                x_t = z_t[:, :-1]
                v_t = self.v_net(t, x_t, self.context)
                dl_dt = v_t.pow(2).sum(dim=1, keepdim=True)
                dz_dt = torch.cat([v_t, dl_dt], dim=1)
                return dz_dt

        self.AugmentedODEFunc = AugmentedODEFunc

    def forward(self, x_src, y_tar):
        y_tar_emb = self.emb(y_tar)
        context = torch.cat([x_src, y_tar_emb], dim=1)
        aug_ode_func = self.AugmentedODEFunc(self.v_net, context)
        z_0 = F.pad(x_src, (0, 1), 'constant', 0.0)
        t_span = torch.tensor([0.0, 1.0]).to(x_src.device)
        path_solution = odeint(
            aug_ode_func, z_0, t_span, method=ODE_METHOD,
            atol=ODE_ATOL_TRAIN, rtol=ODE_RTOL_TRAIN
        )
        z_1 = path_solution[1]
        x_tar = z_1[:, :-1]
        path_cost = z_1[:, -1]
        x_tar = torch.clamp(x_tar, -CLAMP_X, CLAMP_X)
        return x_tar, path_cost


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.device = next(model.parameters()).device
        self.model_class = model.__class__
        self.model_config = {'x_dim': model.x_dim, 'y_dim': model.y_dim, 'hid': model.hid}
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.best_shadow_state = {}
        self.best_loss = float('inf')

    def update(self):
        with torch.no_grad():
            for name, param in self.model.state_dict().items():
                if param.dtype.is_floating_point:
                    self.shadow[name].mul_(self.decay).add_(param, alpha=1 - self.decay)

    def apply_shadow(self):
        self.model.load_state_dict(self.shadow)

    def save_current_shadow_as_best(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_shadow_state = {k: v.clone() for k, v in self.shadow.items()}
            return True
        return False

    def apply_best_shadow(self):
        if not self.best_shadow_state:
            print("  ⚠ Warning: No best shadow saved, using current final shadow.")
            self.apply_shadow()
            return
        print(f"  ✓ Applied best EMA weights (loss: {self.best_loss:.6f})")
        self.model.load_state_dict(self.best_shadow_state)


def warmup_lr(epoch, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


# density score
def nod_score(net_c, x):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    logits = net_c(x)
    logit_noise = logits[:, 2]
    logit_real_max, _ = torch.max(logits[:, :2], dim=1)
    score = logit_noise - logit_real_max
    return score


def build_g_c_models():
    C1 = ClassifierNet(x_dim=x_dim, y_dim=C_OUT_DIM, hid=128).to(device)
    C2 = ClassifierNetLite(x_dim=x_dim, y_dim=C_OUT_DIM, hid=128).to(device)
    G = GeneratorNet(x_dim=x_dim, y_dim=y_dim, hid=128).to(device)

    ema_c1 = EMA(C1, decay=EMA_DECAY)
    ema_c2 = EMA(C2, decay=EMA_DECAY)
    ema_g = EMA(G, decay=EMA_DECAY)

    opt_c1 = torch.optim.AdamW(C1.parameters(), lr=LR_C, weight_decay=1e-4)
    opt_c2 = torch.optim.AdamW(C2.parameters(), lr=LR_C, weight_decay=1e-4)
    opt_g = torch.optim.AdamW(G.parameters(), lr=LR_G, weight_decay=1e-4)

    sched_c1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c1, T_max=EPOCHS, eta_min=MIN_LR)
    sched_c2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c2, T_max=EPOCHS, eta_min=MIN_LR)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=EPOCHS, eta_min=MIN_LR)

    return C1, C2, G, ema_c1, ema_c2, ema_g, opt_c1, opt_c2, opt_g, sched_c1, sched_c2, sched_g


train_diag_template = {
    "epoch": [], "loss_g_mean": [], "loss_g_valid_mean": [], "loss_g_cost_mean": [],
    "loss_c1_class_mean": [], "loss_c1_noise_mean": [],
    "loss_c2_class_mean": [], "loss_c2_noise_mean": [],
    "lr_g": [], "lr_c": []
}


def train_epoch(net_g, net_c1, net_c2, loader, opt_g, opt_c1, opt_c2, ema_g, ema_c1, ema_c2,
                lambda_validity, lambda_cost, lambda_noise, lambda_nod_val, noise_ratio,
                accum_steps=1, diag=False, train_diag=None):
    net_g.train()
    net_c1.train()
    net_c2.train()

    stats = {k: [] for k in ["loss_g", "loss_g_valid", "loss_g_cost",
                             "loss_c1_class", "loss_c1_noise",
                             "loss_c2_class", "loss_c2_noise", "l_nod"]}

    for batch_idx, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        yb_tar = 1 - yb

        noise_batch_size = int(xb.size(0) * noise_ratio)
        xb_noise = (torch.rand(noise_batch_size, x_dim, device=device) * 2 * CLAMP_X) - CLAMP_X
        yb_noise = torch.full((noise_batch_size,), 2, dtype=torch.long, device=device)

        # Train C1
        opt_c1.zero_grad(set_to_none=True)
        loss_c1_class = F.cross_entropy(net_c1(xb), yb)
        loss_c1_noise = F.cross_entropy(net_c1(xb_noise), yb_noise)
        loss_c1 = loss_c1_class + lambda_noise * loss_c1_noise
        loss_c1.backward()
        opt_c1.step()

        # Train C2
        opt_c2.zero_grad(set_to_none=True)
        loss_c2_class = F.cross_entropy(net_c2(xb), yb)
        loss_c2_noise = F.cross_entropy(net_c2(xb_noise), yb_noise)
        loss_c2 = loss_c2_class + lambda_noise * loss_c2_noise
        loss_c2.backward()
        opt_c2.step()

        # Train G
        opt_g.zero_grad(set_to_none=True)
        x_tar, path_cost = net_g(xb, yb_tar)

        logits_tar_c1 = net_c1(x_tar)
        loss_g_valid_c1 = F.cross_entropy(logits_tar_c1, yb_tar, reduction='none')

        logits_tar_c2 = net_c2(x_tar)
        loss_g_valid_c2 = F.cross_entropy(logits_tar_c2, yb_tar, reduction='none')
        loss_g_validity = torch.max(torch.stack([loss_g_valid_c1, loss_g_valid_c2], dim=1), dim=1)[0].mean()

        nod_scores_c1 = nod_score(net_c1, x_tar)
        nod_scores_c2 = nod_score(net_c2, x_tar)
        nod_scores = torch.max(torch.stack([nod_scores_c1, nod_scores_c2], dim=1), dim=1)[0]
        l_nod = F.relu(nod_scores).mean()

        loss_g_cost = path_cost.mean()
        loss_g = lambda_validity * loss_g_validity + lambda_cost * loss_g_cost + lambda_nod_val * l_nod
        loss_g.backward()
        opt_g.step()

        ema_g.update()
        ema_c1.update()
        ema_c2.update()

        if diag and train_diag is not None:
            stats["loss_g"].append(loss_g.item())
            stats["loss_g_valid"].append(loss_g_validity.item())
            stats["loss_g_cost"].append(loss_g_cost.item())
            stats["loss_c1_class"].append(loss_c1_class.item())
            stats["loss_c1_noise"].append(loss_c1_noise.item())
            stats["loss_c2_class"].append(loss_c2_class.item())
            stats["loss_c2_noise"].append(loss_c2_noise.item())
            stats["l_nod"].append(l_nod.item())

    def m(k):
        return float(np.mean(stats[k])) if len(stats[k]) > 0 else 0.0

    if diag and train_diag is not None:
        train_diag["epoch"].append(len(train_diag["epoch"]) + 1)
        train_diag["loss_g_mean"].append(m("loss_g"))
        train_diag["loss_g_valid_mean"].append(m("loss_g_valid"))
        train_diag["loss_g_cost_mean"].append(m("loss_g_cost"))
        train_diag["loss_c1_class_mean"].append(m("loss_c1_class"))
        train_diag["loss_c1_noise_mean"].append(m("loss_c1_noise"))
        train_diag["loss_c2_class_mean"].append(m("loss_c2_class"))
        train_diag["loss_c2_noise_mean"].append(m("loss_c2_noise"))
        train_diag["lr_g"].append(opt_g.param_groups[0]['lr'])
        train_diag["lr_c"].append(opt_c1.param_groups[0]['lr'])
        train_diag.setdefault("l_nod_mean", []).append(m("l_nod"))

    return m("loss_g"), m("loss_c1_class"), m("loss_c1_noise"), m("loss_c2_class"), m("loss_c2_noise"), m("l_nod")


@torch.no_grad()
def predict_proba(net_c, x_np):
    net_c.eval()
    xs = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    if xs.dim() == 1:
        xs = xs.unsqueeze(0)
    result = torch.softmax(net_c(xs), dim=-1).detach().cpu().numpy()
    net_c.train()
    return result


def get_ensemble_agreement(mm_models, x_np, y_tar_val):
    probs_sum, n_models = np.zeros(x_np.shape[0]), 0
    for name, m in mm_models.items():
        try:
            p = m.predict_proba(x_np)
            probs_sum += p[:, 1]
            n_models += 1
        except:
            pass
    return probs_sum / n_models


def fine_tune_efficient_uncertainty(G, C_ref, mm_models, loader, x_dim, lambda_valid, lambda_cost):
    print(f"\n[Fine-tuning] Random Subsampling Strategy (Proxy Distillation)...")
    QUERY_BATCH_SIZE, BATCHES_PER_EP = 8, 3
    C_proxy = ClassifierNet(x_dim, C_OUT_DIM).to(device)
    C_proxy.load_state_dict(C_ref.state_dict())
    C_proxy.train()
    G.train()
    opt_proxy = torch.optim.AdamW(C_proxy.parameters(), lr=1e-4)
    opt_g = torch.optim.AdamW(G.parameters(), lr=5e-5)
    mse_loss = nn.MSELoss()
    ft_loader = DataLoader(loader.dataset, batch_size=BATCH, shuffle=True)
    total_queries = 0

    for ep in range(FT_EPOCHS):
        batch_scores = []
        for i, (xb, yb) in enumerate(ft_loader):
            if i >= BATCHES_PER_EP:
                break
            xb, yb = xb.to(device), yb.to(device)
            yb_tar = 1 - yb
            with torch.no_grad():
                x_cf_tensor, _ = G(xb, yb_tar)
                x_cf_np = x_cf_tensor.cpu().numpy()

            indices = np.random.choice(x_cf_np.shape[0], QUERY_BATCH_SIZE, replace=False)
            x_query_np = x_cf_np[indices]
            x_query_tensor = x_cf_tensor[indices].detach()
            total_queries += QUERY_BATCH_SIZE

            mm_scores_subset = get_ensemble_agreement(mm_models, x_query_np, 1)
            mm_targets = torch.tensor(mm_scores_subset, device=device, dtype=torch.float32)

            for _ in range(FT_INNER_ITER):
                opt_proxy.zero_grad(set_to_none=True)
                pred_real = C_proxy(xb)
                loss_real = F.cross_entropy(pred_real[:, :2], yb)
                pred_fake_small = C_proxy(x_query_tensor)
                prob_fake_small = torch.softmax(pred_fake_small[:, :2], dim=1)[:, 1]
                loss_distill = mse_loss(prob_fake_small, mm_targets)
                loss_proxy = loss_real + 2.0 * loss_distill
                loss_proxy.backward()
                opt_proxy.step()

            for _ in range(FT_INNER_ITER):
                opt_g.zero_grad(set_to_none=True)
                x_cf_new, cost_new = G(xb, yb_tar)
                pred_proxy = C_proxy(x_cf_new)
                log_probs = F.log_softmax(pred_proxy[:, :2], dim=1)
                loss_valid = F.nll_loss(log_probs, yb_tar)
                loss_g = lambda_valid * loss_valid + lambda_cost * cost_new.mean()
                loss_g.backward()
                opt_g.step()
            batch_scores.append(mm_scores_subset.mean())
        if (ep + 1) % 5 == 0:
            print(f"  FT Ep {ep + 1:02d} | Queries: {total_queries} | Avg MM Agreement: {np.mean(batch_scores):.3f}")
    return C_proxy


@torch.no_grad()
def generate_counterfactual_e2e(net_g, C1, x_src, y_src, y_tar, S=S_STEPS):
    net_g.eval()
    n_samples = INFERENCE_SAMPLES
    x_src_t = torch.as_tensor(x_src, dtype=torch.float32, device=device).view(1, -1)
    x_batch = x_src_t.repeat(n_samples, 1)

    if n_samples > 1:
        noise = torch.randn_like(x_batch[1:]) * INFERENCE_NOISE
        x_batch[1:] += noise
        x_batch = torch.clamp(x_batch, -CLAMP_X, CLAMP_X)

    y_tar_t = torch.tensor([y_tar] * n_samples, dtype=torch.long, device=device)
    y_tar_emb = net_g.emb(y_tar_t)
    context = torch.cat([x_batch, y_tar_emb], dim=1)

    class ODEFunc(nn.Module):
        def __init__(self, v_net, context):
            super().__init__()
            self.v_net, self.context = v_net, context

        def forward(self, t, x_t):
            return self.v_net(t, x_t, self.context)

    ode_func = ODEFunc(net_g.v_net, context)
    t_span = torch.linspace(0.0, 1.0, S).to(device)
    path_batch = odeint_normal(ode_func, x_batch, t_span, method=ODE_METHOD, atol=ODE_ATOL_TEST, rtol=ODE_RTOL_TEST)

    final_points = path_batch[-1]
    dists = torch.norm(final_points - x_batch, dim=1)

    logits = C1(final_points)[:, :2]
    preds = torch.argmax(torch.softmax(logits, dim=-1), dim=1)
    valid_mask = (preds == y_tar).cpu().numpy()
    costs_np = dists.cpu().numpy()

    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        best_idx = valid_indices[np.argmin(costs_np[valid_indices])]
    else:
        best_idx = np.argmin(costs_np)

    best_path_np = path_batch[:, best_idx, :].cpu().numpy()
    net_g.train()
    return best_path_np, {}


def compute_strict_success(path, model):
    """Check if predictions at start and end of path are different."""
    x0, x1 = path[0], path[-1]
    y0_pred = model.predict(x0.reshape(1, -1))[0]
    y1_pred = model.predict(x1.reshape(1, -1))[0]
    return int(y1_pred != y0_pred)


def evaluate_split(C1, C2, G, split="val"):
    """Evaluate counterfactual generation on validation or test split."""
    if split == "val":
        X_eval, y_eval = X_val, y_val
    elif split == "test":
        X_eval, y_eval = X_test, y_test
    else:
        raise ValueError("split must be 'val' or 'test'")

    # Evaluation model list
    eval_models = {
        "kNN": knn,
        "MLP": mlp,
        "CatBoost": catboost,
        "RandomForest": rf,
        "XGBoost": xgboost,
        "SVM": svm,
        "TabNet": tabnet
    }

    test_idx = np.random.choice(len(X_eval), size=min(400, len(X_eval)), replace=False)

    # Initialize counters using dictionary
    strict_success_counts = {name: 0 for name in eval_models.keys()}
    classifier_success_count_c1 = 0
    classifier_success_count_c2 = 0
    total_test = 0

    l1_costs = []
    l2_costs = []
    mm_success_records = {name: [] for name in eval_models.keys()}

    for idx in test_idx:
        xs = X_eval[idx]
        ys = int(y_eval[idx])

        # Check if all models predict correctly on source point
        predictions = {name: model.predict(xs.reshape(1, -1))[0] for name, model in eval_models.items()}
        if not all(pred == ys for pred in predictions.values()):
            continue

        y_tar = 1 - ys
        path, log = generate_counterfactual_e2e(G, C1, xs, ys, y_tar, S=S_STEPS)
        x_tar = path[-1]

        # Evaluate classifiers C1 and C2
        p_after_c1 = predict_proba(C1, x_tar[None, :])[0]
        classifier_success_count_c1 += int(np.argmax(p_after_c1) == y_tar)

        p_after_c2 = predict_proba(C2, x_tar[None, :])[0]
        classifier_success_count_c2 += int(np.argmax(p_after_c2) == y_tar)

        # Compute strict success for all evaluation models
        for name, model in eval_models.items():
            strict_success_counts[name] += compute_strict_success(path, model)

        # Compute costs (sparsity removed)
        x_src = xs
        l1_costs.append(float(np.linalg.norm(x_src - x_tar, ord=1)))
        l2_costs.append(float(np.linalg.norm(x_src - x_tar, ord=2)))

        # Record MM success
        y_src = ys
        y_tar_label = 1 - y_src
        for name, model in eval_models.items():
            try:
                pred_orig = model.predict(x_src.reshape(1, -1))[0]
                pred_cf = model.predict(x_tar.reshape(1, -1))[0]
                success = 1.0 if (pred_orig == y_src and pred_cf == y_tar_label) else 0.0
                mm_success_records[name].append(success)
            except Exception:
                mm_success_records[name].append(0.0)

        total_test += 1

    # Compile metrics
    metrics = {}
    if total_test > 0:
        metrics["classifier_success_rate_c1"] = classifier_success_count_c1 / total_test
        metrics["classifier_success_rate_c2"] = classifier_success_count_c2 / total_test

        # Add strict success rates for all models
        for name in eval_models.keys():
            metrics[f"strict_success_rate_{name.lower()}"] = strict_success_counts[name] / total_test
    else:
        metrics["classifier_success_rate_c1"] = 0.0
        metrics["classifier_success_rate_c2"] = 0.0
        for name in eval_models.keys():
            metrics[f"strict_success_rate_{name.lower()}"] = 0.0

    if len(l1_costs) > 0:
        metrics["avg_l1_cost"] = float(np.mean(l1_costs))
        metrics["std_l1_cost"] = float(np.std(l1_costs))
        metrics["avg_l2_cost"] = float(np.mean(l2_costs))
        metrics["std_l2_cost"] = float(np.std(l2_costs))
    else:
        metrics["avg_l1_cost"] = metrics["std_l1_cost"] = 0.0
        metrics["avg_l2_cost"] = metrics["std_l2_cost"] = 0.0

    mm_success_rates = {}
    for name, rec in mm_success_records.items():
        mm_success_rates[name] = float(np.mean(rec)) if len(rec) > 0 else 0.0

    metrics["mm_success_rates"] = mm_success_rates
    metrics["avg_mm_success_rate"] = float(np.mean(list(mm_success_rates.values()))) if mm_success_rates else 0.0
    metrics["total_points"] = total_test

    return metrics


# -------------------- Grid search (silent) --------------------
best_val_mm = -1.0
best_val_l1 = float('inf')
best_J = -1e9
best_config = None
best_state = None

for lv in LAMBDA_VALIDITY_GRID:
    for lc in LAMBDA_COST_GRID:
        for ln in LAMBDA_NOD_GRID:
            C1, C2, G, ema_c1, ema_c2, ema_g, opt_c1, opt_c2, opt_g, sched_c1, sched_c2, sched_g = build_g_c_models()
            train_diag = copy.deepcopy(train_diag_template)

            best_loss_g = float('inf')
            patience_counter = 0

            for ep in range(1, EPOCHS + 1):
                if ep <= WARMUP_EPOCHS:
                    for g in opt_g.param_groups:
                        g['lr'] = warmup_lr(ep - 1, WARMUP_EPOCHS, LR_G)
                    for g in opt_c1.param_groups:
                        g['lr'] = warmup_lr(ep - 1, WARMUP_EPOCHS, LR_C)
                    for g in opt_c2.param_groups:
                        g['lr'] = warmup_lr(ep - 1, WARMUP_EPOCHS, LR_C)

                loss_g, loss_c1, loss_c1n, loss_c2, loss_c2n, l_nod_mean = train_epoch(
                    G, C1, C2, train_loader, opt_g, opt_c1, opt_c2, ema_g, ema_c1, ema_c2,
                    lv, lc, LAMBDA_NOISE, ln, NOISE_RATIO, ACCUM_STEPS,
                    DIAG, train_diag
                )

                if ep > WARMUP_EPOCHS:
                    sched_g.step()
                    sched_c1.step()
                    sched_c2.step()

                if loss_g < best_loss_g - MIN_DELTA:
                    best_loss_g = loss_g
                    patience_counter = 0
                else:
                    patience_counter += 1

                is_best = ema_g.save_current_shadow_as_best(loss_g)
                if is_best:
                    ema_c1.save_current_shadow_as_best(loss_c1 + loss_c1n)
                    ema_c2.save_current_shadow_as_best(loss_c2 + loss_c2n)

                if ep % 10 == 0 or ep == 1:
                    current_lr_g = opt_g.param_groups[0]['lr']
                    cost_loss = train_diag["loss_g_cost_mean"][-1] if DIAG and len(
                        train_diag["loss_g_cost_mean"]) > 0 else 0.0
                    c1_log = f"C1 Loss (Cls {loss_c1:.4f}/Noise {loss_c1n:.4f})"
                    c2_log = f" | C2 Loss (Cls {loss_c2:.4f}/Noise {loss_c2n:.4f})"
                    best_marker = "🌟 NEW BEST!" if is_best else ""
                    print(
                        f"Epoch {ep:03d} | "
                        f"G Loss {loss_g:.4f} (Cost {cost_loss:.4f}, NOD {l_nod_mean:.4f}) | "
                        f"{c1_log}{c2_log} | "
                        f"LR {current_lr_g:.2e} {best_marker}"
                    )

                if patience_counter >= PATIENCE and ep > 100:
                    print(f"\nEarly stopping at epoch {ep}")
                    break

                if patience_counter >= PATIENCE and ep > 100:
                    break

            ema_g.apply_best_shadow()
            ema_c1.apply_best_shadow()
            ema_c2.apply_best_shadow()

            C_proxy = fine_tune_efficient_uncertainty(G, C1, mm_models, train_loader, x_dim, lv, lc)
            C1 = C_proxy

            val_metrics = evaluate_split(C1, C2, G, split="val")
            val_mm = val_metrics["avg_mm_success_rate"]
            val_l1 = val_metrics["avg_l1_cost"]
            J = val_mm - ALPHA_COST * val_l1

            if J > best_J:
                best_J = J
                best_val_mm = val_mm
                best_val_l1 = val_l1
                best_config = (lv, lc, ln)
                best_state = {
                    "C1": copy.deepcopy(C1.state_dict()),
                    "C2": copy.deepcopy(C2.state_dict()),
                    "G": copy.deepcopy(G.state_dict()),
                    "train_diag": copy.deepcopy(train_diag)
                }

LAMBDA_VALIDITY, LAMBDA_COST, lambda_nod = best_config

C1, C2, G, ema_c1, ema_c2, ema_g, opt_c1, opt_c2, opt_g, sched_c1, sched_c2, sched_g = build_g_c_models()
train_diag = copy.deepcopy(train_diag_template)

print("\n" + "=" * 50)
print(f"Retraining with best config on TRAIN set, then evaluate on TEST...")
print("=" * 50)

best_loss_g = float('inf')
patience_counter = 0

for ep in range(1, EPOCHS + 1):
    if ep <= WARMUP_EPOCHS:
        for g in opt_g.param_groups:
            g['lr'] = warmup_lr(ep - 1, WARMUP_EPOCHS, LR_G)
        for g in opt_c1.param_groups:
            g['lr'] = warmup_lr(ep - 1, WARMUP_EPOCHS, LR_C)
        for g in opt_c2.param_groups:
            g['lr'] = warmup_lr(ep - 1, WARMUP_EPOCHS, LR_C)

    loss_g, loss_c1, loss_c1n, loss_c2, loss_c2n, l_nod_mean = train_epoch(
        G, C1, C2, train_loader, opt_g, opt_c1, opt_c2, ema_g, ema_c1, ema_c2,
        LAMBDA_VALIDITY, LAMBDA_COST, LAMBDA_NOISE, lambda_nod, NOISE_RATIO, ACCUM_STEPS,
        DIAG, train_diag
    )

    if ep > WARMUP_EPOCHS:
        sched_g.step()
        sched_c1.step()
        sched_c2.step()

    if loss_g < best_loss_g - MIN_DELTA:
        best_loss_g = loss_g
        patience_counter = 0
    else:
        patience_counter += 1

    is_best = ema_g.save_current_shadow_as_best(loss_g)
    if is_best:
        ema_c1.save_current_shadow_as_best(loss_c1 + loss_c1n)
        ema_c2.save_current_shadow_as_best(loss_c2 + loss_c2n)

    if ep % 10 == 0 or ep == 1:
        current_lr_g = opt_g.param_groups[0]['lr']
        cost_loss = train_diag["loss_g_cost_mean"][-1] if DIAG and len(train_diag["loss_g_cost_mean"]) > 0 else 0.0
        c1_log = f"C1 Loss (Cls {loss_c1:.4f}/Noise {loss_c1n:.4f})"
        c2_log = f" | C2 Loss (Cls {loss_c2:.4f}/Noise {loss_c2n:.4f})"
        best_marker = "🌟 NEW BEST!" if is_best else ""
        print(
            f"Epoch {ep:03d} | "
            f"G Loss {loss_g:.4f} (Cost {cost_loss:.4f}, NOD {l_nod_mean:.4f}) | "
            f"{c1_log}{c2_log} | "
            f"LR {current_lr_g:.2e} {best_marker}"
        )

    if patience_counter >= PATIENCE and ep > 100:
        print(f"\nEarly stopping at epoch {ep}")
        break

print("\nTraining completed with best config!")
print("=" * 50)
print("Applying BEST EMA weights...")
ema_g.apply_best_shadow()
ema_c1.apply_best_shadow()
ema_c2.apply_best_shadow()

print("\n[Phase 2] Starting Proxy Fine-tuning with Ensemble Feedback (best config)...")
C_proxy = fine_tune_efficient_uncertainty(G, C1, mm_models, train_loader, x_dim, LAMBDA_VALIDITY, LAMBDA_COST)
C1 = C_proxy
print("[Phase 2] Fine-tuning completed.")

# -------------------- Detailed Evaluation on TEST --------------------
print("\n" + "=" * 50)
print("Computing detailed metrics (Full Evaluation) on TEST...")
print("=" * 50)
with torch.no_grad():
    C1.eval()
    xs_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
    pred_c1 = torch.argmax(C1(xs_test_t)[:, :2], dim=-1)
    classifier_model_acc_c1 = (pred_c1.cpu().numpy() == y_test).mean()
    print(f"Classifier (C1) model test accuracy (on data): {classifier_model_acc_c1:.3f}")
    C1.train()

    C2.eval()
    pred_c2 = torch.argmax(C2(xs_test_t)[:, :2], dim=-1)
    classifier_model_acc_c2 = (pred_c2.cpu().numpy() == y_test).mean()
    print(f"Classifier (C2) model test accuracy (on data): {classifier_model_acc_c2:.3f}")
    C2.train()

print(f"(Baseline) kNN classifier test accuracy: {knn_test_acc:.3f}")
print(f"(Baseline) MLP classifier test accuracy: {mlp_test_acc:.3f}")
print(f"(Baseline) CatBoost classifier test accuracy: {catboost_test_acc:.3f}")
print(f"(Baseline) Random Forest test accuracy: {rf_test_acc:.3f}")
print(f"(Baseline) XGBoost test accuracy: {xgboost_test_acc:.3f}")
print(f"(Baseline) SVM test accuracy: {svm_test_acc:.3f}")
print(f"(Baseline) TabNet test accuracy: {tabnet_test_acc:.3f}")

test_metrics = evaluate_split(C1, C2, G, split="test")

classifier_success_rate_c1 = test_metrics["classifier_success_rate_c1"]
classifier_success_rate_c2 = test_metrics["classifier_success_rate_c2"]
strict_success_rate_knn = test_metrics["strict_success_rate_knn"]
strict_success_rate_mlp = test_metrics["strict_success_rate_mlp"]
strict_success_rate_catboost = test_metrics["strict_success_rate_catboost"]
strict_success_rate_rf = test_metrics["strict_success_rate_randomforest"]
strict_success_rate_xgboost = test_metrics["strict_success_rate_xgboost"]
strict_success_rate_svm = test_metrics["strict_success_rate_svm"]
strict_success_rate_tabnet = test_metrics["strict_success_rate_tabnet"]

avg_l2_cost = test_metrics["avg_l2_cost"]
std_l2_cost = test_metrics["std_l2_cost"]
mm_success_rates = test_metrics["mm_success_rates"]
avg_mm_success_rate = test_metrics["avg_mm_success_rate"]

print(f"\n--- CF Success Rates (on {test_metrics['total_points']} TEST points) ---")
print(f"(E2E) Classifier (C1) model CF success: {classifier_success_rate_c1:.3f}")
print(f"(E2E) Classifier (C2) model CF success: {classifier_success_rate_c2:.3f}")
print(f"(Strict) kNN CF success              : {strict_success_rate_knn:.3f}")
print(f"(Strict) MLP CF success              : {strict_success_rate_mlp:.3f}")
print(f"(Strict) CatBoost CF success         : {strict_success_rate_catboost:.3f}")
print(f"(Strict) Random Forest CF success    : {strict_success_rate_rf:.3f}")
print(f"(Strict) XGBoost CF success          : {strict_success_rate_xgboost:.3f}")
print(f"(Strict) SVM CF success              : {strict_success_rate_svm:.3f}")
print(f"(Strict) TabNet CF success           : {strict_success_rate_tabnet:.3f}")
print("=" * 50)

# -------------------- Adversarial Baseline --------------------
print("\n" + "=" * 50)
print("Computing adversarial baseline...")
print("=" * 50)
adv_success_knn = 0
adv_success_mlp = 0
adv_success_catboost = 0
adv_total = 0

for _ in range(500):
    idx = np.random.randint(0, len(X_test))
    x_src = X_test[idx]
    y_src = y_test[idx]
    if (knn.predict(x_src.reshape(1, -1))[0] != y_src or
            mlp.predict(x_src.reshape(1, -1))[0] != y_src or
            catboost.predict(x_src.reshape(1, -1))[0] != y_src):
        continue
    y_tar = 1 - y_src
    x_adv = x_src + np.random.normal(0, 0.8, size=x_src.shape)
    if knn.predict(x_adv.reshape(1, -1))[0] == y_tar:
        adv_success_knn += 1
    if mlp.predict(x_adv.reshape(1, -1))[0] == y_tar:
        adv_success_mlp += 1
    if catboost.predict(x_adv.reshape(1, -1))[0] == y_tar:
        adv_success_catboost += 1
    adv_total += 1
    if adv_total >= 50:
        break

if adv_total > 0:
    adv_rate_knn = adv_success_knn / adv_total
    adv_rate_mlp = adv_success_mlp / adv_total
    adv_rate_catboost = adv_success_catboost / adv_total
    print(f"Adversarial baseline (on {adv_total} points):")
    print(f"  kNN success rate: {adv_rate_knn:.2f}")
    print(f"  MLP success rate: {adv_rate_mlp:.2f}")
    print(f"  CatBoost success rate: {adv_rate_catboost:.2f}")
else:
    print("⚠ Could not find 50 valid starting points for adversarial baseline.")
print("=" * 50)

# -------------------- Final Summary --------------------
print("\n✓ All done!")
print("=" * 60)
print(f"FINAL SUMMARY (E2E DensityFlow)")
print("=" * 60)

print("\n--- [ 1. Baseline Classifier Accuracy ] ---")
print(f" • (E2E) Classifier (C1) model test accuracy: {classifier_model_acc_c1:.3f}")
print(f" • (E2E) Classifier (C2) model test accuracy: {classifier_model_acc_c2:.3f}")
print(f" • (Strict) kNN test accuracy              : {knn_test_acc:.3f}")
print(f" • (Strict) MLP test accuracy              : {mlp_test_acc:.3f}")
print(f" • (Strict) CatBoost test accuracy         : {catboost_test_acc:.3f}")
print(f" • (Strict) XGBoost test accuracy          : {xgboost_test_acc:.3f}")
print(f" • (Strict) RandomForest test accuracy     : {rf_test_acc:.3f}")
print(f" • (Strict) SVM test accuracy              : {svm_test_acc:.3f}")
print(f" • (Strict) TabNet test accuracy           : {tabnet_test_acc:.3f}")

print("\n--- [ 2. Counterfactual Success Rate (TEST) ] ---")
print(f" • (E2E) Classifier (C1) model CF success: {classifier_success_rate_c1:.3f}")
print(f" • (E2E) Classifier (C2) model CF success: {classifier_success_rate_c2:.3f}")
print(f" • (Strict) kNN CF success              : {strict_success_rate_knn:.3f}")
print(f" • (Strict) MLP CF success              : {strict_success_rate_mlp:.3f}")
print(f" • (Strict) CatBoost CF success         : {strict_success_rate_catboost:.3f}")
print(f" • (Strict) XGBoost CF success          : {strict_success_rate_xgboost:.3f}")
print(f" • (Strict) RandomForest CF success     : {strict_success_rate_rf:.3f}")
print(f" • (Strict) SVM CF success              : {strict_success_rate_svm:.3f}")
print(f" • (Strict) TabNet CF success           : {strict_success_rate_tabnet:.3f}")

print("\n--- [ 3. Adversarial Baseline (Noise, TEST) ] ---")
if adv_total > 0:
    print(f" • (Strict) kNN adv. baseline     : {adv_rate_knn:.3f}")
    print(f" • (Strict) MLP adv. baseline     : {adv_rate_mlp:.3f}")
    print(f" • (Strict) CatBoost adv. baseline: {adv_rate_catboost:.3f}")
else:
    print(" • Adversarial baseline: N/A")

print("\n--- [ 4. CF Metrics (All Black-box Models, TEST) ] ---")
print(f" • Avg L2 cost                    : {avg_l2_cost:.4f} ± {std_l2_cost:.4f}")
print(f" • Model-wise MM Success Rate:")
for name in ["kNN", "MLP", "CatBoost", "RandomForest", "XGBoost", "TabNet", "SVM"]:
    print(f"    - {name}: {mm_success_rates[name]:.4f}")
print(f" • Avg MM Success Rate (all models, TEST): {avg_mm_success_rate:.4f}")
print("=" * 60)