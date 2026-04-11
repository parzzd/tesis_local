# app/pipeline.py  –  Funciones compartidas del pipeline de pose-estimation
#
# Centraliza: pool_frame_to_51, frame_visible, pool_scores, featurize_T51,
#             norm_apply y predict_window.  Usado por server.py, inferencia_video.py
#             y train_lgbm.py para evitar divergencias.

import json
from pathlib import Path
from typing import List

import numpy as np
import joblib
from keras.models import load_model
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════
# Constantes por defecto
# ══════════════════════════════════════════════════════════
SEQ_LEN      = 32
CONF_MIN     = 0.10
MIN_VIS_FRAC = 0.30
HYST_GAP     = 0.10

# --- Indices COCO de joints ---
#  0=nose  1=L_eye  2=R_eye  3=L_ear  4=R_ear
#  5=L_shoulder  6=R_shoulder  7=L_elbow  8=R_elbow
#  9=L_wrist  10=R_wrist  11=L_hip  12=R_hip
# 13=L_knee  14=R_knee  15=L_ankle  16=R_ankle

# Pares de distancias
PAIR_DISTS = [
    (5, 6),    # hombro-hombro
    (9, 10),   # muñeca-muñeca
    (0, 9),    # nariz-muñeca_izq  (golpe)
    (0, 10),   # nariz-muñeca_der  (golpe)
    (11, 12),  # cadera-cadera
    (15, 16),  # tobillo-tobillo
    (9, 12),   # muñeca_izq-cadera_der  (cruce)
    (10, 11),  # muñeca_der-cadera_izq  (cruce)
    (5, 11),   # hombro_izq-cadera_izq  (torso)
    (6, 12),   # hombro_der-cadera_der  (torso)
]

# Triplets para angulos articulares: (A, vertice, B)
ANGLE_JOINTS = [
    (5, 7, 9),    # hombro_izq -> codo_izq -> muñeca_izq
    (6, 8, 10),   # hombro_der -> codo_der -> muñeca_der
    (11, 13, 15),  # cadera_izq -> rodilla_izq -> tobillo_izq
    (12, 14, 16),  # cadera_der -> rodilla_der -> tobillo_der
    (7, 5, 11),   # codo_izq -> hombro_izq -> cadera_izq
    (8, 6, 12),   # codo_der -> hombro_der -> cadera_der
    (5, 0, 6),    # hombro_izq -> nariz -> hombro_der  (inclinacion cabeza)
]


# ══════════════════════════════════════════════════════════
# Utilidades de frame
# ══════════════════════════════════════════════════════════
def pool_frame_to_51(kps_f: np.ndarray | None, W: int, H: int) -> np.ndarray:
    """(P,17,3) -> (51,)  pooling por joint con mayor confianza, normalizado por W,H."""
    out = np.zeros((17, 3), dtype=np.float32)
    if kps_f is None or kps_f.size == 0:
        return out.reshape(-1)
    conf_j = np.nan_to_num(kps_f[..., 2], nan=0.0)
    for j in range(17):
        idx = int(np.argmax(conf_j[:, j]))
        c = conf_j[idx, j]
        if c > 0:
            x, y, _ = kps_f[idx, j, :]
            if np.isfinite(x) and np.isfinite(y):
                out[j, 0] = np.clip(x / max(W, 1), 0.0, 1.0)
                out[j, 1] = np.clip(y / max(H, 1), 0.0, 1.0)
                out[j, 2] = float(np.clip(c, 0.0, 1.0))
    return out.reshape(-1)


def frame_visible(kps_f: np.ndarray | None, conf_min: float = CONF_MIN) -> bool:
    if kps_f is None or kps_f.size == 0:
        return False
    conf = np.nan_to_num(kps_f[..., 2], nan=0.0)
    return bool((conf >= conf_min).any())


# ══════════════════════════════════════════════════════════
# Pooling de scores a nivel video
# ══════════════════════════════════════════════════════════
def pool_scores(scores: List[float], pool: str = "topk", topk_frac: float = 0.2) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=np.float32)
    if pool == "max":
        return float(arr.max())
    if pool == "mean":
        return float(arr.mean())
    k = max(1, int(len(arr) * topk_frac))
    return float(np.partition(arr, -k)[-k:].mean())


# ══════════════════════════════════════════════════════════
# Featurizador tabular para LGBM  (DEBE coincidir con train_lgbm.py)
# ══════════════════════════════════════════════════════════
def _stats(a: np.ndarray) -> np.ndarray:
    """mean, std, min, max por columna."""
    return np.concatenate([a.mean(0), a.std(0), a.min(0), a.max(0)], axis=0)


def _angle_between(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angulo en radianes en el vertice, para cada frame.  (T,)"""
    va = a - vertex  # (T, 2)
    vb = b - vertex
    cos = np.sum(va * vb, axis=1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-8)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def featurize_T51(X_win: np.ndarray) -> np.ndarray:
    """
    Desde una ventana (T, 51) genera el vector tabular para LGBM.

    Bloques de features:
      1. Stats x, y, velocidad, confianza  (4 * 4stats * 17joints = 272)
      2. Aceleracion stats                  (4stats * 17joints = 68)
      3. Distancias entre pares             (10 pares * 2 = 20)
      4. Angulos articulares stats          (7 angulos * 4stats = 28)

    Total: 272 + 68 + 20 + 28 = 388 features

    Returns: (1, D) array listo para predict.
    """
    T = X_win.shape[0]
    xyz = X_win.reshape(T, 17, 3)
    x, y, c = xyz[..., 0], xyz[..., 1], xyz[..., 2]  # (T, 17) cada uno

    # --- Velocidades (1ra derivada) ---
    dx = np.diff(x, axis=0, prepend=x[0:1])
    dy = np.diff(y, axis=0, prepend=y[0:1])
    v = np.sqrt(dx * dx + dy * dy)

    # --- Aceleracion (2da derivada, magnitud) ---
    ddx = np.diff(dx, axis=0, prepend=dx[0:1])
    ddy = np.diff(dy, axis=0, prepend=dy[0:1])
    acc = np.sqrt(ddx * ddx + ddy * ddy)

    # --- Stats basicas ---
    Fx = _stats(x)
    Fy = _stats(y)
    Fv = _stats(v)
    Fc = _stats(c)
    Fa = _stats(acc)

    # --- Distancias entre pares ---
    D = []
    for (i, j) in PAIR_DISTS:
        dij = np.sqrt((x[:, i] - x[:, j]) ** 2 + (y[:, i] - y[:, j]) ** 2)
        D += [dij.mean(), dij.std()]
    D = np.array(D, dtype=np.float32)

    # --- Angulos articulares ---
    angles = []
    for (a_idx, v_idx, b_idx) in ANGLE_JOINTS:
        pt_a = np.stack([x[:, a_idx], y[:, a_idx]], axis=1)    # (T, 2)
        pt_v = np.stack([x[:, v_idx], y[:, v_idx]], axis=1)
        pt_b = np.stack([x[:, b_idx], y[:, b_idx]], axis=1)
        ang = _angle_between(pt_a, pt_v, pt_b)                 # (T,)
        angles.append(ang)
    angles = np.stack(angles, axis=1)  # (T, 7)
    Fang = _stats(angles)

    feat = np.concatenate([Fx, Fy, Fv, Fc, Fa, D, Fang], axis=0).astype(np.float32)
    return feat.reshape(1, -1)


# ══════════════════════════════════════════════════════════
# Normalización para el modelo Keras (LSTM)
# ══════════════════════════════════════════════════════════
def norm_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """X: (1, T, 51) -> normalizado."""
    T, F = X.shape[1], X.shape[2]
    X2 = X.reshape(-1, F)
    Xn = (X2 - mu) / (sd + 1e-6)
    return Xn.reshape(1, T, F).astype("float32")


# ══════════════════════════════════════════════════════════
# Predicción fusionada  LSTM + LGBM (soporta stacking)
# ══════════════════════════════════════════════════════════
def predict_window(Xw: np.ndarray, keras_model, mu, sd,
                   lgbm=None, fusion_w: float = 0.5,
                   stacker=None) -> float:
    """
    Xw: (T, 51)
    fusion_w: peso para fusion lineal (ignorado si stacker != None)
    stacker:  modelo meta-learner (LogisticRegression) entrenado sobre [p_lstm, p_lgbm]
    """
    # --- Keras (LSTM) ---
    X = Xw[np.newaxis, ...]
    X = norm_apply(X, mu, sd)
    p_keras = float(keras_model.predict(X, verbose=0).ravel()[0])

    if lgbm is None or fusion_w <= 0.0:
        return np.clip(p_keras, 0.0, 1.0)

    # --- LGBM ---
    feat = featurize_T51(Xw)
    expected = getattr(lgbm, "n_features_in_", None)
    if expected is not None and feat.shape[1] != expected:
        return np.clip(p_keras, 0.0, 1.0)

    try:
        p_lgbm = float(lgbm.predict_proba(feat)[:, 1][0])
    except Exception:
        return np.clip(p_keras, 0.0, 1.0)

    # --- Fusion ---
    if stacker is not None:
        try:
            meta_feat = np.array([[p_keras, p_lgbm]], dtype=np.float32)
            p_fused = float(stacker.predict_proba(meta_feat)[:, 1][0])
            return np.clip(p_fused, 0.0, 1.0)
        except Exception:
            pass

    # Fallback: fusion lineal
    return float(np.clip((1.0 - fusion_w) * p_keras + fusion_w * p_lgbm, 0.0, 1.0))


# ══════════════════════════════════════════════════════════
# Carga de artefactos (modelos + stats + threshold + stacker)
# ══════════════════════════════════════════════════════════
def load_artifacts(models_dir: str | Path, pose_weights: str):
    """
    Retorna: (keras_model, mu, sd, thr_on, thr_off, lgbm, pose, stacker)
    """
    models_dir = Path(models_dir)

    keras_path  = models_dir / "mix_cnn_lstm_T32_F51.keras"
    stats_path  = models_dir / "mix_cnn_lstm_T32_F51_norm_stats.npz"
    thr_path    = models_dir / "mix_cnn_lstm_T32_F51_threshold.json"
    lgbm_path   = models_dir / "lgbm_model.pkl"
    stacker_path = models_dir / "stacker.pkl"

    keras_model = load_model(str(keras_path), compile=False)

    stats = np.load(stats_path)
    mu = stats["mean"].astype("float32")
    sd = stats["std"].astype("float32")

    thr_on = 0.5
    if thr_path.exists():
        thr_on = float(json.loads(thr_path.read_text(encoding="utf-8")).get("best_threshold", 0.5))
    thr_off = max(0.0, thr_on - HYST_GAP)

    lgbm = None
    if lgbm_path.exists():
        try:
            lgbm = joblib.load(lgbm_path)
            print(f"[BOOT] LGBM cargado desde {lgbm_path}")
        except Exception as e:
            print(f"[BOOT] LGBM falló al cargar ({e}), continuo sin LGBM")

    stacker = None
    if stacker_path.exists():
        try:
            stacker = joblib.load(stacker_path)
            print(f"[BOOT] Stacker cargado desde {stacker_path}")
        except Exception as e:
            print(f"[BOOT] Stacker falló ({e}), usando fusion lineal")

    pose = YOLO(pose_weights)
    print(f"[BOOT] Keras={keras_path.name} | THR_ON={thr_on:.2f} THR_OFF={thr_off:.2f}"
          f" | LGBM={'ON' if lgbm else 'OFF'} | Stacker={'ON' if stacker else 'OFF'}")

    return keras_model, mu, sd, thr_on, thr_off, lgbm, pose, stacker
