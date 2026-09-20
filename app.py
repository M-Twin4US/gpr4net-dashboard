"""
GPR4Net Digital Twin Dashboard
Scenarios: Tunnel Lining (800 / 600 MHz) | Pipeline (1 GHz)
"""

import sys
import time
import base64
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.interpolate import CubicSpline, splprep, splev
from matplotlib.path import Path as MplPath
from scipy.ndimage import zoom

# ── Model source ───────────────────────────────────────────────────────────────
# All paths are relative to the repo root so the app is portable to any
# hosting platform (Streamlit Community Cloud, Hugging Face Spaces, …).
ROOT      = Path(__file__).parent.resolve()
CKPT_800  = ROOT / 'checkpoints' / 'tunnel_d1b1.pth'
CKPT_600  = ROOT / 'checkpoints' / 'tunnel_d1a_600mhz.pth'
CKPT_PIPE = ROOT / 'checkpoints' / 'pipe_d2b1.pth'
sys.path.insert(0, str(ROOT))
from models.d1b1_model import D1B1Model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GPR4Net Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── High-resolution download config for all Plotly figures ────────────────────
PLOTLY_CONFIG = {
    'toImageButtonOptions': {
        'format':   'png',
        'filename': 'GPR4Net_figure',
        'scale':       3,    # 3× current display resolution (preserves aspect ratio)
    },
    'displaylogo': False,
}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .dt-header {
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        padding: 1.2rem 1.8rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
    }
    .dt-header h1 { color: #e0f7fa; font-size: 1.6rem; margin: 0; }
    .dt-header p  { color: #80cbc4; font-size: 0.85rem; margin: 0.2rem 0 0 0; }
    div[data-testid="stSidebar"] { background-color: #f8fafc; }
    /* Compact sidebar */
    div[data-testid="stSidebar"] .stSlider       { margin-bottom: -0.6rem; }
    div[data-testid="stSidebar"] .stRadio        { margin-bottom: -0.4rem; }
    div[data-testid="stSidebar"] .stSelectbox    { margin-bottom: -0.4rem; }
    div[data-testid="stSidebar"] h5              { margin: 0.3rem 0 0.1rem 0; font-size: 0.78rem; color: #2c5364; text-transform: uppercase; letter-spacing: 0.04em; }
    div[data-testid="stSidebar"] hr              { margin: 0.4rem 0; }
    div[data-testid="stSidebar"] label           { font-size: 0.80rem !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Tunnel ─────────────────────────────────────────────────────────────────────
T_DX      = 0.004
T_NX, T_NY, T_FREE = 500, 350, 325
T_IFACE_X = np.array([0.0, 1.0, 2.0])
T_IDX_AIR, T_IDX_GROT, T_IDX_ROCK, T_IDX_VOID = 0, 1, 2, 3

# ── Pipeline ───────────────────────────────────────────────────────────────────
P_DX      = 0.004
P_NX, P_NY, P_FREE = 375, 350, 325

PIPE_SPECS = {
    'Metallic DN250': dict(od=0.274, wall=0.0040, material='metallic'),
    'Metallic DN350': dict(od=0.378, wall=0.0085, material='metallic'),
    'Plastic DN50':   dict(od=0.050, wall=0.0080, material='plastic'),
    'Plastic DN250':  dict(od=0.250, wall=0.0420, material='plastic'),
}

# ── Preprocessing constants ────────────────────────────────────────────────────
T_TARGET_H, T_TARGET_W = 256, 256   # tunnel training resolution
P_TARGET_H, P_TARGET_W = 184, 256   # pipeline training resolution
SIG_MAX  = 0.1
T_EPS_MAX_MODEL = 10.0   # tunnel model: EPS_MAX=10 (water excluded)
P_EPS_MAX_MODEL = 80.0   # pipeline model: EPS_MAX=80

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    def _load(path):
        m  = D1B1Model(features=[14, 28, 56, 112, 224], num_heads=4)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        m.load_state_dict(ck['model_state_dict'])
        m.eval()
        return m
    return _load(CKPT_800), _load(CKPT_600), _load(CKPT_PIPE)

# ══════════════════════════════════════════════════════════════════════════════
# TUNNEL GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
def build_tunnel_geometry(void_x, void_y, void_R, void_delta, seed):
    data = np.zeros((T_NX, T_NY), dtype=np.int32)
    rng    = np.random.default_rng(seed)
    ctrl_y = rng.uniform(0.55, 0.75, size=3)
    spline = CubicSpline(T_IFACE_X, ctrl_y)
    xs     = np.arange(T_NX) * T_DX
    iface  = np.clip(np.round(spline(xs) / T_DX).astype(int), 1, T_FREE - 1)
    for xi in range(T_NX):
        data[xi, :iface[xi]]         = T_IDX_ROCK
        data[xi, iface[xi]:T_FREE]   = T_IDX_GROT
        data[xi, T_FREE:]            = T_IDX_AIR
    rng2   = np.random.default_rng(seed + 1000)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    radii  = void_R * (1 + rng2.uniform(-void_delta, void_delta, 8))
    px = np.append(void_x + radii * np.cos(angles), void_x + radii[0] * np.cos(angles[0]))
    py = np.append(void_y + radii * np.sin(angles), void_y + radii[0] * np.sin(angles[0]))
    tck, _ = splprep([px, py], s=0, per=True, k=3)
    xf, yf = splev(np.linspace(0, 1, 2000), tck)
    xi_c = (np.arange(T_NX) + 0.5) * T_DX
    yi_c = (np.arange(T_NY) + 0.5) * T_DX
    Xc, Yc = np.meshgrid(xi_c, yi_c, indexing='ij')
    pts    = np.column_stack([Xc.ravel(), Yc.ravel()])
    inside = MplPath(np.column_stack([xf, yf])).contains_points(pts).reshape(T_NX, T_NY)
    data[inside & (data == T_IDX_GROT)] = T_IDX_VOID
    return data

def tunnel_to_model_input(data, rock_er, rock_sig, grot_er, grot_sig, void_state):
    void_er  = 80.0 if void_state == 'Saturated' else 1.0
    void_sig = 5e-4 if void_state == 'Saturated' else 0.0
    er_lut  = {T_IDX_AIR: 1.0, T_IDX_GROT: grot_er,  T_IDX_ROCK: rock_er,  T_IDX_VOID: void_er}
    sig_lut = {T_IDX_AIR: 0.0, T_IDX_GROT: grot_sig, T_IDX_ROCK: rock_sig, T_IDX_VOID: void_sig}
    er_map  = np.vectorize(er_lut.get)(data).astype(np.float32)
    sig_map = np.vectorize(sig_lut.get)(data).astype(np.float32)
    sat_msk = ((data == T_IDX_VOID) & (void_state == 'Saturated')).astype(np.float32)
    er_map  = np.flipud(er_map[:, :T_FREE].T)
    sig_map = np.flipud(sig_map[:, :T_FREE].T)
    sat_msk = np.flipud(sat_msk[:, :T_FREE].T)
    def rsz(a, o=1): return zoom(a, (T_TARGET_H/a.shape[0], T_TARGET_W/a.shape[1]), order=o).astype(np.float32)
    ch0 = np.clip((er_map - 1.0) / (T_EPS_MAX_MODEL - 1.0), 0, 1)
    ch1 = sat_msk
    ch2 = np.clip(sig_map, 0, SIG_MAX) / SIG_MAX
    geom = np.stack([rsz(ch0), rsz(ch1, 0), rsz(ch2)], axis=0)
    return geom, er_map

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
def build_pipe_geometry(pipe_name, x_c, y_c, soil_er, soil_sig, pipe_er, pipe_sig):
    er_map  = np.ones((P_NX, P_NY), dtype=np.float32)
    sig_map = np.zeros((P_NX, P_NY), dtype=np.float32)
    pec_map = np.zeros((P_NX, P_NY), dtype=np.float32)
    er_map[:,  :P_FREE] = soil_er
    sig_map[:, :P_FREE] = soil_sig
    spec    = PIPE_SPECS[pipe_name]
    r_outer = spec['od'] / 2.0
    r_inner = r_outer - spec['wall']
    xi_c = (np.arange(P_NX) + 0.5) * P_DX
    yi_c = (np.arange(P_NY) + 0.5) * P_DX
    Xc, Yc = np.meshgrid(xi_c, yi_c, indexing='ij')
    dist       = np.sqrt((Xc - x_c)**2 + (Yc - y_c)**2)
    wall_mask  = (dist <= r_outer) & (dist > r_inner)
    inner_mask = dist <= r_inner
    if spec['material'] == 'metallic':
        pec_map[wall_mask] = 1.0
        er_map[wall_mask]  = 1.0
        sig_map[wall_mask] = 0.0
    else:
        er_map[wall_mask]  = pipe_er
        sig_map[wall_mask] = pipe_sig
    er_map[inner_mask]  = 1.0
    sig_map[inner_mask] = 0.0
    return er_map, sig_map, pec_map

def pipe_to_model_input(er_map, sig_map, pec_map):
    # Crop free space first (y=0..P_FREE-1 = soil only), then orient
    # This matches preprocess_geometry.py: data[:, :FREE_TOP] then flipud(arr.T)
    er_m  = np.flipud(er_map[:, :P_FREE].T)
    sig_m = np.flipud(sig_map[:, :P_FREE].T)
    pec_m = np.flipud(pec_map[:, :P_FREE].T)
    def rsz(a, o=1): return zoom(a, (P_TARGET_H/a.shape[0], P_TARGET_W/a.shape[1]), order=o).astype(np.float32)
    ch0 = (er_m - 1.0) / (P_EPS_MAX_MODEL - 1.0)
    ch1 = np.clip(sig_m, 0, SIG_MAX) / SIG_MAX
    ch2 = pec_m
    geom = np.stack([rsz(ch0), rsz(ch2, 0), rsz(ch1)], axis=0)   # [εr, PEC, σ]
    return geom, er_m, pec_m

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def predict(model, geom_np):
    x = torch.from_numpy(geom_np).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(x)[0, 0].numpy()
    return pred, (time.perf_counter() - t0) * 1000

# ── Measure actual inference time on this machine ─────────────────────────────
@st.cache_resource
def measure_inference_ms(_model):
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        _model(x)                          # warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        with torch.no_grad():
            _model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))

# ══════════════════════════════════════════════════════════════════════════════
# OAT SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
import torch.nn.functional as F

# Parameter definitions: name → (min, max, default, display_format)
SURFACE_Y = 1.3   # antenna level in gprMax coordinates (m from domain bottom)

OAT_TUNNEL = {
    'Void x (m)':                (0.20,  1.80,  1.00,  '{:.2f}'),
    'Depth below surface (m)':   (0.20,  0.40,  0.30,  '{:.3f}'),  # = 1.3 - void_y
    'Void radius (m)':           (0.024, 0.080, 0.050, '{:.3f}'),
    'Void irregularity':         (0.10,  0.40,  0.25,  '{:.2f}'),
    'Rock εr':                   (3.0,   8.0,   5.5,   '{:.1f}'),
    'Rock σ (S/m)':              (0.0,   1e-3,  5e-4,  '{:.1e}'),
    'Grouting εr':               (5.0,   10.0,  7.5,   '{:.1f}'),
    'Grouting σ (S/m)':          (1e-3,  1e-2,  5e-3,  '{:.1e}'),
}

OAT_PIPE = {
    'Pipe x (m)':                (0.50,  1.00,  0.75,  '{:.2f}'),
    'Pipe depth below surface (m)': (0.325, 0.575, 0.45, '{:.3f}'),  # = 1.3 - pipe_y
    'Soil εr':                   (3.0,   10.0,  6.5,   '{:.1f}'),
    'Soil σ (S/m)':              (1e-7,  1e-3,  1e-5,  '{:.1e}'),
}

# ── Metric helpers ─────────────────────────────────────────────────────────────
def _ssim(pred, ref):
    p = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
    g = torch.from_numpy(ref).unsqueeze(0).unsqueeze(0)
    C1, C2 = 0.01**2, 0.03**2
    w = 11; pad = w // 2
    mp = F.avg_pool2d(p, w, 1, pad); mg = F.avg_pool2d(g, w, 1, pad)
    sp = F.avg_pool2d(p**2, w, 1, pad) - mp**2
    sg = F.avg_pool2d(g**2, w, 1, pad) - mg**2
    spg = F.avg_pool2d(p*g, w, 1, pad) - mp*mg
    return float(((2*mp*mg+C1)*(2*spg+C2)/((mp**2+mg**2+C1)*(sp+sg+C2))).mean())

def compute_metrics(pred, ref):
    mse  = float(np.mean((pred - ref) ** 2))
    l1   = float(np.mean(np.abs(pred - ref)))
    ssim = _ssim(pred, ref)
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-10)))
    return {'MSE': mse, 'L1': l1, 'SSIM dev': 1 - ssim, 'PSNR (dB)': psnr}

# ── OAT runner ─────────────────────────────────────────────────────────────────
# Parameters whose input display uses σ map instead of εr map
SIGMA_PARAMS = {'Rock σ (S/m)', 'Grouting σ (S/m)', 'Soil σ (S/m)'}

def run_oat(scenario, model, baseline, n_steps=9, oat_seed=42, progress_cb=None):
    """Returns results, sweeps, bscan_base, er_base, sig_base, bscans, geom_disps, elapsed"""
    t_start = time.perf_counter()

    if 'Tunnel' in scenario:
        void_y_base = SURFACE_Y - baseline.get('depth', baseline.get('void_y', 1.0))
        d = build_tunnel_geometry(baseline['void_x'], void_y_base,
                                  baseline['void_R'], baseline['void_delta'], oat_seed)
        g, er_base = tunnel_to_model_input(d, baseline['rock_er'], baseline['rock_sig'],
                                            baseline['grot_er'], baseline['grot_sig'], 'Dry')
        sl = {T_IDX_AIR: 0.0, T_IDX_GROT: baseline['grot_sig'],
              T_IDX_ROCK: baseline['rock_sig'], T_IDX_VOID: 0.0}
        sig_base = np.flipud(np.vectorize(sl.get)(d).astype(np.float32)[:, :T_FREE].T)
        param_defs = OAT_TUNNEL
        key_map = {
            'Void x (m)':              'void_x',
            'Depth below surface (m)': 'depth',      # converted to void_y below
            'Void radius (m)':         'void_R',
            'Void irregularity':       'void_delta',
            'Rock εr':                 'rock_er',
            'Rock σ (S/m)':            'rock_sig',
            'Grouting εr':             'grot_er',
            'Grouting σ (S/m)':        'grot_sig',
        }
    else:
        pipe_y_base = SURFACE_Y - baseline.get('pipe_depth',
                                                baseline.get('pipe_y', 0.85))
        em, sm, pm = build_pipe_geometry(
            baseline['pipe_name'], baseline['pipe_x'], pipe_y_base,
            baseline['soil_er'], baseline['soil_sig'], 1.0, 0.0)
        g, er_base, _ = pipe_to_model_input(em, sm, pm)
        sig_base = np.flipud(sm[:, :P_FREE].T)
        param_defs = OAT_PIPE
        key_map = {
            'Pipe x (m)':                   'pipe_x',
            'Pipe depth below surface (m)': 'pipe_depth',
            'Soil εr':                      'soil_er',
            'Soil σ (S/m)':                 'soil_sig',
        }

    total = len(param_defs) * n_steps + 1
    count = 0
    bscan_base, _ = predict(model, g)
    count += 1
    if progress_cb: progress_cb(count / total, "Baseline computed")

    results, sweeps, bscans, geom_disps = {}, {}, {}, {}
    for pname, (pmin, pmax, _, _) in param_defs.items():
        vals = np.linspace(pmin, pmax, n_steps)
        sweeps[pname] = vals
        metrics_list, bscan_list, geom_list = [], [], []
        for i, v in enumerate(vals):
            b = baseline.copy()
            b[key_map[pname]] = float(v)
            if 'Tunnel' in scenario:
                void_y_b = SURFACE_Y - b.get('depth', b.get('void_y', 1.0))
                d2 = build_tunnel_geometry(b['void_x'], void_y_b,
                                           b['void_R'], b['void_delta'], oat_seed)
                gn, er_d = tunnel_to_model_input(d2, b['rock_er'], b['rock_sig'],
                                                  b['grot_er'], b['grot_sig'], 'Dry')
                if pname in SIGMA_PARAMS:
                    sl2 = {T_IDX_AIR: 0.0, T_IDX_GROT: b['grot_sig'],
                           T_IDX_ROCK: b['rock_sig'], T_IDX_VOID: 0.0}
                    disp = np.flipud(np.vectorize(sl2.get)(d2).astype(np.float32)[:, :T_FREE].T)
                else:
                    disp = er_d
            else:
                pipe_y_b = SURFACE_Y - b.get('pipe_depth', b.get('pipe_y', 0.85))
                em2, sm2, pm2 = build_pipe_geometry(
                    b['pipe_name'], b['pipe_x'], pipe_y_b,
                    b['soil_er'], b['soil_sig'], 1.0, 0.0)
                gn, er_d, _ = pipe_to_model_input(em2, sm2, pm2)
                disp = np.flipud(sm2[:, :P_FREE].T) if pname in SIGMA_PARAMS else er_d
            bs, _ = predict(model, gn)
            metrics_list.append(compute_metrics(bs, bscan_base))
            bscan_list.append(bs)
            geom_list.append(disp)
            count += 1
            if progress_cb: progress_cb(count / total, f"{pname}  ({i+1}/{n_steps})")
        results[pname]    = metrics_list
        bscans[pname]     = bscan_list
        geom_disps[pname] = geom_list

    elapsed = time.perf_counter() - t_start
    return results, sweeps, bscan_base, er_base, sig_base, bscans, geom_disps, elapsed

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
DT_NS        = 9.434617346998736e-3   # ns per timestep
T_END_NS     = 2121 * DT_NS           # ~20.0 ns (both datasets)
T_START_TUNNEL_NS  = 370 * DT_NS     # ~3.49 ns
T_START_PIPE_NS    = 250 * DT_NS     # ~2.36 ns

def make_figure(er_map, bscan, left_title, right_title, infer_ms,
                domain_w=2.0, domain_h=1.3, t_start_ns=T_START_TUNNEL_NS,
                ant_x_start=0.08, ant_x_end=1.96,
                pec_map=None, er_vmax=10.0):
    # Physical coordinates in metres / ns
    # ant_x_start/end define actual antenna coverage for the B-scan x-axis
    x_geom  = np.linspace(0, domain_w, er_map.shape[1])
    y_geom  = np.linspace(0, domain_h, er_map.shape[0])
    x_bscan = np.linspace(ant_x_start, ant_x_end, bscan.shape[1])
    y_bscan = np.linspace(t_start_ns, T_END_NS, bscan.shape[0])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'<b>Asset State</b> — {left_title}',
                        f'<b>Surrogate GPR Response</b> — {right_title}'),
        horizontal_spacing=0.18,
    )

    # Left: εr map
    fig.add_trace(
        go.Heatmap(z=er_map, x=x_geom, y=y_geom,
                   colorscale='Viridis', zmin=1, zmax=er_vmax,
                   colorbar=dict(title=dict(text='εr', side='right'),
                                 x=0.44, xanchor='left',
                                 len=0.85, thickness=12),
                   showscale=True),
        row=1, col=1,
    )

    # Overlay PEC pixels in orange for pipeline
    if pec_map is not None and pec_map.max() > 0:
        pec_display = np.where(pec_map > 0.5, 1.0, np.nan)
        fig.add_trace(
            go.Heatmap(z=pec_display, x=x_geom, y=y_geom,
                       colorscale=[[0, 'rgba(242,128,13,0.9)'],
                                   [1, 'rgba(242,128,13,0.9)']],
                       zmin=0, zmax=1,
                       showscale=False, hoverinfo='skip'),
            row=1, col=1,
        )

    # Right: predicted B-scan
    fig.add_trace(
        go.Heatmap(z=bscan, x=x_bscan, y=y_bscan,
                   colorscale='Gray', zmin=0, zmax=1,
                   colorbar=dict(title=dict(text='Amplitude', side='right'),
                                 x=1.02, xanchor='left',
                                 len=0.85, thickness=12),
                   showscale=True),
        row=1, col=2,
    )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=70, t=50, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Arial', size=12),
        annotations=list(fig.layout.annotations),
    )
    fig.update_xaxes(title_text='Horizontal distance (m)', row=1, col=1)
    fig.update_xaxes(title_text='Horizontal distance (m)', row=1, col=2)
    fig.update_yaxes(title_text='Depth (m)', autorange='reversed', row=1, col=1)
    fig.update_yaxes(title_text='Travel time (ns)', autorange='reversed', row=1, col=2)
    if pec_map is not None:
        # Restore equal-aspect on asset state so pipe looks circular
        fig.update_yaxes(scaleanchor='x', scaleratio=1, row=1, col=1)
        # Match GPR response visual width to asset state:
        # scaleratio = (ant_x_range * domain_h) / (t_range_ns * domain_w)
        x_range = ant_x_end - ant_x_start
        t_range = T_END_NS - t_start_ns
        sr = (x_range * domain_h) / (t_range * domain_w)
        fig.update_yaxes(scaleanchor='x2', scaleratio=sr, row=1, col=2)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
LOGO_PATH = ROOT / 'assets' / 'logo.png'
logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()

st.markdown(f"""
<div class="dt-header">
  <div style="display:flex; align-items:center; justify-content:space-between;">
    <div>
      <h1 style="margin:0; color:#e0f7fa; font-size:1.45rem; font-weight:700;">
        GPR4Net: A Lightweight Surrogate for GPR Forward Simulation<br>
        Towards Predictive Digital Twin
      </h1>
      <p style="margin:0.35rem 0 0 0; color:#80cbc4; font-size:0.82rem;">
        Part of <b style="color:#b2ebf2;">M-Twin4US</b> ·
        MSCA Project ·
        Maintenance-oriented Digital Twin for Underground Infrastructure
      </p>
      <p style="margin:0.2rem 0 0 0; font-size:0.80rem;">
        <a href="https://github.com/M-Twin4US/GPR4Net" target="_blank"
           style="color:#80deea; text-decoration:none;">
          ⬡ github.com/M-Twin4US/GPR4Net
        </a>
        &nbsp;·&nbsp; Dataset &amp; source code
      </p>
    </div>
    <div style="background:white; border-radius:6px; padding:6px 12px;
                display:flex; align-items:center; flex-shrink:0; margin-left:1.5rem;
                height:68px; box-sizing:border-box; overflow:hidden;">
      <img src="data:image/png;base64,{logo_b64}"
           style="height:40px; width:auto; display:block; object-fit:contain;">
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.spinner('Loading surrogate models...'):
    model_800, model_600, model_pipe = load_models()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Scenario")
    scenario = st.radio("Select scenario",
                        ["Tunnel Lining", "Pipeline"],
                        label_visibility='collapsed')
    st.divider()

    # ── TUNNEL ────────────────────────────────────────────────────────────────
    if 'Tunnel' in scenario:
        freq = st.radio("Antenna Frequency", ["800 MHz", "600 MHz"], horizontal=True)

        st.markdown("##### 📍 Void Position")
        c1, c2 = st.columns(2)
        void_x = c1.slider("x (m)",     0.20, 1.80, 1.00, 0.01)
        _void_depth = c2.slider("Depth below surface (m)", 0.20, 0.40, 0.30, 0.005,
                                key='sb_void_depth')
        void_y = 1.3 - _void_depth

        st.markdown("##### 🔵 Void Geometry")
        c1, c2 = st.columns(2)
        void_R     = c1.slider("Radius (m)",    0.024, 0.080, 0.050, 0.001)
        void_delta = c2.slider("Irregularity",  0.10,  0.40,  0.20,  0.01,
                               help="0.10 = circular · 0.40 = irregular")
        void_state = st.radio("Void state", ["Dry", "Saturated"], horizontal=True)

        st.markdown("##### 🧱 Grouting / Lining")
        c1, c2 = st.columns(2)
        grot_er  = c1.slider("εr",      5.0,  10.0, 7.5,  0.1,  key='grot_er')
        grot_sig = c2.slider("σ (S/m)", 1e-3, 1e-2, 5e-3, 1e-4, key='grot_sig', format="%.4f")

        st.markdown("##### ⛰️ Host Rock")
        c1, c2 = st.columns(2)
        rock_er  = c1.slider("εr",      3.0,  8.0,  5.0,  0.1,  key='rock_er')
        rock_sig = c2.slider("σ (S/m)", 0.0,  1e-3, 5e-4, 1e-5, key='rock_sig', format="%.5f")

        st.markdown("##### 〰️ Interface Shape")
        seed = st.slider("Seed", 0, 999, 42,
                         help="Changes the rock–grouting boundary curvature")

    # ── PIPELINE ──────────────────────────────────────────────────────────────
    else:
        pipe_name = st.selectbox("Pipe type", list(PIPE_SPECS.keys()))

        st.markdown("##### 📍 Pipe Position")
        c1, c2 = st.columns(2)
        pipe_x = c1.slider("x (m)",     0.50,  1.00,  0.75,  0.01)
        _pipe_depth = c2.slider("Depth below surface (m)", 0.325, 0.575, 0.45, 0.005,
                                key='sb_pipe_depth')
        pipe_y = 1.3 - _pipe_depth

        st.markdown("##### 🌍 Soil Properties")
        c1, c2 = st.columns(2)
        soil_er  = c1.slider("εr", 3.0, 10.0, 6.0, 0.1)
        soil_sig = c2.select_slider(
            "σ (S/m)",
            options=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=1e-5, format_func=lambda x: f"{x:.0e}",
        )
        if PIPE_SPECS[pipe_name]['material'] == 'plastic':
            st.markdown("##### 🔧 Pipe Material")
            c1, c2 = st.columns(2)
            pipe_er  = c1.slider("εr", 3.0, 4.0, 3.5, 0.1)
            pipe_sig = c2.select_slider(
                "σ (S/m)",
                options=[1e-11, 5e-11, 1e-10],
                value=1e-11, format_func=lambda x: f"{x:.0e}",
            )
        else:
            pipe_er, pipe_sig = 1.0, 0.0   # PEC — not editable

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_fwd, tab_gen, tab_oat = st.tabs([
    "⚛  Single Simulation",
    "⛁  Synthetic Dataset Generator",
    "📊  Uncertainty Quantification",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FORWARD SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_fwd:
    with st.spinner("Running surrogate model..."):
        if 'Tunnel' in scenario:
            data             = build_tunnel_geometry(void_x, void_y, void_R, void_delta, seed)
            geom_np, er_disp = tunnel_to_model_input(data, rock_er, rock_sig,
                                                      grot_er, grot_sig, void_state)
            model            = model_800 if freq == "800 MHz" else model_600
            bscan, infer_ms  = predict(model, geom_np)
            fig = make_figure(er_disp, bscan,
                              left_title='Tunnel Lining',
                              right_title=f'{freq} Prediction',
                              infer_ms=infer_ms, domain_w=2.0, domain_h=1.3,
                              t_start_ns=T_START_TUNNEL_NS,
                              ant_x_start=0.08, ant_x_end=1.96,
                              er_vmax=10.0)
        else:
            er_m, sig_m, pec_m       = build_pipe_geometry(pipe_name, pipe_x, pipe_y,
                                                            soil_er, soil_sig, pipe_er, pipe_sig)
            geom_np, er_disp, pec_disp = pipe_to_model_input(er_m, sig_m, pec_m)
            bscan, infer_ms           = predict(model_pipe, geom_np)
            fig = make_figure(er_disp, bscan,
                              left_title=f'Pipeline ({pipe_name})',
                              right_title='1 GHz Prediction',
                              infer_ms=infer_ms, domain_w=1.5, domain_h=1.3,
                              t_start_ns=T_START_PIPE_NS,
                              ant_x_start=0.0, ant_x_end=1.5,
                              pec_map=pec_disp, er_vmax=10.0)

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    c1, c2, c3, c4 = st.columns(4)
    if 'Tunnel' in scenario:
        c1.metric("Void position",  f"x={void_x:.2f} m, depth={1.3-void_y:.2f} m")
        c2.metric("Void radius",    f"{void_R*100:.1f} cm")
        c3.metric("Void state",     void_state)
        c4.metric("Inference time", f"{round(infer_ms)} ms")
    else:
        c1.metric("Pipe type",      pipe_name)
        c2.metric("Pipe position",  f"x={pipe_x:.2f} m, depth={1.3-pipe_y:.2f} m")
        c3.metric("Soil εr",        f"{soil_er:.1f}")
        c4.metric("Inference time", f"{round(infer_ms)} ms")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — UNCERTAINTY QUANTIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_oat:
    st.markdown("### B-scan Prediction Variability")

    if 'Tunnel' in scenario:
        st.markdown(
            "The void horizontal position is directly observable from the GPR hyperbola apex "
            "and is treated as **known**. All other subsurface parameters — void depth, shape, "
            "and material properties — are uncertain and modelled as normal distributions N(μ, σ²). "
            "The surrogate propagates this uncertainty through the full EM simulation, producing N "
            "predicted B-scans. The mean and standard deviation of those B-scans directly show "
            "**what you expect to see** and **where in the image uncertainty is highest** — "
            "something no analytical formula can produce."
        )
    else:
        st.markdown(
            "The pipe horizontal position is directly observable from the GPR hyperbola apex "
            "and is treated as **known**. Soil properties and pipe depth are uncertain and "
            "modelled as normal distributions N(μ, σ²). "
            "The surrogate propagates this uncertainty through the full EM simulation, producing N "
            "predicted B-scans showing how the hyperbolic reflection pattern varies."
        )

    # ── Uncertain parameters ───────────────────────────────────────────────────
    TT_UNCERTAIN_PIPE = {
        'Pipe depth below surface (m)': (0.325, 0.575, 0.45, '{:.3f}'),
        'Soil εr':           (3.0,   10.0,  6.5,   '{:.1f}'),
        'Soil σ (S/m)':      (1e-7,  1e-3,  1e-5,  '{:.1e}'),
    }
    # Depth below surface = 1.3 - void_y  →  range [0.20, 0.40] m
    TT_UNCERTAIN = {
        'Depth below surface (m)': (0.20,  0.40,  0.30,  '{:.3f}'),
        'Void radius (m)':         (0.024, 0.080, 0.050, '{:.3f}'),
        'Void irregularity':       (0.10,  0.40,  0.25,  '{:.2f}'),
        'Rock εr':                 (3.0,   8.0,   5.5,   '{:.1f}'),
        'Rock σ (S/m)':            (0.0,   1e-3,  5e-4,  '{:.1e}'),
        'Grouting εr':             (5.0,   10.0,  7.5,   '{:.1f}'),
        'Grouting σ (S/m)':        (1e-3,  1e-2,  5e-3,  '{:.1e}'),
    }
    C_LIGHT = 3e8   # speed of light (m/s)

    st.markdown("#### ⚙️ Settings & parameter distributions")
    tt_c1, tt_c2, tt_c3 = st.columns(3)
    tt_n    = tt_c1.select_slider("Monte Carlo samples (N)",
                                  options=[64, 128, 256, 512], value=128)
    tt_seed = tt_c2.slider("Random seed", 0, 999, 0, key='tt_seed')
    if 'Tunnel' in scenario:
        tt_freq  = tt_c3.radio("Frequency", ["800 MHz", "600 MHz"],
                               horizontal=True, key='tt_freq')
        tt_model = model_800 if tt_freq == "800 MHz" else model_600
    else:
        tt_c3.markdown("**Frequency:** 1 GHz")
        tt_model = model_pipe

    # Fixed / known parameters
    st.markdown("**Fixed / known parameters** *(directly observable from GPR)*")
    fc1, fc2 = st.columns(2)
    if 'Tunnel' in scenario:
        tt_void_x     = fc1.slider("Void x (m)",     0.20, 1.80, 1.00, 0.01, key='tt_vx')
        tt_iface_seed = fc2.slider("Interface seed", 0, 999, 42, key='tt_iseed')
        tt_pipe_x     = 0.75   # unused
    else:
        tt_pipe_x     = fc1.slider("Pipe x (m)",     0.50, 1.00, 0.75, 0.01, key='tt_px')
        tt_pipe_name  = fc2.selectbox("Pipe type", list(PIPE_SPECS.keys()), key='tt_pname')
        tt_void_x     = tt_pipe_x   # reuse variable name
        tt_iface_seed = 42

    # Uncertain parameters
    st.markdown(
        "**Uncertain parameters** — set μ (best estimate) and σ (uncertainty). "
        "<small>Default σ = range/6.</small>",
        unsafe_allow_html=True)

    tt_params = {}
    param_list = list((TT_UNCERTAIN if 'Tunnel' in scenario else TT_UNCERTAIN_PIPE).items())
    for row_start in range(0, len(param_list), 2):
        cols = st.columns(2)
        for ci, (pname, (pmin, pmax, pdef, pfmt)) in \
                enumerate(param_list[row_start:row_start + 2]):
            with cols[ci]:
                st.markdown(
                    f"**{pname}** <small>[{pfmt.format(pmin)}, {pfmt.format(pmax)}]</small>",
                    unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                mu  = c1.number_input("μ", value=float(pdef),
                                      min_value=float(pmin), max_value=float(pmax),
                                      step=float((pmax-pmin)/20),
                                      key=f'tt_mu_{pname}', format="%.4g")
                sig = c2.number_input("σ", value=float((pmax-pmin)/6),
                                      min_value=0.0,
                                      max_value=float((pmax-pmin)/2),
                                      step=float((pmax-pmin)/40),
                                      key=f'tt_sig_{pname}', format="%.4g")
                tt_params[pname] = (mu, sig, pmin, pmax)

    # Clear stale results if parameter set changed
    if 'tt_results' in st.session_state:
        old_keys = set(st.session_state['tt_results'].get('samples', {}).keys())
        new_keys = {'Void x (m)'} | set(TT_UNCERTAIN.keys())
        if old_keys != new_keys:
            del st.session_state['tt_results']

    tt_btn = st.button("▶  Run B-scan UQ", type="primary", key='tt_btn')

    if tt_btn:
        rng = np.random.default_rng(tt_seed)
        samples = {}
        samples['Void x (m)'] = np.full(tt_n, tt_void_x)
        # Sample uncertain parameters
        for pname, (mu, sig, pmin, pmax) in tt_params.items():
            s = rng.normal(mu, sig, tt_n) if sig > 1e-12 else np.full(tt_n, mu)
            samples[pname] = np.clip(s, pmin, pmax)

        # ── Deterministic prediction at mean parameter values ────────────────
        if 'Tunnel' in scenario:
            mu_depth = tt_params['Depth below surface (m)'][0]
            mu_vR    = tt_params['Void radius (m)'][0]
            mu_vd    = tt_params['Void irregularity'][0]
            mu_rer   = tt_params['Rock εr'][0]
            mu_rs    = tt_params['Rock σ (S/m)'][0]
            mu_ger   = tt_params['Grouting εr'][0]
            mu_gs    = tt_params['Grouting σ (S/m)'][0]
            d_mean   = build_tunnel_geometry(tt_void_x, SURFACE_Y - mu_depth,
                                              mu_vR, mu_vd, tt_iface_seed)
            g_mean, _ = tunnel_to_model_input(d_mean, mu_rer, mu_rs, mu_ger, mu_gs, 'Dry')
        else:
            mu_pdepth = tt_params['Pipe depth below surface (m)'][0]
            mu_ser    = tt_params['Soil εr'][0]
            mu_ss     = tt_params['Soil σ (S/m)'][0]
            em_m, sm_m, pm_m = build_pipe_geometry(
                tt_pipe_name, tt_pipe_x, SURFACE_Y - mu_pdepth,
                mu_ser, mu_ss, 1.0, 0.0)
            g_mean, _, _ = pipe_to_model_input(em_m, sm_m, pm_m)
        bscan_at_mean, _ = predict(tt_model, g_mean)

        tt_prog = st.progress(0.0)
        tt_stat = st.empty()
        all_bscans_tt, apex_times = [], []

        for i in range(tt_n):
            if 'Tunnel' in scenario:
                depth = samples['Depth below surface (m)'][i]
                vy    = SURFACE_Y - depth
                vx    = samples['Void x (m)'][i]
                vR    = samples['Void radius (m)'][i]
                vd    = samples['Void irregularity'][i]
                rer   = samples['Rock εr'][i]
                rs    = samples['Rock σ (S/m)'][i]
                ger   = samples['Grouting εr'][i]
                gs    = samples['Grouting σ (S/m)'][i]
                d     = build_tunnel_geometry(vx, vy, vR, vd, tt_iface_seed)
                g, _  = tunnel_to_model_input(d, rer, rs, ger, gs, 'Dry')
                t_expected_ns = 2.0 * depth / (C_LIGHT / np.sqrt(ger)) * 1e9
                t_start_uq    = T_START_TUNNEL_NS
                known_x       = vx
                domain_w_uq   = 2.0
            else:
                pipe_depth = samples['Pipe depth below surface (m)'][i]
                pipe_y_val = SURFACE_Y - pipe_depth
                ser   = samples['Soil εr'][i]
                ss    = samples['Soil σ (S/m)'][i]
                em, sm_arr, pm = build_pipe_geometry(
                    tt_pipe_name, tt_pipe_x, pipe_y_val, ser, ss, 1.0, 0.0)
                g, _, _ = pipe_to_model_input(em, sm_arr, pm)
                t_expected_ns = 2.0 * pipe_depth / (C_LIGHT / np.sqrt(ser)) * 1e9
                t_start_uq    = T_START_PIPE_NS
                known_x       = tt_pipe_x
                domain_w_uq   = 1.5

            bs, _ = predict(tt_model, g)
            all_bscans_tt.append(bs)

            # Physics-based apex detection
            row_expected = int((t_expected_ns - t_start_uq) /
                               (T_END_NS - t_start_uq) * 255)
            r_lo     = max(0,   row_expected - 25)
            r_hi     = min(255, row_expected + 25)
            col      = int(np.clip(known_x / domain_w_uq * 256, 0, 255))
            peak_row = r_lo + int(np.argmax(bs[r_lo:r_hi+1, col]))
            t_apex   = t_start_uq + peak_row / 255 * (T_END_NS - t_start_uq)
            apex_times.append(t_apex)

            tt_prog.progress((i+1)/tt_n)
            tt_stat.text(f"⏳ Sample {i+1}/{tt_n}")

        tt_prog.empty(); tt_stat.empty()
        st.session_state['tt_results'] = dict(
            bscans=all_bscans_tt, samples=samples,
            apex_times=apex_times, n=tt_n,
            bscan_at_mean=bscan_at_mean,
            scenario=scenario,
        )

    tt_data = st.session_state.get('tt_results')
    if tt_data and tt_data['scenario'] == scenario:
        all_bs_tt  = np.array(tt_data['bscans'])
        apex_times = np.array(tt_data['apex_times'])
        samples    = tt_data['samples']
        N_tt       = tt_data['n']
        is_pipe_uq = 'Pipeline' in scenario
        void_x_tt  = (tt_data['samples'].get('Pipe depth (m)',
                      tt_data['samples'].get('Void x (m)', np.array([1.0])))).mean()
        void_x_tt  = tt_pipe_x if is_pipe_uq else tt_data['samples']['Void x (m)'].mean()
        bs_at_mean = tt_data.get('bscan_at_mean', all_bs_tt.mean(axis=0))
        t_start_r  = T_START_PIPE_NS if is_pipe_uq else T_START_TUNNEL_NS
        dw         = 1.5 if is_pipe_uq else 2.0
        t_ns       = np.linspace(t_start_r, T_END_NS, 256)
        bs_std_tt  = all_bs_tt.std(axis=0)
        x_bs       = np.linspace(0, dw, 256)
        col_idx    = int(np.clip(void_x_tt / dw * 256, 0, 255))

        ci_lo  = np.percentile(apex_times, 2.5)
        ci_hi  = np.percentile(apex_times, 97.5)
        t_mean = apex_times.mean()
        t_std  = apex_times.std()

        st.caption(
            f"✅ {N_tt} Monte Carlo samples  ·  "
            f"Apex travel time: μ = {t_mean:.2f} ns · σ = {t_std:.2f} ns · "
            f"95% PI = [{ci_lo:.2f}, {ci_hi:.2f}] ns"
        )

        # ── Panel 1: B-scan at mean + Uncertainty map ─────────────────────────
        st.markdown("#### B-scan at Mean Parameters & Prediction Variability Map")
        st.caption(
            "**Left:** B-scan predicted by the surrogate using the **mean (μ) values** of "
            "all uncertain parameters — the central estimate of what the GPR response "
            "would look like. "
            "**Right:** pixel-wise standard deviation (STD) across all N Monte Carlo "
            "samples — brighter regions indicate where the predicted signal is most "
            "responsive to variations in the input parameters. "
            "The dashed line marks the known horizontal position."
        )
        fig_maps = make_subplots(rows=1, cols=2,
                                  subplot_titles=['B-scan at μ (central estimate)',
                                                  'Prediction variability map (pixel-wise STD)'],
                                  horizontal_spacing=0.12)
        fig_maps.add_trace(
            go.Heatmap(z=bs_at_mean, x=x_bs, y=t_ns,
                       colorscale='Gray', zmin=0, zmax=1,
                       colorbar=dict(title='Amp', thickness=10, x=0.44,
                                     xanchor='left'),
                       showscale=True), row=1, col=1)
        fig_maps.add_trace(
            go.Heatmap(z=bs_std_tt, x=x_bs, y=t_ns,
                       colorscale='Hot', zmin=0,
                       colorbar=dict(title='STD', thickness=10, x=1.02,
                                     xanchor='left'),
                       showscale=True), row=1, col=2)
        for col in [1, 2]:
            fig_maps.add_vline(x=void_x_tt, line_dash='dash',
                               line_color='#2980b9', line_width=1.5,
                               row=1, col=col)
        fig_maps.update_yaxes(autorange='reversed')
        fig_maps.update_xaxes(title_text='Horizontal distance (m)', row=1, col=1)
        fig_maps.update_xaxes(title_text='Horizontal distance (m)', row=1, col=2)
        fig_maps.update_yaxes(title_text='Travel time (ns)', row=1, col=1)
        fig_maps.update_layout(
            height=400, margin=dict(l=60, r=60, t=50, b=40),
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(family='Arial', size=11),
            title=dict(text=f'void x = {void_x_tt:.2f} m  ·  N = {N_tt} samples',
                       font_size=12, x=0.5))
        st.plotly_chart(fig_maps, use_container_width=True, config=PLOTLY_CONFIG)

        # ── Panel 2: Apex travel time vs all uncertain parameters ────────────────
        st.markdown("#### Apex Travel Time vs Uncertain Parameters")
        st.caption(
            "Each dot is one Monte Carlo sample. Red line = mean trend. "
            "Blue band = 95% prediction interval. "
            "Steep slope = that parameter strongly influences travel time."
        )
        up_pnames = [p for p in samples.keys() if p != 'Void x (m)']
        ncols_up  = 3
        nrows_up  = int(np.ceil(len(up_pnames) / ncols_up))
        n_bins_up = min(8, N_tt // 5)

        fig_sc2 = make_subplots(rows=nrows_up, cols=ncols_up,
                                 horizontal_spacing=0.10, vertical_spacing=0.15)
        for pi, pname in enumerate(up_pnames):
            r = pi // ncols_up + 1
            c = pi % ncols_up + 1
            xp_raw = samples[pname]

            # Transform Grouting εr → √εr so the t ∝ √εr relationship is linear
            if pname == 'Grouting εr':
                xp      = np.sqrt(xp_raw)
                x_label = '√(Grouting εr)'
            else:
                xp      = xp_raw
                x_label = pname

            q_edges = np.percentile(xp, np.linspace(0, 100, n_bins_up+1))
            bidx    = np.digitize(xp, q_edges[1:-1])
            bx, bm, blo, bhi = [], [], [], []
            for b in range(n_bins_up):
                mask = bidx == b
                if mask.sum() < 2:
                    continue
                bx.append(float(np.mean(xp[mask])))
                bm.append(float(np.mean(apex_times[mask])))
                blo.append(float(np.percentile(apex_times[mask], 2.5)))
                bhi.append(float(np.percentile(apex_times[mask], 97.5)))

            fig_sc2.add_trace(go.Scatter(
                x=xp, y=apex_times, mode='markers',
                marker=dict(size=4, color='#aab4c8', opacity=0.5),
                showlegend=False), row=r, col=c)
            if bx:
                fig_sc2.add_trace(go.Scatter(
                    x=bx + bx[::-1], y=bhi + blo[::-1],
                    fill='toself', fillcolor='rgba(41,128,185,0.18)',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=(pi == 0), name='95% PI'), row=r, col=c)
                fig_sc2.add_trace(go.Scatter(
                    x=bx, y=bm, mode='lines+markers',
                    line=dict(color='#c0392b', width=2),
                    marker=dict(size=6),
                    showlegend=(pi == 0), name='Mean'), row=r, col=c)

            fig_sc2.update_xaxes(title_text=x_label, title_font_size=8, row=r, col=c)
            if c == 1:
                fig_sc2.update_yaxes(title_text='Apex travel time (ns)',
                                     title_font_size=8, row=r, col=c)

        fig_sc2.update_layout(
            height=300 * nrows_up,
            margin=dict(l=60, r=30, t=30, b=30),
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(family='Arial', size=9),
            legend=dict(orientation='h', x=0.5, xanchor='center', y=1.02))
        st.plotly_chart(fig_sc2, use_container_width=True, config=PLOTLY_CONFIG)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SYNTHETIC DATASET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
import zipfile, io, tempfile, os
import pandas as pd

def numpy_to_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

# Training parameter bounds — user cannot exceed these
TUNNEL_BOUNDS = {
    'void_x_m':        (0.20,  1.80),
    'depth_m':         (0.20,  0.40),
    'void_R_m':        (0.024, 0.080),
    'void_delta':      (0.10,  0.40),
    'rock_er':         (3.0,   8.0),
    'rock_sigma':      (0.0,   1e-3),
    'grouting_er':     (5.0,   10.0),
    'grouting_sigma':  (1e-3,  1e-2),
}
PIPE_BOUNDS = {
    'pipe_x_m':        (0.50,  1.00),
    'pipe_depth_m':    (0.725, 0.975),
    'soil_er':         (3.0,   10.0),
    'soil_sigma':      (1e-7,  1e-3),
}

with tab_gen:
    st.markdown("### Synthetic Dataset Generator")
    st.markdown(
        "Generate a labelled synthetic GPR dataset using the surrogate model. "
        "Parameters are sampled using the **Sobol sequence** (same strategy as "
        "the training dataset) within your specified ranges. Each sample produces "
        "a predicted B-scan and the corresponding material maps, ready for "
        "training data-driven interpretation models. "
        "The surrogate reduces generation time from **hours to seconds**."
    )

    with st.expander("⚙️ Dataset configuration", expanded=True):
        gc1, gc2, gc3 = st.columns(3)
        _gen_options = ["Tunnel-800 MHz", "Tunnel-600 MHz", "Pipeline"]
        # Auto-sync to sidebar scenario when user switches scenarios
        _expected_default = 'Pipeline' if 'Pipeline' in scenario else 'Tunnel-800 MHz'
        if st.session_state.get('_last_scenario') != scenario:
            st.session_state['gen_scenario'] = _expected_default
            st.session_state['_last_scenario'] = scenario
        gen_scenario = gc1.selectbox(
            "Scenario",
            _gen_options,
            key='gen_scenario')
        gen_n_raw = gc2.select_slider(
            "Number of samples (N)",
            options=[64, 128, 256, 512], value=256,
            key='gen_n')
        gen_seed = gc3.slider("Sobol seed", 0, 99, 42, key='gen_seed')

        # ── Void state (tunnel) / Pipe type (pipeline) ────────────────────────
        if 'Pipeline' not in gen_scenario:
            gen_void_state = gc1.selectbox(
                "Void state",
                ["Dry only", "Saturated only", "50/50 split", "Random"],
                key='gen_vs')
            gen_pipe_name  = 'Metallic DN250'   # unused for tunnel
            gen_void_state_val = gen_void_state
        else:
            gen_pipe_name  = gc1.selectbox(
                "Pipe type", list(PIPE_SPECS.keys()), key='gen_pipe_name')
            gen_void_state = "N/A"
            gen_void_state_val = "N/A"

        st.markdown("**Parameter ranges** *(slide to narrow from training bounds)*")
        st.markdown(
            "<small>Grey bounds = training dataset limits · "
            "Adjust to focus on your site-specific conditions.</small>",
            unsafe_allow_html=True)

        gen_ranges = {}
        if 'Pipeline' not in gen_scenario:
            bounds = TUNNEL_BOUNDS
            labels = {
                'void_x_m':       'Void x (m)',
                'depth_m':        'Depth below surface (m)',
                'void_R_m':       'Void radius (m)',
                'void_delta':     'Void irregularity',
                'rock_er':        'Rock εr',
                'rock_sigma':     'Rock σ (S/m)',
                'grouting_er':    'Grouting εr',
                'grouting_sigma': 'Grouting σ (S/m)',
            }
            fmts = {
                'void_x_m': '%.2f', 'depth_m': '%.3f', 'void_R_m': '%.3f',
                'void_delta': '%.2f', 'rock_er': '%.1f',
                'rock_sigma': '%.5f', 'grouting_er': '%.1f',
                'grouting_sigma': '%.4f',
            }
        else:
            bounds = PIPE_BOUNDS
            labels = {
                'pipe_x_m':    'Pipe x (m)',
                'pipe_depth_m':'Pipe depth (m)',
                'soil_er':     'Soil εr',
                'soil_sigma':  'Soil σ (S/m)',
            }
            fmts = {
                'pipe_x_m': '%.2f', 'pipe_depth_m': '%.3f',
                'soil_er': '%.1f', 'soil_sigma': '%.6f',
            }

        param_items = list(bounds.items())
        for row_start in range(0, len(param_items), 2):
            cols = st.columns(2)
            for ci, (key, (bmin, bmax)) in enumerate(param_items[row_start:row_start+2]):
                label = labels[key]
                fmt   = fmts[key]
                step  = (bmax - bmin) / 40
                rmin, rmax = cols[ci].slider(
                    f"{label}  [{fmt % bmin}, {fmt % bmax}]",
                    min_value=float(bmin), max_value=float(bmax),
                    value=(float(bmin), float(bmax)),
                    step=float(step), key=f'gen_{key}',
                    format=fmt)
                gen_ranges[key] = (rmin, rmax)

    gen_btn = st.button("▶  Generate Dataset", type="primary", key='gen_btn')

    if gen_btn:
        from scipy.stats.qmc import Sobol as SobolSampler

        param_keys = list(gen_ranges.keys())
        D    = len(param_keys)
        m    = int(np.ceil(np.log2(max(gen_n_raw, 2))))
        N    = 2 ** m
        sampler = SobolSampler(d=D, scramble=True, seed=gen_seed)
        raw  = sampler.random_base2(m)   # (N, D) in [0,1]

        # Scale to user-defined ranges
        lo   = np.array([gen_ranges[k][0] for k in param_keys])
        hi   = np.array([gen_ranges[k][1] for k in param_keys])
        params_arr = raw * (hi - lo) + lo   # (N, D)

        # Assign void state
        if gen_void_state == "Dry only":
            void_states = ['dry'] * N
        elif gen_void_state == "Saturated only":
            void_states = ['saturated'] * N
        elif gen_void_state == "50/50 split":
            void_states = ['dry'] * (N // 2) + ['saturated'] * (N - N // 2)
        elif gen_void_state == "Random":
            rng_vs = np.random.default_rng(gen_seed)
            void_states = ['saturated' if v else 'dry'
                           for v in rng_vs.integers(0, 2, N)]
        else:
            void_states = ['N/A'] * N

        # Select model
        if gen_scenario == "Tunnel-800 MHz":
            gen_model = model_800
        elif gen_scenario == "Tunnel-600 MHz":
            gen_model = model_600
        else:
            gen_model = model_pipe

        gen_prog = st.progress(0.0)
        gen_stat = st.empty()

        rows_csv   = []
        bscans_out = []
        geom_out   = []
        raw_geom_out = []

        for i in range(N):
            p = params_arr[i]
            pdict = {k: float(v) for k, v in zip(param_keys, p)}

            if 'Pipeline' not in gen_scenario:
                depth  = pdict['depth_m']
                vy     = SURFACE_Y - depth
                vx     = pdict['void_x_m']
                vR     = pdict['void_R_m']
                vd     = pdict['void_delta']
                rer    = pdict['rock_er']
                rs     = pdict['rock_sigma']
                ger    = pdict['grouting_er']
                gs     = pdict['grouting_sigma']
                vs     = void_states[i]

                d_geom = build_tunnel_geometry(vx, vy, vR, vd, gen_seed)
                geom_np, er_disp = tunnel_to_model_input(
                    d_geom, rer, rs, ger, gs, vs.capitalize())

                # Raw material maps (physical units, 2-channel)
                void_er  = 80.0 if vs == 'saturated' else 1.0
                void_sig = 5e-4 if vs == 'saturated' else 0.0
                er_lut  = {T_IDX_AIR: 1.0, T_IDX_GROT: ger,
                           T_IDX_ROCK: rer, T_IDX_VOID: void_er}
                sig_lut = {T_IDX_AIR: 0.0, T_IDX_GROT: gs,
                           T_IDX_ROCK: rs,  T_IDX_VOID: void_sig}
                er_raw  = np.flipud(np.vectorize(er_lut.get)(d_geom)
                                    .astype(np.float32)[:, :T_FREE].T)
                sig_raw = np.flipud(np.vectorize(sig_lut.get)(d_geom)
                                    .astype(np.float32)[:, :T_FREE].T)
                from scipy.ndimage import zoom as _zoom
                raw_geom = np.stack([
                    _zoom(er_raw,  (256/er_raw.shape[0],  256/er_raw.shape[1]),  order=1),
                    _zoom(sig_raw, (256/sig_raw.shape[0], 256/sig_raw.shape[1]), order=1),
                ], axis=0).astype(np.float32)

                row = {'sample_id': i+1, 'void_x_m': vx,
                       'depth_below_surface_m': depth, 'void_y_gprmax_m': vy,
                       'void_R_m': vR, 'void_delta': vd, 'void_state': vs,
                       'rock_er': rer, 'rock_sigma': rs,
                       'grouting_er': ger, 'grouting_sigma': gs}
            else:
                px   = pdict['pipe_x_m']
                py   = pdict['pipe_depth_m']
                ser  = pdict['soil_er']
                ss   = pdict['soil_sigma']

                em, sm, pm = build_pipe_geometry(
                    gen_pipe_name, px, py, ser, ss, 1.0, 0.0)
                geom_np, er_disp, _ = pipe_to_model_input(em, sm, pm)
                er_raw  = np.flipud(em[:, :P_FREE].T).astype(np.float32)
                sig_raw = np.flipud(sm[:, :P_FREE].T).astype(np.float32)
                from scipy.ndimage import zoom as _zoom
                raw_geom = np.stack([
                    _zoom(er_raw,  (256/er_raw.shape[0],  256/er_raw.shape[1]),  order=1),
                    _zoom(sig_raw, (256/sig_raw.shape[0], 256/sig_raw.shape[1]), order=1),
                ], axis=0).astype(np.float32)

                row = {'sample_id': i+1, 'pipe_x_m': px,
                       'pipe_depth_m': py, 'soil_er': ser,
                       'soil_sigma': ss, 'pipe_type': gen_pipe_name}

            bs, _ = predict(gen_model, geom_np)
            bscans_out.append(bs)
            geom_out.append(geom_np)
            raw_geom_out.append(raw_geom)
            rows_csv.append(row)

            gen_prog.progress((i+1)/N)
            gen_stat.text(f"⏳ Sample {i+1}/{N}")

        gen_prog.empty(); gen_stat.empty()

        st.session_state['gen_results'] = dict(
            bscans=bscans_out, geoms=geom_out,
            raw_geoms=raw_geom_out, rows_csv=rows_csv,
            N=N, scenario=gen_scenario,
        )

    gen_data = st.session_state.get('gen_results')
    if gen_data and gen_data['scenario'] == gen_scenario:
        N_gen   = gen_data['N']
        bscans  = gen_data['bscans']
        rows_csv = gen_data['rows_csv']

        st.caption(f"✅ {N_gen} samples generated")

        # ── B-scan preview grid ────────────────────────────────────────────────
        st.markdown("#### Preview (first 9 samples)")
        n_prev = min(9, N_gen)
        ncols_p = 3
        nrows_p = int(np.ceil(n_prev / ncols_p))
        t_ns_gen = np.linspace(
            T_START_TUNNEL_NS if 'Pipeline' not in gen_scenario else T_START_PIPE_NS,
            T_END_NS, 256)

        fig_prev = make_subplots(rows=nrows_p, cols=ncols_p,
                                  horizontal_spacing=0.02, vertical_spacing=0.05)
        for i in range(n_prev):
            r = i // ncols_p + 1
            c = i % ncols_p + 1
            fig_prev.add_trace(
                go.Heatmap(z=bscans[i], colorscale='Gray', zmin=0, zmax=1,
                           showscale=False),
                row=r, col=c)
            fig_prev.update_xaxes(showticklabels=False, row=r, col=c)
            fig_prev.update_yaxes(showticklabels=False, autorange='reversed',
                                   row=r, col=c)
            fig_prev.update_yaxes(title_text=f'#{i+1}', title_font_size=8,
                                   row=r, col=c)
        fig_prev.update_layout(
            height=220 * nrows_p,
            margin=dict(l=30, r=10, t=20, b=10),
            paper_bgcolor='white', plot_bgcolor='white')
        st.plotly_chart(fig_prev, use_container_width=True, config=PLOTLY_CONFIG)

        # ── Build zip in memory ────────────────────────────────────────────────
        st.markdown("#### Download")
        with st.spinner("Packaging dataset zip..."):
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                # parameters.csv
                df_csv = pd.DataFrame(gen_data['rows_csv'])
                zf.writestr('parameters.csv', df_csv.to_csv(index=False))
                # per-sample arrays
                for i in range(N_gen):
                    zf.writestr(f'bscans/{i+1}.npy',
                                numpy_to_bytes(gen_data['bscans'][i]))
                    zf.writestr(f'geometry/{i+1}.npy',
                                numpy_to_bytes(gen_data['geoms'][i]))
                    zf.writestr(f'raw_geometry/{i+1}.npy',
                                numpy_to_bytes(gen_data['raw_geoms'][i]))
            zip_buf.seek(0)

        fname = f"GPR4Net_{gen_scenario.replace(' ', '_')}_{N_gen}samples.zip"
        st.download_button(
            label=f"⬇️  Download dataset ({N_gen} samples)",
            data=zip_buf,
            file_name=fname,
            mime='application/zip',
        )
        st.markdown(
            "<small>"
            "<b>bscans/</b> — predicted B-scan (256×256, normalised [0,1]) · "
            "<b>geometry/</b> — model input (3×256×256, normalised) · "
            "<b>raw_geometry/</b> — physical material maps (2×256×256): "
            "Ch0 = εr (physical), Ch1 = σ (S/m) · "
            "<b>parameters.csv</b> — all parameter values per sample"
            "</small>",
            unsafe_allow_html=True)
