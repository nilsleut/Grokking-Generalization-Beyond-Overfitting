"""
Grokking — Mechanistic Interpretability
========================================
Basiert auf: Nanda et al. (2023) "Progress Measures for Grokking
via Mechanistic Interpretability" (arXiv:2301.05217)

Was dieses Skript zeigt:
  1. Checkpoint-Training: Modell bei Step 3k (pre-grokking) und
     Step 10k (post-grokking) speichern
  2. Embedding-Geometrie: Token-Embeddings in 2D via PCA visualisieren
     → Erwartung: Kreis nach Grokking, Chaos davor
  3. Fourier-Analyse: dominante Frequenzen in den Embeddings
     → Erwartung: wenige scharfe Peaks nach Grokking
  4. Attention-Pattern: welche Tokens attended die =-Position?

Voraussetzung: grokking_correct.py muss im selben Ordner liegen
(GrokkingTransformer und Config werden importiert)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from dataclasses import dataclass
from pathlib import Path

# Importiere Modell und Config aus dem Trainings-Skript
from grokking_correct import GrokkingTransformer, Config


# ══════════════════════════════════════════════════════════════
# Schritt 1: Training mit Checkpoint-Speicherung
# ══════════════════════════════════════════════════════════════

CHECKPOINT_STEPS = [500, 1000, 1200, 1400, 1600, 1800, 2000,
                    2200, 2400, 2600, 2800, 3000,
                    5000, 7500, 10000, 15000, 20000]
CHECKPOINT_DIR   = Path("checkpoints")

# Standalone-Parameter (nicht Teil von Config)
TRAIN_FRAC   = 0.20
WEIGHT_DECAY = 1.0

def make_dataset(cfg, train_frac):
    pairs   = [(a, b, (a + b) % cfg.p)
               for a in range(cfg.p) for b in range(cfg.p)]
    inputs  = torch.tensor([[a, b, cfg.p] for a, b, _ in pairs], dtype=torch.long)
    labels  = torch.tensor([c for _, _, c in pairs],             dtype=torch.long)
    dataset = TensorDataset(inputs, labels)
    n_train = int(len(dataset) * train_frac)
    n_val   = len(dataset) - n_train
    torch.manual_seed(cfg.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    return (DataLoader(train_ds, cfg.batch_size, shuffle=True),
            DataLoader(val_ds,   cfg.batch_size, shuffle=False))


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def train_with_checkpoints(cfg, train_frac, weight_decay):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.manual_seed(cfg.seed)
    train_loader, val_loader = make_dataset(cfg, train_frac)
    model = GrokkingTransformer(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=weight_decay,
                                  betas=cfg.betas)
    def lr_lambda(s): return min(1.0, s / 500)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    checkpoint_steps_set = set(CHECKPOINT_STEPS)
    max_step = max(CHECKPOINT_STEPS)

    print(f"Training bis Step {max_step} mit Checkpoints bei: {CHECKPOINT_STEPS}")
    print(f"{'step':>8}  {'train_acc':>10}  {'val_acc':>10}")
    print("─" * 35)

    def infinite(loader):
        while True:
            yield from loader

    step = 0
    for x, y in infinite(train_loader):
        if step >= max_step:
            break
        model.train()
        x, y = x.to(cfg.device), y.to(cfg.device)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1

        if step in checkpoint_steps_set:
            t_acc = eval_acc(model, train_loader, cfg.device)
            v_acc = eval_acc(model, val_loader,   cfg.device)
            print(f"{step:8d}  {t_acc*100:9.2f}%  {v_acc*100:9.2f}%")
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'train_acc': t_acc,
                'val_acc': v_acc,
            }, CHECKPOINT_DIR / f"step_{step:06d}.pt")

    print(f"\nCheckpoints gespeichert in: {CHECKPOINT_DIR}/")
    return model


# ══════════════════════════════════════════════════════════════
# Schritt 2: Embedding-Geometrie (PCA)
# ══════════════════════════════════════════════════════════════

def load_checkpoint(step, cfg):
    path = CHECKPOINT_DIR / f"step_{step:06d}.pt"
    ckpt = torch.load(path, map_location='cpu')
    model = GrokkingTransformer(cfg)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt['train_acc'], ckpt['val_acc']


def plot_embedding_geometry(cfg, steps_to_plot=None):
    """
    PCA der Token-Embeddings für 0..p-1 zu verschiedenen Checkpoints.
    Nach Grokking: Embeddings liegen auf einem Kreis (Fourier-Struktur).
    Davor: keine erkennbare Struktur.
    """
    if steps_to_plot is None:
        steps_to_plot = [1000, 3000, 7500, 15000]

    fig, axes = plt.subplots(1, len(steps_to_plot),
                             figsize=(5 * len(steps_to_plot), 5))
    fig.suptitle('Embedding-Geometrie: Token-Embeddings 0–96 (PCA)',
                 fontsize=13, fontweight='bold')

    for ax, step in zip(axes, steps_to_plot):
        model, t_acc, v_acc = load_checkpoint(step, cfg)

        # Nur Zahlen-Tokens 0..p-1 (nicht '=' Token)
        emb = model.tok_emb.weight[:cfg.p].detach().numpy()  # [97, d_model]

        # PCA auf 2D
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(emb)  # [97, 2]

        # Farbe = Token-Wert (0..96)
        colors = np.arange(cfg.p)
        sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                        c=colors, cmap='hsv', s=40, alpha=0.8)

        # Token-Werte annotieren (nur jeden 10.)
        for i in range(0, cfg.p, 10):
            ax.annotate(str(i), emb_2d[i], fontsize=7,
                        ha='center', va='bottom', alpha=0.7)

        var_explained = pca.explained_variance_ratio_[:2].sum()
        ax.set_title(f'Step {step:,}\ntrain={t_acc*100:.0f}%  val={v_acc*100:.0f}%\n'
                     f'PCA var={var_explained*100:.1f}%',
                     fontsize=10)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('embedding_geometry.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gespeichert: embedding_geometry.png")


# ══════════════════════════════════════════════════════════════
# Schritt 3: Fourier-Analyse der Embeddings
# ══════════════════════════════════════════════════════════════

def plot_fourier_analysis(cfg, steps_to_plot=None):
    """
    FFT der Embedding-Dimensionen über Token-Index.
    Nach Grokking: scharfe Peaks bei wenigen Frequenzen (Fourier-Features).
    Nanda et al. (2023): das Netz lernt cos/sin Darstellungen der Zahlen mod p.
    """
    if steps_to_plot is None:
        steps_to_plot = [1000, 3000, 7500, 15000]

    fig, axes = plt.subplots(1, len(steps_to_plot),
                             figsize=(5 * len(steps_to_plot), 4))
    fig.suptitle('Fourier-Analyse der Token-Embeddings',
                 fontsize=13, fontweight='bold')

    for ax, step in zip(axes, steps_to_plot):
        model, t_acc, v_acc = load_checkpoint(step, cfg)
        emb = model.tok_emb.weight[:cfg.p].detach().numpy()  # [97, d_model]

        # FFT über Token-Dimension (Achse 0) für jede Embedding-Dimension
        fft_power = np.abs(np.fft.fft(emb, axis=0)) ** 2  # [97, d_model]

        # Mittlere Power über alle Embedding-Dimensionen
        mean_power = fft_power.mean(axis=1)  # [97]
        freqs = np.arange(cfg.p)

        # Nur erste Hälfte (symmetrisch)
        half = cfg.p // 2
        ax.bar(freqs[1:half], mean_power[1:half],
               color='#7c6af7', alpha=0.8, width=0.8)
        ax.set_xlabel('Frequenz k')
        ax.set_ylabel('Mittlere Fourier-Power')
        ax.set_title(f'Step {step:,}\nval={v_acc*100:.0f}%', fontsize=10)
        ax.set_xlim(0, half)

        # Top-3 Frequenzen markieren
        top3 = np.argsort(mean_power[1:half])[-3:] + 1
        for k in top3:
            ax.axvline(k, color='#f76a8c', linewidth=1.5,
                       alpha=0.7, linestyle='--')
            ax.text(k, mean_power[k] * 1.02, str(k),
                    ha='center', fontsize=8, color='#f76a8c')

    plt.tight_layout()
    plt.savefig('fourier_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gespeichert: fourier_analysis.png")


# ══════════════════════════════════════════════════════════════
# Schritt 3b: Fourier-Projektion (Kreis-Visualisierung)
# ══════════════════════════════════════════════════════════════

def find_dominant_frequency(emb, p):
    """
    Findet die dominante Fourier-Frequenz k in den Embeddings.
    FFT über Token-Achse, mittlere Power über alle Dimensionen,
    dann argmax über k=1..p//2.
    """
    fft_power = np.abs(np.fft.fft(emb, axis=0)) ** 2  # [p, d_model]
    mean_power = fft_power.mean(axis=1)                # [p]
    half = p // 2
    # k=0 ignorieren (DC-Komponente)
    dominant_k = np.argmax(mean_power[1:half]) + 1
    return dominant_k, mean_power


def plot_fourier_projection(cfg, steps_to_plot=None, fixed_k=None):
    """
    Fourier-Projektion der Token-Embeddings auf cos/sin Basis.
    Projiziert auf cos(2πkx/p) und sin(2πkx/p) für die dominante
    Frequenz k → ergibt nach Grokking einen sauberen Kreis.

    Nanda et al. (2023): das Netz lernt Fourier-Features,
    d.h. die Embeddings kodieren Zahlen als Punkte auf einem Kreis
    mit Frequenz k. PCA zeigt das nicht, weil PCA Varianz maximiert,
    nicht Fourier-Struktur.

    Args:
        cfg:            Config-Objekt
        steps_to_plot:  Liste von Checkpoint-Steps
        fixed_k:        Feste Frequenz k verwenden (None = auto-detect)
    """
    if steps_to_plot is None:
        steps_to_plot = [1000, 3000, 7500, 15000]

    fig, axes = plt.subplots(1, len(steps_to_plot),
                             figsize=(5 * len(steps_to_plot), 5))
    fig.suptitle('Fourier-Projektion: Embeddings auf cos/sin Basis\n'
                 '(Nanda et al. 2023 — Kreis-Struktur nach Grokking)',
                 fontsize=13, fontweight='bold')

    x = np.arange(cfg.p)  # Token-Indizes 0..p-1

    for ax, step in zip(axes, steps_to_plot):
        model, t_acc, v_acc = load_checkpoint(step, cfg)
        emb = model.tok_emb.weight[:cfg.p].detach().numpy()  # [97, d_model]

        # Dominante Frequenz bestimmen
        if fixed_k is not None:
            k = fixed_k
        else:
            k, mean_power = find_dominant_frequency(emb, cfg.p)

        # Fourier-Basisvektoren
        cos_vec = np.cos(2 * np.pi * k * x / cfg.p)  # [97]
        sin_vec = np.sin(2 * np.pi * k * x / cfg.p)  # [97]

        # Fourier-Projektion (Nanda et al.):
        # 1. Fourier-Koeffizienten pro Embedding-Dimension berechnen:
        #    c_cos[d] = sum_x emb[x,d] * cos(2πkx/p)
        # 2. Projektion für jeden Token:
        #    coord_cos[t] = sum_d emb[t,d] * c_cos[d]
        W = emb.T  # [d_model, 97]
        c_cos = W @ cos_vec  # [d_model]
        c_sin = W @ sin_vec  # [d_model]

        coord_cos = emb @ c_cos  # [97]
        coord_sin = emb @ c_sin  # [97]

        # Normalisierung für bessere Visualisierung
        scale = max(np.abs(coord_cos).max(), np.abs(coord_sin).max())
        if scale > 0:
            coord_cos /= scale
            coord_sin /= scale

        # Plot: Kreis-Visualisierung
        sc = ax.scatter(coord_cos, coord_sin,
                        c=x, cmap='hsv', s=50, alpha=0.85,
                        edgecolors='white', linewidths=0.3)

        # Token-Werte annotieren (jeden 10.)
        for t in range(0, cfg.p, 10):
            ax.annotate(str(t), (coord_cos[t], coord_sin[t]),
                        fontsize=7, ha='center', va='bottom', alpha=0.7)

        # Referenz-Kreis einzeichnen
        r = np.sqrt(coord_cos**2 + coord_sin**2).mean()
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                '--', color='gray', linewidth=1, alpha=0.4)

        # Kreisförmigkeit messen: std(Radien) / mean(Radien)
        radii = np.sqrt(coord_cos**2 + coord_sin**2)
        circularity = 1.0 - (radii.std() / radii.mean()) if radii.mean() > 0 else 0.0

        ax.set_title(f'Step {step:,} (k={k})\n'
                     f'train={t_acc*100:.0f}%  val={v_acc*100:.0f}%\n'
                     f'Kreisförmigkeit={circularity:.2f}',
                     fontsize=10)
        ax.set_xlabel('cos-Projektion')
        ax.set_ylabel('sin-Projektion')
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('fourier_projection.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gespeichert: fourier_projection.png")


# ══════════════════════════════════════════════════════════════
# Schritt 3c: Dominante Frequenz über Zeit
# ══════════════════════════════════════════════════════════════

def plot_frequency_over_time(cfg):
    """
    Liniengraph: dominante Fourier-Frequenz k über Training-Steps,
    mit Val Acc als überlagerte zweite y-Achse.

    Zeigt direkt:
      - wann der Frequenzwechsel (z.B. k=45 → k=36) passiert
      - ob er gleichzeitig mit dem Grokking-Moment auftritt,
        davor oder danach
    """
    steps, dom_freqs, val_accs, train_accs = [], [], [], []

    for step in CHECKPOINT_STEPS:
        path = CHECKPOINT_DIR / f"step_{step:06d}.pt"
        if not path.exists():
            continue

        model, t_acc, v_acc = load_checkpoint(step, cfg)
        emb = model.tok_emb.weight[:cfg.p].detach().numpy()
        k, _ = find_dominant_frequency(emb, cfg.p)

        steps.append(step)
        dom_freqs.append(k)
        val_accs.append(v_acc)
        train_accs.append(t_acc)

    if not steps:
        print("Keine Checkpoints gefunden — überspringe.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.suptitle('Dominante Fourier-Frequenz & Val Accuracy über Training',
                 fontsize=13, fontweight='bold')

    # ── Linke y-Achse: dominante Frequenz ──────────────────
    color_freq = '#7c6af7'
    ax1.plot(steps, dom_freqs, 'o-', color=color_freq,
             linewidth=2.5, markersize=7, label='Dominante Frequenz k',
             zorder=3)
    ax1.set_xlabel('Training Step', fontsize=11)
    ax1.set_ylabel('Dominante Frequenz k', fontsize=11, color=color_freq)
    ax1.tick_params(axis='y', labelcolor=color_freq)
    ax1.set_ylim(0, cfg.p // 2)
    ax1.grid(True, alpha=0.2)

    # Frequenzwerte annotieren
    for s, k in zip(steps, dom_freqs):
        ax1.annotate(str(k), (s, k), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8,
                     color=color_freq, fontweight='bold')

    # ── Rechte y-Achse: Val Acc ────────────────────────────
    ax2 = ax1.twinx()
    color_val = '#e05a6a'
    color_train = '#f7a46a'
    ax2.plot(steps, [v * 100 for v in val_accs], 's--', color=color_val,
             linewidth=2, markersize=5, label='Val Acc', alpha=0.8)
    ax2.plot(steps, [t * 100 for t in train_accs], 's--', color=color_train,
             linewidth=1.5, markersize=4, label='Train Acc', alpha=0.6)
    ax2.set_ylabel('Accuracy (%)', fontsize=11, color=color_val)
    ax2.tick_params(axis='y', labelcolor=color_val)
    ax2.set_ylim(-2, 105)

    # Grokking-Schwelle markieren
    ax2.axhline(95, color=color_val, linewidth=1, linestyle=':',
                alpha=0.5, label='95% Schwelle')

    # Frequenzsprung-Region hervorheben (falls erkennbar)
    if len(dom_freqs) >= 2:
        for i in range(1, len(dom_freqs)):
            if dom_freqs[i] != dom_freqs[i - 1]:
                ax1.axvspan(steps[i - 1], steps[i],
                            alpha=0.15, color='#f7e76a',
                            label='Frequenzwechsel' if i == next(
                                j for j in range(1, len(dom_freqs))
                                if dom_freqs[j] != dom_freqs[j - 1]
                            ) else None)

    # Legenden zusammenführen
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='center right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('frequency_over_time.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gespeichert: frequency_over_time.png")

    # Tabellarische Zusammenfassung
    print(f"\n{'Step':>8}  {'k':>4}  {'Val Acc':>8}  {'Train Acc':>10}")
    print('─' * 35)
    for s, k, v, t in zip(steps, dom_freqs, val_accs, train_accs):
        print(f"{s:8d}  {k:4d}  {v*100:7.1f}%  {t*100:9.1f}%")


# ══════════════════════════════════════════════════════════════
# Schritt 4: Attention-Pattern
# ══════════════════════════════════════════════════════════════

def plot_attention_patterns(cfg, steps_to_plot=None):
    """
    Attention-Gewichte der =-Position (Index 2) über alle Heads.
    Zeigt welche Input-Tokens die Ausgabe beeinflusst.
    Erwartung nach Grokking: strukturiertes Muster, beide a und b beachtet.
    """
    if steps_to_plot is None:
        steps_to_plot = [1000, 3000, 7500, 15000]

    # Beispiel-Input: a=3, b=5, label=8
    a, b = 3, 5
    example_input = torch.tensor([[a, b, cfg.p]])  # [1, 3]

    fig, axes = plt.subplots(cfg.n_heads, len(steps_to_plot),
                             figsize=(4 * len(steps_to_plot), 3 * cfg.n_heads))
    fig.suptitle(f'Attention-Pattern: Input [{a}, {b}, =]  →  {(a+b)%cfg.p}',
                 fontsize=13, fontweight='bold')

    token_labels = [str(a), str(b), '=']

    for j, step in enumerate(steps_to_plot):
        model, t_acc, v_acc = load_checkpoint(step, cfg)

        # Attention-Gewichte extrahieren (per-head, nicht gemittelt)
        with torch.no_grad():
            B, T = example_input.shape
            pos = torch.arange(T, device=example_input.device).unsqueeze(0)
            h = model.tok_emb(example_input) + model.pos_emb(pos)
            _, w = model.attn(h, h, h,
                              need_weights=True,
                              average_attn_weights=False)
            # w shape: [B, n_heads, T, T]

        w = w[0]  # [n_heads, 3, 3]

        for i in range(cfg.n_heads):
            ax = axes[i, j] if cfg.n_heads > 1 else axes[j]
            im = ax.imshow(w[i].numpy(), cmap='Blues',
                           vmin=0, vmax=1, aspect='auto')
            ax.set_xticks([0, 1, 2]); ax.set_xticklabels(token_labels)
            ax.set_yticks([0, 1, 2]); ax.set_yticklabels(token_labels)

            for row in range(3):
                for col in range(3):
                    ax.text(col, row, f'{w[i, row, col]:.2f}',
                            ha='center', va='center', fontsize=8,
                            color='white' if w[i, row, col] > 0.6 else 'black')

            if i == 0:
                ax.set_title(f'Step {step:,}\nval={v_acc*100:.0f}%', fontsize=9)
            ax.set_ylabel(f'Head {i}', fontsize=8)

    plt.tight_layout()
    plt.savefig('attention_patterns.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gespeichert: attention_patterns.png")


# ══════════════════════════════════════════════════════════════
# Schritt 5: Übersichtsplot — Val Acc + Embedding-Struktur
# ══════════════════════════════════════════════════════════════

def plot_summary(cfg):
    """
    Kombinierter Plot: Val Acc über Zeit + PCA-Visualisierung
    bei drei Schlüssel-Zeitpunkten (pre / during / post grokking).
    """
    # Val Acc Kurve aus allen Checkpoints rekonstruieren
    steps, val_accs, train_accs = [], [], []
    for step in CHECKPOINT_STEPS:
        path = CHECKPOINT_DIR / f"step_{step:06d}.pt"
        if path.exists():
            ckpt = torch.load(path, map_location='cpu')
            steps.append(step)
            val_accs.append(ckpt['val_acc'])
            train_accs.append(ckpt['train_acc'])

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 4, width_ratios=[2, 1, 1, 1])

    # Accuracy-Kurve
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(steps, [a * 100 for a in train_accs],
             'o-', color='#e05a6a', linewidth=2, label='Train', markersize=5)
    ax0.plot(steps, [a * 100 for a in val_accs],
             'o-', color='#7c6af7', linewidth=2, label='Val', markersize=5)
    ax0.set_xlabel('Step', fontsize=11)
    ax0.set_ylabel('Accuracy (%)', fontsize=11)
    ax0.set_title('Training — Grokking', fontsize=11, fontweight='bold')
    ax0.legend(); ax0.grid(True, alpha=0.3)
    ax0.set_ylim(-2, 105)

    # PCA bei 3 Zeitpunkten
    key_steps = [1000, 5000, 15000]
    labels    = ['Pre-Grokking', 'During', 'Post-Grokking']
    colors_panel = ['#e05a6a', '#f7a46a', '#6af7c2']

    for idx, (step, label, col) in enumerate(zip(key_steps, labels, colors_panel)):
        ax = fig.add_subplot(gs[idx + 1])
        path = CHECKPOINT_DIR / f"step_{step:06d}.pt"
        if not path.exists():
            ax.text(0.5, 0.5, f'Step {step}\nnicht gefunden',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        model, t_acc, v_acc = load_checkpoint(step, cfg)
        emb   = model.tok_emb.weight[:cfg.p].detach().numpy()
        pca   = PCA(n_components=2)
        emb2d = pca.fit_transform(emb)

        ax.scatter(emb2d[:, 0], emb2d[:, 1],
                   c=np.arange(cfg.p), cmap='hsv', s=30, alpha=0.85)
        ax.set_title(f'{label}\nStep {step:,} | val={v_acc*100:.0f}%',
                     fontsize=9, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

        # Kreis einzeichnen falls post-grokking
        if v_acc > 0.9:
            r = np.sqrt((emb2d ** 2).sum(axis=1)).mean()
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(r * np.cos(theta), r * np.sin(theta),
                    '--', color='gray', linewidth=1, alpha=0.4)

    plt.tight_layout()
    plt.savefig('grokking_mechanistic_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gespeichert: grokking_mechanistic_summary.png")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Konfiguration: train_frac=0.20, wd=1.0 → Grokking bei ~2500 steps
    cfg = Config(
        num_steps=max(CHECKPOINT_STEPS),
    )
    print(f"Config: p={cfg.p}, train_frac={TRAIN_FRAC}, "
          f"weight_decay={WEIGHT_DECAY}")
    print()

    # 1. Training mit Checkpoints
    # (überspringen falls Checkpoints bereits existieren)
    missing = [s for s in CHECKPOINT_STEPS
               if not (CHECKPOINT_DIR / f"step_{s:06d}.pt").exists()]
    if missing:
        print(f"Fehlende Checkpoints: {missing} — starte Training...")
        train_with_checkpoints(cfg, TRAIN_FRAC, WEIGHT_DECAY)
    else:
        print("Alle Checkpoints vorhanden — überspringe Training.")

    print()

    # 2. Embedding-Geometrie
    print("Plotte Embedding-Geometrie...")
    plot_embedding_geometry(cfg, steps_to_plot=[1000, 3000, 7500, 15000])

    # 3. Fourier-Analyse
    print("Plotte Fourier-Analyse...")
    plot_fourier_analysis(cfg, steps_to_plot=[1000, 3000, 7500, 15000])

    # 3b. Fourier-Projektion (Kreis)
    print("Plotte Fourier-Projektion...")
    plot_fourier_projection(cfg, steps_to_plot=[1000, 3000, 7500, 15000])

    # 3c. Dominante Frequenz über Zeit
    print("Plotte Frequenz über Zeit...")
    plot_frequency_over_time(cfg)

    # 4. Attention-Pattern
    print("Plotte Attention-Pattern...")
    plot_attention_patterns(cfg, steps_to_plot=[1000, 3000, 7500, 15000])

    # 5. Übersichtsplot
    print("Plotte Zusammenfassung...")
    plot_summary(cfg)

    print("\nFertig. Ausgaben:")
    print("  embedding_geometry.png")
    print("  fourier_analysis.png")
    print("  fourier_projection.png")
    print("  frequency_over_time.png")
    print("  attention_patterns.png")
    print("  grokking_mechanistic_summary.png")