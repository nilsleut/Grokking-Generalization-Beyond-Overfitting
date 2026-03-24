"""
Sparse Autoencoder (SAE) Analysis on Grokking Transformer
===========================================================
Basiert auf:
  - Elhage et al. (2022) "Toy Models of Superposition"
    https://transformer-circuits.pub/2022/toy_model/index.html
  - Bricken et al. (2023) "Towards Monosemanticity"
    https://transformer-circuits.pub/2023/monosemantic-features/index.html
  - Nanda et al. (2023) "Progress Measures for Grokking via Mechanistic Interpretability"
    arXiv:2301.05217

Zentrale Vorhersage:
  SAE-Features auf dem post-grokking Modell sollten den dominanten
  Fourier-Frequenzen entsprechen die in der mechanistischen Analyse
  identifiziert wurden — eine Feature-Richtung pro Frequenz k.

Experiment-Design:
  1. SAE auf MLP-Aktivierungen des trainierten Grokking-Transformers
  2. Vergleich pre-grokking vs post-grokking Checkpoints
  3. Fourier-Analyse der SAE-Features → Test der Vorhersage
  4. Feature-Aktivierungsmuster über alle (a,b) Eingaben

Voraussetzung:
  - grokking_correct.py im selben Ordner (GrokkingTransformer, Config)
  - checkpoints/ Ordner mit gespeicherten Checkpoints
    (aus grokking_mechanistic.py — Steps 1000, 3000, 7500, 15000, 20000)

Ausgaben:
  - sae_features_pre.png     — SAE-Features vor Grokking
  - sae_features_post.png    — SAE-Features nach Grokking
  - sae_fourier_match.png    — Fourier-Analyse der SAE-Features
  - sae_activation_grid.png  — Feature-Aktivierungen über (a,b) Raum
  - sae_comparison.png       — Übersichtsplot pre vs post
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from grokking_correct import GrokkingTransformer, Config


# ══════════════════════════════════════════════════════════════
# SAE Konfiguration
# ══════════════════════════════════════════════════════════════

@dataclass
class SAEConfig:
    # Architektur
    d_in: int = 128          # Input-Dimension = d_model des Transformers
    d_sae: int = 1024        # SAE hidden dim — 8x d_in für bessere Überparametrisierung
                             # Überbestimmt: mehr Features als Dimensionen

    # Training
    lr: float = 1e-3
    l1_coeff: float = 1e-3   # Sparsity-Regularisierung — kritischer Parameter
                             # zu hoch → alle Features tot (dead features)
                             # zu tief → keine Sparsität
    num_steps: int = 50_000  # Mehr Steps für bessere Konvergenz
    batch_size: int = 256
    seed: int = 42

    # Welche Aktivierung extrahieren
    # 'post_attn'  = nach Attention + LayerNorm
    # 'post_ff'    = nach Feed-Forward + LayerNorm (empfohlen)
    # 'pre_head'   = direkt vor dem Output-Head
    activation_site: str = 'post_ff'

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_STEPS = [1000, 3000, 7500, 15000, 20000]


# ══════════════════════════════════════════════════════════════
# Sparse Autoencoder
# ══════════════════════════════════════════════════════════════

class SparseAutoencoder(nn.Module):
    """
    Standard SAE nach Bricken et al. (2023):

      encode: h = ReLU(W_enc(x - b_pre) + b_enc)
      decode: x̂ = W_dec h + b_pre

    Loss = ||x - x̂||² + λ ||h||₁

    W_dec ist normiert (unit columns) um Feature-Richtungen
    interpretierbar zu halten.
    """
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder
        self.W_enc  = nn.Parameter(torch.empty(cfg.d_in, cfg.d_sae))
        self.b_enc  = nn.Parameter(torch.zeros(cfg.d_sae))

        # Decoder (columns werden auf Einheitslänge normiert)
        self.W_dec  = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in))
        self.b_pre  = nn.Parameter(torch.zeros(cfg.d_in))

        self._init_weights()

    def _init_weights(self):
        # Kaiming für Encoder
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity='relu')
        # Decoder: normierte zufällige Richtungen
        nn.init.kaiming_uniform_(self.W_dec)
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        """Normiert Decoder-Spalten auf Einheitslänge."""
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_in] → h: [B, d_sae]"""
        return F.relu((x - self.b_pre) @ self.W_enc + self.b_enc)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, d_sae] → x̂: [B, d_in]"""
        return h @ self.W_dec + self.b_pre

    def forward(self, x: torch.Tensor):
        h   = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

    def loss(self, x: torch.Tensor):
        x_hat, h = self(x)
        recon_loss   = (x - x_hat).pow(2).mean()
        sparsity_loss = self.cfg.l1_coeff * h.abs().mean()
        total_loss    = recon_loss + sparsity_loss
        return total_loss, recon_loss, sparsity_loss, h


# ══════════════════════════════════════════════════════════════
# Aktivierungen extrahieren
# ══════════════════════════════════════════════════════════════

def get_all_inputs(cfg: Config):
    """Alle (a, b) Paare — vollständiger Input-Raum."""
    pairs  = [(a, b) for a in range(cfg.p) for b in range(cfg.p)]
    inputs = torch.tensor([[a, b, cfg.p] for a, b in pairs], dtype=torch.long)
    labels = torch.tensor([(a + b) % cfg.p for a, b in pairs], dtype=torch.long)
    return inputs, labels, pairs


@torch.no_grad()
def extract_activations(model: GrokkingTransformer,
                        inputs: torch.Tensor,
                        site: str,
                        batch_size: int = 512) -> torch.Tensor:
    """
    Extrahiert Aktivierungen an einem bestimmten Punkt im Transformer.

    site='post_ff': Aktivierung nach Feed-Forward + LayerNorm
                    Shape: [N, d_model]
    """
    model.eval()
    all_acts = []

    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        B, T  = batch.shape
        pos   = torch.arange(T).unsqueeze(0)
        h     = model.tok_emb(batch) + model.pos_emb(pos)

        # Attention Block
        h2, _ = model.attn(h, h, h)
        h     = model.norm1(h + h2)

        if site == 'post_attn':
            # Aktivierung nach Attention + LayerNorm, letzte Position
            all_acts.append(h[:, -1, :].cpu())
            continue

        # Feed-Forward Block
        h = model.norm2(h + model.ff(h))

        if site == 'post_ff':
            all_acts.append(h[:, -1, :].cpu())

        elif site == 'pre_head':
            all_acts.append(h[:, -1, :].cpu())

    return torch.cat(all_acts, dim=0)  # [N, d_model]


def load_checkpoint(step: int, cfg: Config):
    path = CHECKPOINT_DIR / f"step_{step:06d}.pt"
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = GrokkingTransformer(cfg)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt.get('train_acc', None), ckpt.get('val_acc', None)


# ══════════════════════════════════════════════════════════════
# SAE Training
# ══════════════════════════════════════════════════════════════

def train_sae(activations: torch.Tensor,
              sae_cfg: SAEConfig,
              verbose: bool = True) -> SparseAutoencoder:
    """
    Trainiert einen SAE auf den gegebenen Aktivierungen.

    activations: [N, d_in] — alle extrahierten Aktivierungen
    """
    torch.manual_seed(sae_cfg.seed)
    sae = SparseAutoencoder(sae_cfg)
    optimizer = torch.optim.Adam(sae.parameters(), lr=sae_cfg.lr)

    # Aktivierungen normieren (wichtig für stabile SAE-Training)
    act_mean = activations.mean(dim=0, keepdim=True)
    act_std  = activations.std(dim=0, keepdim=True).clamp(min=1e-8)
    acts_norm = (activations - act_mean) / act_std

    dataset    = TensorDataset(acts_norm)
    loader     = DataLoader(dataset, sae_cfg.batch_size, shuffle=True)

    log_interval = sae_cfg.num_steps // 10

    def infinite(loader):
        while True:
            yield from loader

    step = 0
    for (batch,) in infinite(loader):
        if step >= sae_cfg.num_steps:
            break

        total, recon, sparse, h = sae.loss(batch)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        # Decoder nach jedem Step normieren
        sae._normalize_decoder()

        step += 1

        if verbose and step % log_interval == 0:
            dead = (h.abs().max(dim=0).values < 1e-6).sum().item()
            frac_dead = dead / sae_cfg.d_sae
            print(f"  step {step:6d} | loss {total.item():.4f} "
                  f"| recon {recon.item():.4f} "
                  f"| sparse {sparse.item():.4f} "
                  f"| dead features: {dead}/{sae_cfg.d_sae} "
                  f"({frac_dead*100:.1f}%)")

    # Normierungsfaktoren zurückgeben für spätere Analyse
    sae.act_mean = act_mean
    sae.act_std  = act_std

    return sae


# ══════════════════════════════════════════════════════════════
# Analyse: Fourier-Match
# ══════════════════════════════════════════════════════════════

def fourier_power_of_features(sae: SparseAutoencoder,
                               cfg: Config) -> np.ndarray:
    """
    Berechnet Fourier-Power der SAE-Feature-Richtungen.

    Jede Feature-Richtung w_i = W_dec[i] ist ein Vektor im d_model Raum.
    Wir projizieren auf Token-Embeddings und berechnen FFT über Token-Index.

    Return: [d_sae, p//2] — Fourier-Power pro Feature und Frequenz
    """
    # W_dec: [d_sae, d_in] — jede Zeile ist eine Feature-Richtung
    W_dec = sae.W_dec.detach().numpy()  # [d_sae, d_in]
    return W_dec  # wird in plot-Funktion weiterverarbeitet


@torch.no_grad()
def compute_feature_activations(sae: SparseAutoencoder,
                                 activations: torch.Tensor) -> np.ndarray:
    """
    Aktivierungen aller SAE-Features für alle (a,b) Eingaben.
    Return: [p*p, d_sae] — Feature-Aktivierungen
    """
    acts_norm = (activations - sae.act_mean) / sae.act_std
    h = sae.encode(acts_norm)
    return h.numpy()  # [p*p, d_sae]


# ══════════════════════════════════════════════════════════════
# Visualisierung
# ══════════════════════════════════════════════════════════════

def plot_sae_fourier_match(sae: SparseAutoencoder,
                           activations: torch.Tensor,
                           model: GrokkingTransformer,
                           cfg: Config,
                           step: int,
                           val_acc: float,
                           save_path: str):
    """
    Hauptplot: testet ob SAE-Features den Fourier-Frequenzen entsprechen.

    Panel 1: Fourier-Power der Token-Embeddings (Referenz)
    Panel 2: Fourier-Power der top SAE-Features
    Panel 3: Feature-Sparsität (L0-Norm)
    Panel 4: Rekonstruktionsfehler pro Token-Paar
    """
    # Token-Embedding Fourier (Referenz aus mechanistischer Analyse)
    emb        = model.tok_emb.weight[:cfg.p].detach().numpy()
    fft_emb    = np.abs(np.fft.fft(emb, axis=0)) ** 2
    mean_power_emb = fft_emb.mean(axis=1)
    half = cfg.p // 2

    # SAE Feature-Aktivierungen
    feature_acts = compute_feature_activations(sae, activations)  # [p*p, d_sae]

    # Top-k aktivste Features (nach mittlerer Aktivierung)
    mean_acts  = feature_acts.mean(axis=0)  # [d_sae]
    top_k      = 10
    top_idx    = np.argsort(mean_acts)[-top_k:]

    # Fourier-Analyse der top Features über (a, ?) für festes b
    # Für Feature i: wie aktiviert es sich als Funktion von a?
    feature_acts_2d = feature_acts.reshape(cfg.p, cfg.p, -1)  # [p, p, d_sae]
    # Mitteln über b-Dimension
    feature_acts_a  = feature_acts_2d.mean(axis=1)  # [p, d_sae]

    fft_features = np.abs(np.fft.fft(feature_acts_a, axis=0)) ** 2  # [p, d_sae]
    # Top-Feature Fourier-Power
    top_feature_power = fft_features[:, top_idx].mean(axis=1)  # [p]

    # L0-Sparsität: Anteil aktiver Features pro Input
    l0 = (feature_acts > 0).sum(axis=1)  # [p*p]

    # Rekonstruktionsfehler
    acts_norm = (activations - sae.act_mean) / sae.act_std
    with torch.no_grad():
        x_hat, _ = sae(acts_norm)
        recon_err = (acts_norm - x_hat).pow(2).mean(dim=1).numpy()
    recon_err_2d = recon_err.reshape(cfg.p, cfg.p)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'SAE Analyse — Step {step:,}  |  val={val_acc*100:.0f}%  |  '
                 f'd_sae={sae.cfg.d_sae}, λ={sae.cfg.l1_coeff}',
                 fontsize=13, fontweight='bold')

    # Panel 1: Embedding Fourier (Referenz)
    ax = axes[0, 0]
    ax.bar(np.arange(1, half), mean_power_emb[1:half],
           color='#7c6af7', alpha=0.8, width=0.8)
    top3_emb = np.argsort(mean_power_emb[1:half])[-3:] + 1
    for k in top3_emb:
        ax.axvline(k, color='#f76a8c', linewidth=2, linestyle='--', alpha=0.8)
        ax.text(k, mean_power_emb[k] * 1.05, f'k={k}',
                ha='center', fontsize=8, color='#f76a8c', fontweight='bold')
    ax.set_xlabel('Frequenz k'); ax.set_ylabel('Mittlere Fourier-Power')
    ax.set_title('Token-Embedding Fourier (Referenz)', fontweight='bold')
    ax.set_xlim(0, half)

    # Panel 2: Top-SAE-Feature Fourier
    ax = axes[0, 1]
    ax.bar(np.arange(1, half), top_feature_power[1:half],
           color='#6af7c2', alpha=0.8, width=0.8)
    top3_feat = np.argsort(top_feature_power[1:half])[-3:] + 1
    for k in top3_feat:
        ax.axvline(k, color='#f76a8c', linewidth=2, linestyle='--', alpha=0.8)
        ax.text(k, top_feature_power[k] * 1.05, f'k={k}',
                ha='center', fontsize=8, color='#f76a8c', fontweight='bold')
    # Referenz-Frequenzen aus Embeddings einzeichnen
    for k in top3_emb:
        ax.axvline(k, color='#7c6af7', linewidth=1, linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequenz k'); ax.set_ylabel('Mittlere Fourier-Power')
    ax.set_title(f'Top-{top_k} SAE-Feature Fourier\n'
                 f'(Lila = Referenzfrequenzen aus Embeddings)',
                 fontweight='bold')
    ax.set_xlim(0, half)

    # Panel 3: L0-Sparsität
    ax = axes[1, 0]
    ax.hist(l0, bins=30, color='#f7a46a', alpha=0.8, edgecolor='white')
    ax.axvline(l0.mean(), color='red', linewidth=2,
               label=f'Mean L0 = {l0.mean():.1f}')
    ax.set_xlabel(f'Aktive Features (von {sae.cfg.d_sae})')
    ax.set_ylabel('Anzahl Inputs')
    ax.set_title('Feature-Sparsität (L0)', fontweight='bold')
    ax.legend()

    # Panel 4: Rekonstruktionsfehler
    ax = axes[1, 1]
    im = ax.imshow(recon_err_2d, cmap='RdYlGn_r', aspect='auto')
    ax.set_xlabel('b'); ax.set_ylabel('a')
    ax.set_title('Rekonstruktionsfehler pro (a, b)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='MSE')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Gespeichert: {save_path}")

    return top3_emb, top3_feat


def plot_feature_activation_grid(sae: SparseAutoencoder,
                                  activations: torch.Tensor,
                                  cfg: Config,
                                  step: int,
                                  n_features: int = 6,
                                  save_path: str = 'sae_activation_grid.png'):
    """
    Zeigt für die top-n aktivsten SAE-Features wie sie über den (a,b) Raum aktivieren.
    Erwartung: Features die Fourier-Frequenzen entsprechen sollten
    sinusförmige Muster zeigen.
    """
    feature_acts = compute_feature_activations(sae, activations)
    feature_acts_2d = feature_acts.reshape(cfg.p, cfg.p, -1)

    mean_acts = feature_acts.mean(axis=0)
    top_idx   = np.argsort(mean_acts)[-n_features:][::-1]

    fig, axes = plt.subplots(2, n_features // 2, figsize=(4 * n_features // 2, 8))
    axes = axes.flatten()
    fig.suptitle(f'SAE Feature-Aktivierungen über (a,b) Raum — Step {step:,}',
                 fontsize=13, fontweight='bold')

    for plot_i, feat_i in enumerate(top_idx):
        ax    = axes[plot_i]
        grid  = feature_acts_2d[:, :, feat_i]
        im    = ax.imshow(grid, cmap='RdBu_r', aspect='auto',
                          vmin=-np.abs(grid).max(), vmax=np.abs(grid).max())
        ax.set_title(f'Feature {feat_i}\nmean={mean_acts[feat_i]:.3f}',
                     fontsize=9)
        ax.set_xlabel('b'); ax.set_ylabel('a')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Gespeichert: {save_path}")


def plot_pre_post_comparison(results: dict, cfg: Config,
                              save_path: str = 'sae_comparison.png'):
    """
    Vergleich pre-grokking vs post-grokking SAE-Features.
    Zeigt ob die Fourier-Struktur im SAE sichtbar wird.
    """
    steps    = sorted(results.keys())
    n_steps  = len(steps)
    half     = cfg.p // 2

    fig, axes = plt.subplots(2, n_steps, figsize=(5 * n_steps, 8))
    fig.suptitle('SAE Fourier-Features: pre vs post Grokking',
                 fontsize=13, fontweight='bold')

    for j, step in enumerate(steps):
        r        = results[step]
        val_acc  = r['val_acc']
        top_emb  = r['top_emb_freqs']
        top_feat = r['top_feat_freqs']
        power_emb  = r['power_emb']
        power_feat = r['power_feat']

        # Row 0: Embedding Fourier
        ax = axes[0, j]
        ax.bar(np.arange(1, half), power_emb[1:half],
               color='#7c6af7', alpha=0.8, width=0.8)
        for k in top_emb:
            ax.axvline(k, color='#f76a8c', linewidth=2,
                       linestyle='--', alpha=0.8)
        ax.set_title(f'Step {step:,} | val={val_acc*100:.0f}%\nEmbedding Fourier',
                     fontsize=9, fontweight='bold')
        ax.set_xlim(0, half)
        if j == 0:
            ax.set_ylabel('Embedding\nFourier Power', fontsize=10)

        # Row 1: SAE Feature Fourier
        ax = axes[1, j]
        ax.bar(np.arange(1, half), power_feat[1:half],
               color='#6af7c2', alpha=0.8, width=0.8)
        for k in top_feat:
            ax.axvline(k, color='#f76a8c', linewidth=2,
                       linestyle='--', alpha=0.8)
        for k in top_emb:
            ax.axvline(k, color='#7c6af7', linewidth=1,
                       linestyle=':', alpha=0.5)

        # Match-Score: Anteil der top-3 SAE-Frequenzen die mit Embedding übereinstimmen
        match = len(set(top_emb) & set(top_feat))
        ax.set_title(f'SAE Feature Fourier\nFrequenz-Match: {match}/3',
                     fontsize=9, fontweight='bold',
                     color='#4caf7d' if match >= 2 else '#e05a6a')
        ax.set_xlim(0, half)
        if j == 0:
            ax.set_ylabel('SAE Feature\nFourier Power', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Gespeichert: {save_path}")


# ══════════════════════════════════════════════════════════════
# L1-Koeffizient Sweep
# ══════════════════════════════════════════════════════════════

def sweep_l1(activations: torch.Tensor,
             cfg: Config,
             post_step: int = 15000,
             l1_values: list = None) -> dict:
    """
    Sweept über verschiedene L1-Koeffizienten um optimale Sparsität zu finden.
    Zu hoch → dead features. Zu niedrig → keine Sparsität.
    Gibt Rekonstruktionsfehler und Dead-Feature-Rate zurück.
    """
    if l1_values is None:
        # Erweiterter Sweep — λ=0.0001 war zu schwach (L0~500)
        # Ziel: L0 < 50 bei < 20% dead features
        l1_values = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1]

    results = {}
    print(f"\nL1-Sweep auf Step {post_step} Checkpoints:")
    print(f"{'l1':>8}  {'recon':>8}  {'sparse':>8}  {'dead%':>8}  {'mean_L0':>8}")
    print("─" * 50)

    sae_cfg = SAEConfig(num_steps=10_000)  # Kurzes Training für Sweep
    for l1 in l1_values:
        sae_cfg.l1_coeff = l1
        sae = train_sae(activations, sae_cfg, verbose=False)

        # Evaluierung
        acts_norm = (activations - sae.act_mean) / sae.act_std
        with torch.no_grad():
            x_hat, h = sae(acts_norm)
        recon  = (acts_norm - x_hat).pow(2).mean().item()
        sparse = h.abs().mean().item()
        dead   = (h.abs().max(dim=0).values < 1e-6).float().mean().item()
        l0     = (h > 0).float().sum(dim=1).mean().item()

        results[l1] = dict(recon=recon, sparse=sparse, dead=dead, l0=l0)
        print(f"{l1:8.1e}  {recon:8.4f}  {sparse:8.4f}  "
              f"{dead*100:7.1f}%  {l0:8.1f}")

    return results


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    grok_cfg = Config()
    sae_cfg  = SAEConfig()

    print("=" * 60)
    print("SAE Analyse auf Grokking Transformer")
    print("=" * 60)
    print(f"Transformer: p={grok_cfg.p}, d_model={grok_cfg.d_model}")
    print(f"SAE:         d_sae={sae_cfg.d_sae}, λ={sae_cfg.l1_coeff}")
    print(f"Site:        {sae_cfg.activation_site}")
    print()

    # Checkpoints prüfen
    available = [s for s in CHECKPOINT_STEPS
                 if (CHECKPOINT_DIR / f"step_{s:06d}.pt").exists()]
    if not available:
        print("FEHLER: Keine Checkpoints gefunden.")
        print(f"Bitte zuerst grokking_mechanistic.py ausführen.")
        exit(1)
    print(f"Verfügbare Checkpoints: {available}")

    # Inputs (vollständiger (a,b) Raum)
    inputs, labels, pairs = get_all_inputs(grok_cfg)
    print(f"Input-Raum: {len(inputs)} Paare ({grok_cfg.p}²)")

    # ── Schritt 1: L1-Sweep auf post-grokking Checkpoint ──
    post_step = max(available)
    print(f"\n[1/4] L1-Sweep auf Step {post_step}...")
    model_post, _, val_acc_post = load_checkpoint(post_step, grok_cfg)
    acts_post = extract_activations(model_post, inputs,
                                     sae_cfg.activation_site)
    sweep_results = sweep_l1(acts_post, grok_cfg, post_step)

    # Bestes L1: niedrigstes Recon bei L0 < 50 und < 30% dead features
    # L0 < 50 erzwingt echte Sparsität — ohne das sind SAE-Features nicht interpretierbar
    candidates = [l1 for l1, r in sweep_results.items()
                  if r['dead'] < 0.30 and r['l0'] < 50]
    if not candidates:
        # Fallback: niedrigstes L0 das noch lernbar ist
        candidates = [l1 for l1, r in sweep_results.items() if r['dead'] < 0.50]
    best_l1 = min(candidates,
                  key=lambda l1: sweep_results[l1]['recon'],
                  default=sae_cfg.l1_coeff)
    print(f"\nGewähltes L1: {best_l1}")
    sae_cfg.l1_coeff = best_l1

    # ── Schritt 2: SAE auf allen Checkpoints trainieren ──
    print(f"\n[2/4] SAE Training auf {len(available)} Checkpoints...")
    comparison_results = {}

    steps_to_analyze = [s for s in [1000, 7500, 15000, 20000]
                        if s in available]

    for step in steps_to_analyze:
        print(f"\n  Checkpoint Step {step}:")
        model, t_acc, v_acc = load_checkpoint(step, grok_cfg)
        acts = extract_activations(model, inputs, sae_cfg.activation_site)

        print(f"  Aktivierungen: {acts.shape}  "
              f"(mean={acts.mean():.3f}, std={acts.std():.3f})")

        sae = train_sae(acts, sae_cfg, verbose=True)

        # Fourier-Analyse für Vergleich speichern
        emb        = model.tok_emb.weight[:grok_cfg.p].detach().numpy()
        fft_emb    = np.abs(np.fft.fft(emb, axis=0)) ** 2
        power_emb  = fft_emb.mean(axis=1)
        half       = grok_cfg.p // 2
        top_emb    = list(np.argsort(power_emb[1:half])[-3:] + 1)

        feature_acts   = compute_feature_activations(sae, acts)
        feat_acts_2d   = feature_acts.reshape(grok_cfg.p, grok_cfg.p, -1)
        feat_acts_a    = feat_acts_2d.mean(axis=1)
        fft_feat       = np.abs(np.fft.fft(feat_acts_a, axis=0)) ** 2
        mean_acts_feat = feature_acts.mean(axis=0)
        top_k_idx      = np.argsort(mean_acts_feat)[-10:]
        power_feat     = fft_feat[:, top_k_idx].mean(axis=1)
        top_feat       = list(np.argsort(power_feat[1:half])[-3:] + 1)

        comparison_results[step] = dict(
            val_acc=v_acc or 0.0,
            top_emb_freqs=top_emb,
            top_feat_freqs=top_feat,
            power_emb=power_emb,
            power_feat=power_feat,
            sae=sae,
            acts=acts,
        )

        # Einzelplot
        label = 'pre' if (v_acc or 0) < 0.5 else 'post'
        plot_sae_fourier_match(
            sae, acts, model, grok_cfg, step, v_acc or 0.0,
            save_path=f'sae_features_{label}_step{step}.png'
        )

    # ── Schritt 3: Aktivierungsgitter für post-grokking ──
    print(f"\n[3/4] Feature-Aktivierungsgitter...")
    best_step = max(comparison_results.keys())
    r = comparison_results[best_step]
    plot_feature_activation_grid(
        r['sae'], r['acts'], grok_cfg, best_step,
        save_path='sae_activation_grid.png'
    )

    # ── Schritt 4: Vergleichsplot ──
    print(f"\n[4/4] Pre vs Post Vergleich...")
    plot_pre_post_comparison(
        comparison_results, grok_cfg,
        save_path='sae_comparison.png'
    )

    # ── Zusammenfassung ──
    print("\n" + "=" * 60)
    print("ERGEBNISSE")
    print("=" * 60)
    print(f"\n{'Step':>8}  {'Val Acc':>8}  {'Emb Freqs':>12}  "
          f"{'SAE Freqs':>12}  {'Match':>6}")
    print("─" * 55)
    for step, r in sorted(comparison_results.items()):
        match = len(set(r['top_emb_freqs']) & set(r['top_feat_freqs']))
        print(f"{step:8d}  {r['val_acc']*100:7.1f}%  "
              f"{str(r['top_emb_freqs']):>12}  "
              f"{str(r['top_feat_freqs']):>12}  "
              f"{match}/3")

    print("\nAusgaben:")
    print("  sae_features_pre_step*.png   — SAE pre-grokking")
    print("  sae_features_post_step*.png  — SAE post-grokking")
    print("  sae_activation_grid.png      — Feature-Aktivierungen")
    print("  sae_comparison.png           — Übersichtsplot")