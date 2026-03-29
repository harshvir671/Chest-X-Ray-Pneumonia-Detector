from __future__ import annotations

import html

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def create_lung_overlay(
    image: Image.Image,
    label: str,
    confidence: float,
    heatmap: np.ndarray | None,
) -> Image.Image:
    base = ImageOps.contain(image.convert("RGB"), (900, 1100), method=Image.Resampling.LANCZOS)
    width, height = base.size
    lung_mask = _lung_mask((width, height))
    lung_alpha = np.asarray(lung_mask, dtype=np.float32) / 255.0
    base_array = np.asarray(base, dtype=np.float32)

    if label == "Normal":
        tint = np.zeros_like(base_array)
        tint[..., 0] = 74
        tint[..., 1] = 197
        tint[..., 2] = 165
        alpha = lung_alpha[..., None] * (0.16 + 0.12 * confidence)
        overlay = base_array * (1.0 - alpha) + tint * alpha
        outlined = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        return _draw_lung_outline(outlined, "#1B8F72")

    heat = _resize_heatmap(heatmap, base.size) if heatmap is not None else _fallback_intensity_map(base)
    heat = np.clip(heat, 0.0, 1.0) ** 0.8
    activity = np.clip((0.2 + heat * 0.8) * lung_alpha, 0.0, 1.0)
    colorized = _warm_heatmap(heat)
    alpha = activity[..., None] * (0.18 + 0.42 * confidence)
    overlay = base_array * (1.0 - alpha) + colorized * alpha

    cool_tint = np.zeros_like(base_array)
    cool_tint[..., 0] = 38
    cool_tint[..., 1] = 97
    cool_tint[..., 2] = 111
    cool_alpha = lung_alpha[..., None] * 0.06
    overlay = overlay * (1.0 - cool_alpha) + cool_tint * cool_alpha

    outlined = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    return _draw_lung_outline(outlined, "#C44536")


def render_lung_status_card(label: str, confidence: float, pneumonia_probability: float) -> str:
    severity = max(0.18, min(0.95, pneumonia_probability if label == "Pneumonia" else confidence))
    fill_left = "url(#healthyLung)" if label == "Normal" else "url(#pneumonicLung)"
    fill_right = fill_left
    glow_color = "#73D3AF" if label == "Normal" else "#E76F51"
    accent = "#1B8F72" if label == "Normal" else "#C44536"
    subtitle = "Lung fields look calm and clean." if label == "Normal" else "Inflammatory pattern detected by the model."

    if label == "Pneumonia":
        flare_markup = f"""
        <circle cx="118" cy="208" r="{18 + severity * 16:.1f}" fill="#F4A261" opacity="{0.18 + severity * 0.18:.2f}" />
        <circle cx="242" cy="190" r="{20 + severity * 18:.1f}" fill="#D1495B" opacity="{0.20 + severity * 0.24:.2f}" />
        <circle cx="222" cy="246" r="{10 + severity * 12:.1f}" fill="#F77F00" opacity="{0.16 + severity * 0.16:.2f}" />
        """
    else:
        flare_markup = """
        <circle cx="110" cy="192" r="10" fill="#EAFBF4" opacity="0.75" />
        <circle cx="250" cy="210" r="8" fill="#DFF8EE" opacity="0.7" />
        <circle cx="184" cy="132" r="5" fill="#FFFFFF" opacity="0.75" />
        """

    return f"""
    <div class="lung-hero-card">
      <svg viewBox="0 0 360 430" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{html.escape(label)} lung illustration">
        <defs>
          <linearGradient id="bgGlow" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#FFF8EE" />
            <stop offset="100%" stop-color="#F4EBDD" />
          </linearGradient>
          <linearGradient id="healthyLung" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#98E9C3" />
            <stop offset="100%" stop-color="#1B8F72" />
          </linearGradient>
          <linearGradient id="pneumonicLung" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#F6BD60" />
            <stop offset="100%" stop-color="#C44536" />
          </linearGradient>
          <filter id="softGlow">
            <feGaussianBlur stdDeviation="12" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <rect x="8" y="8" width="344" height="414" rx="28" fill="url(#bgGlow)" />
        <circle cx="180" cy="100" r="58" fill="{glow_color}" opacity="0.10" filter="url(#softGlow)" />
        <rect x="164" y="55" width="32" height="78" rx="16" fill="#C7D3D4" />
        <rect x="126" y="128" width="108" height="18" rx="9" fill="#C7D3D4" />
        <path d="M118 150 C74 158, 52 198, 64 255 C72 316, 108 344, 154 334 C173 329, 174 312, 173 268 L173 166 C173 156, 153 146, 118 150 Z" fill="{fill_left}" stroke="{accent}" stroke-width="4" />
        <path d="M242 150 C286 158, 308 198, 296 255 C288 316, 252 344, 206 334 C187 329, 186 312, 187 268 L187 166 C187 156, 207 146, 242 150 Z" fill="{fill_right}" stroke="{accent}" stroke-width="4" />
        <path d="M180 142 L180 308" stroke="#F8F4EC" stroke-width="12" stroke-linecap="round" opacity="0.8" />
        {flare_markup}
        <text x="28" y="376" fill="#16302B" font-size="28" font-family="Georgia, serif" font-weight="700">{html.escape(label)} Lungs</text>
        <text x="28" y="404" fill="#4F5D5F" font-size="15" font-family="Arial, sans-serif">{html.escape(subtitle)}</text>
      </svg>
    </div>
    """


def _lung_mask(size: tuple[int, int]) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    draw.ellipse(
        (
            int(width * 0.12),
            int(height * 0.12),
            int(width * 0.47),
            int(height * 0.90),
        ),
        fill=230,
    )
    draw.ellipse(
        (
            int(width * 0.53),
            int(height * 0.12),
            int(width * 0.88),
            int(height * 0.90),
        ),
        fill=230,
    )
    draw.rounded_rectangle(
        (
            int(width * 0.45),
            int(height * 0.08),
            int(width * 0.55),
            int(height * 0.95),
        ),
        radius=int(width * 0.03),
        fill=0,
    )

    return mask.filter(ImageFilter.GaussianBlur(radius=max(10, width // 55)))


def _draw_lung_outline(image: Image.Image, color: str) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    line_width = max(4, width // 180)

    draw.ellipse(
        (
            int(width * 0.12),
            int(height * 0.12),
            int(width * 0.47),
            int(height * 0.90),
        ),
        outline=color,
        width=line_width,
    )
    draw.ellipse(
        (
            int(width * 0.53),
            int(height * 0.12),
            int(width * 0.88),
            int(height * 0.90),
        ),
        outline=color,
        width=line_width,
    )
    return output


def _resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    heat = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255))
    heat = heat.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(heat, dtype=np.float32) / 255.0


def _fallback_intensity_map(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    baseline = array.mean()
    return np.clip((array - baseline) * 2.2 + 0.35, 0.0, 1.0)


def _warm_heatmap(heat: np.ndarray) -> np.ndarray:
    red = np.clip(110 + heat * 145, 0, 255)
    green = np.clip(60 + heat * 120, 0, 255)
    blue = np.clip(15 + (1.0 - heat) * 55, 0, 255)
    return np.stack([red, green, blue], axis=-1)
