"""
Unsupervised segmentation for SAR oil slick candidate detection.

segment_felzenszwalb() receives:
  band (np.ndarray): 2D float32 array — single band from the GeoTIFF tile.
  cfg  (dict):       algorithm parameters from config.json.

It returns a boolean mask (same shape as band) where True = candidate oil
pixel (dark region). Oil slicks appear as dark patches in SAR imagery
(reduced backscatter due to dampened surface roughness), so the algorithm
targets low-intensity regions.

Chosen as the sole classical baseline after comparing all 7 originally
implemented candidates (CFAR, Otsu, GMM, SLIC+threshold, Felzenszwalb,
watershed, mean shift) against the ground-truth masks: Felzenszwalb had the
best IoU (0.276) with the only reasonably balanced precision/recall tradeoff
(0.42 / 0.44) — see results/iou_vs_ground_truth.csv.
"""
import numpy as np


def segment_felzenszwalb(band: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Felzenszwalb graph-based segmentation then dark-segment threshold.
    Segments can be large and irregular (controlled by scale). Segments whose
    mean intensity falls below dark_percentile of all segment means are flagged.
    """
    from skimage.segmentation import felzenszwalb

    scale    = float(cfg.get("scale", 100))
    sigma    = float(cfg.get("sigma", 0.5))
    min_size = int(cfg.get("min_size", 50))
    dark_pct = float(cfg.get("dark_percentile", 30))

    b_norm   = _normalize(band)
    segments = felzenszwalb(b_norm, scale=scale, sigma=sigma,
                             min_size=min_size, channel_axis=None)

    seg_ids   = np.unique(segments)
    seg_means = np.array([b_norm[segments == sid].mean() for sid in seg_ids])
    threshold = np.percentile(seg_means, dark_pct)

    dark_mask = np.zeros(band.shape, dtype=bool)
    for sid, mean_val in zip(seg_ids, seg_means):
        if mean_val < threshold:
            dark_mask[segments == sid] = True
    return dark_mask


# ─────────────────────────────────────────────────────────────────────────────
# Mask → GeoDataFrame conversion
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_geodataframe(binary_mask: np.ndarray, transform, crs,
                          min_pixels: int = 100):
    """
    Convert a boolean mask to a GeoDataFrame of georeferenced polygons.
    Polygons smaller than min_pixels are discarded.
    """
    import geopandas as gpd
    from shapely.geometry import shape
    from rasterio.features import shapes

    mask_u8   = binary_mask.astype(np.uint8)
    pixel_area = abs(transform.a * transform.e)
    min_area   = min_pixels * pixel_area

    polygons = [
        shape(geom)
        for geom, val in shapes(mask_u8, mask=mask_u8, transform=transform)
        if val == 1 and shape(geom).area >= min_area
    ]

    if not polygons:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    return gpd.GeoDataFrame(geometry=polygons, crs=crs).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(band: np.ndarray) -> np.ndarray:
    b = band.astype(np.float32)
    lo, hi = b.min(), b.max()
    return (b - lo) / (hi - lo) if hi > lo else np.zeros_like(b)


# Registry: name → function
ALGO_FUNCS = {
    "felzenszwalb": segment_felzenszwalb,
}
