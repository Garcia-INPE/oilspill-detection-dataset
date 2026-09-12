"""
Standalone feature extraction for the classical detection pipeline.

Ported from:
  src/lib/FunGetFeat_Geom.py  — geometric features
  src/lib/FunGetFeat_Stat.py  — radiometric/statistical features
  src/lib/Functions.py        — shared utilities (get_utm_zone, get_epsg_from_latlon,
                                get_masked_array_from_vector)

src/Config.py and src/lib/FunPlot.py are NOT imported here: Config.py reads a
hardcoded shapefile at import time, making direct import non-portable.
The feature math is identical to the original; only the import chain is removed.
"""
import math
import warnings

import cv2
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import from_bounds
from scipy.stats import ks_2samp, mannwhitneyu
from shapely import geometry

# ─────────────────────────────────────────────────────────────────────────────
# Utilities (ported from src/lib/Functions.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_utm_zone(latitude: float, longitude: float) -> str:
    """Return UTM zone string (e.g. '15Q') for a lat/lon coordinate."""
    import utm as _utm
    _, _, zone_number, zone_letter = _utm.from_latlon(latitude, longitude)
    return f"{zone_number}{zone_letter.upper()}"


def get_epsg_from_latlon(lat: float, lon: float) -> tuple:
    """Return (utm_band_str, epsg_code_str) for the UTM zone covering lat/lon."""
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1).zfill(2)
    epsg_code = "326" + utm_band if lat >= 0 else "327" + utm_band
    return utm_band, epsg_code


def get_masked_array_from_vector(raster, vectors, filled=False, crop=True, invert=False):
    """Clip a rasterio dataset to vector geometries and return a masked array."""
    with warnings.catch_warnings():
        # rasterio internally triggers numpy 2.5 .shape deprecation
        warnings.simplefilter("ignore", DeprecationWarning)
        masked_array, transform = rasterio_mask(
            raster, vectors, filled=filled, crop=crop, invert=invert, nodata=np.nan)
    return masked_array, transform


# ─────────────────────────────────────────────────────────────────────────────
# Geometric features (ported from src/lib/FunGetFeat_Geom.py)
# ─────────────────────────────────────────────────────────────────────────────

_AREA_FACTOR = 1_000_000   # m² → km²
_LEN_FACTOR  = 1_000       # m  → km


_HU_RASTER_LONG_SIDE = 256


def _get_hu_moments(pol_gdf: gpd.GeoDataFrame) -> list:
    """
    Rasterize the polygon to a binary mask and compute Hu moments directly.
    Hu moments are translation/scale/rotation invariant, so no plotting
    library or disk round-trip is needed. Rasterizing per polygon via
    matplotlib (render -> save JPEG -> re-read) was the actual source of a
    multi-GB memory leak during batch runs: pyplot figures form reference
    cycles that outpace Python's cyclic GC across hundreds of polygons/tile.
    """
    minx, miny, maxx, maxy = pol_gdf.total_bounds
    width_m  = max(maxx - minx, 1e-9)
    height_m = max(maxy - miny, 1e-9)

    if width_m >= height_m:
        px_w = _HU_RASTER_LONG_SIDE
        px_h = max(int(round(_HU_RASTER_LONG_SIDE * height_m / width_m)), 1)
    else:
        px_h = _HU_RASTER_LONG_SIDE
        px_w = max(int(round(_HU_RASTER_LONG_SIDE * width_m / height_m)), 1)

    transform = from_bounds(minx, miny, maxx, maxy, px_w, px_h)
    mask = rasterize(
        [(geom, 255) for geom in pol_gdf.geometry],
        out_shape=(px_h, px_w),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments)
    result = []
    for i in range(7):
        val = float(hu[i][0])
        if val != 0:
            result.append(-1.0 * math.copysign(1.0, val) * math.log10(abs(val)))
        else:
            result.append(0.0)
    return result


def get_feat_geom(gdf_poly: gpd.GeoDataFrame, dict_ret: dict) -> dict:
    """
    Compute geometric features for a single-polygon GeoDataFrame.

    Populates dict_ret with:
      CENTR_KM_LAT, CENTR_KM_LON, UTM_ZONE, AREA_KM2, PERIM_KM,
      COMPLEX_MEAS, SPREAD, SHP_FACT, HU_MOM1..7, CIRCULARITY,
      PERI_AREA_RATIO, AREA_KM_DS, PERIM_KM_DS
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    centr_lat = float(gdf_poly.centroid.y.iloc[0])
    centr_lon = float(gdf_poly.centroid.x.iloc[0])
    warnings.filterwarnings("default", category=UserWarning)

    utm_zone = get_utm_zone(centr_lat, centr_lon)
    _, crs_epsg = get_epsg_from_latlon(centr_lat, centr_lon)

    pol_m = gdf_poly.to_crs(crs_epsg)
    area_km2  = pol_m.area.iloc[0]   / _AREA_FACTOR
    perim_km  = pol_m.length.iloc[0] / _LEN_FACTOR

    complexity = (perim_km ** 2) / area_km2 if area_km2 > 0 else 0.0
    minx, miny, maxx, maxy = pol_m.total_bounds / _LEN_FACTOR
    spread     = (maxy - miny) / (maxx - minx) if (maxx - minx) > 0 else 0.0
    shp_fact   = (perim_km ** 2) / (4 * np.pi * area_km2) if area_km2 > 0 else 0.0
    circularity = (4 * np.pi * area_km2) / (perim_km ** 2) if perim_km > 0 else 0.0
    peri_area  = perim_km / area_km2 if area_km2 > 0 else 0.0

    hu = _get_hu_moments(pol_m)

    dict_ret.update({
        "CENTR_KM_LAT":    centr_lat,
        "CENTR_KM_LON":    centr_lon,
        "UTM_ZONE":        utm_zone,
        "AREA_KM2":        area_km2,
        "PERIM_KM":        perim_km,
        "COMPLEX_MEAS":    complexity,
        "SPREAD":          spread,
        "SHP_FACT":        shp_fact,
        "HU_MOM1": hu[0], "HU_MOM2": hu[1], "HU_MOM3": hu[2],
        "HU_MOM4": hu[3], "HU_MOM5": hu[4], "HU_MOM6": hu[5], "HU_MOM7": hu[6],
        "CIRCULARITY":     circularity,
        "PERI_AREA_RATIO": peri_area,
        # For detected polygons, DS values equal computed values (no source shapefile)
        "AREA_KM_DS":  area_km2,
        "PERIM_KM_DS": perim_km,
    })
    return dict_ret


# ─────────────────────────────────────────────────────────────────────────────
# Radiometric/statistical features (ported from src/lib/FunGetFeat_Stat.py)
# ─────────────────────────────────────────────────────────────────────────────

def _power_mean_ratio(dist1: np.ndarray, dist2: np.ndarray, p: float) -> float:
    """Ratio of power means of two distributions (matches src/FunGetFeat_Stat.power_mean_ratio)."""
    arr1 = np.asarray(dist1, dtype=np.float64)
    arr2 = np.asarray(dist2, dtype=np.float64)
    try:
        if abs(p) < 1e-9:
            mean1 = np.exp(np.mean(np.log(np.abs(arr1) + 1e-12)))
            mean2 = np.exp(np.mean(np.log(np.abs(arr2) + 1e-12)))
        else:
            mean1 = float(np.mean(arr1 ** p) ** (1.0 / p))
            mean2 = float(np.mean(arr2 ** p) ** (1.0 / p))
        return float(mean1 / mean2) if mean2 != 0 else float("inf")
    except Exception:
        return float("nan")


def get_feat_stat(gdf_all_polys: gpd.GeoDataFrame,
                  gdf_poly: gpd.GeoDataFrame,
                  dict_ret: dict,
                  tiff_file) -> dict:
    """
    Compute radiometric and statistical features for one polygon vs. its background.

    gdf_all_polys: all detected candidate polygons in this tile (used to define
                   the background region within the polygon's bounding box).
    gdf_poly:      the single polygon being measured.
    tiff_file:     open rasterio dataset for the tile.

    Populates dict_ret with FG_*, BG_*, FG_BG_* fields.
    """
    # ── Foreground (inside the polygon) ──────────────────────────────────────
    masked_fg_raw, _ = get_masked_array_from_vector(
        tiff_file, gdf_poly.geometry, filled=False, crop=True, invert=False)

    if masked_fg_raw.mask.all():
        return dict_ret

    dict_ret["FG_STD"]     = float(np.ma.std(masked_fg_raw))
    dict_ret["FG_VAR"]     = float(np.ma.var(masked_fg_raw))
    dict_ret["FG_MIN"]     = float(np.ma.min(masked_fg_raw))
    dict_ret["FG_MAX"]     = float(np.ma.max(masked_fg_raw))
    dict_ret["FG_MEAN"]    = float(np.ma.mean(masked_fg_raw))
    dict_ret["FG_MEDIAN"]  = float(np.ma.median(masked_fg_raw))
    dict_ret["FG_VAR_COEF"] = (
        dict_ret["FG_STD"] / dict_ret["FG_MEAN"]
        if dict_ret["FG_MEAN"] != 0 else 0.0)

    # ── Background (bbox area outside all detected polygons) ─────────────────
    bbox_pol = geometry.box(*gdf_poly.total_bounds)
    gdf_bbox = gpd.GeoDataFrame(
        gpd.GeoSeries(bbox_pol), columns=["geometry"], crs=tiff_file.crs)

    warnings.filterwarnings("error")
    try:
        gdf_in_bbox = gpd.overlay(
            gdf_all_polys, gdf_bbox, how="intersection",
            keep_geom_type=True, make_valid=True)
    except Exception:
        warnings.resetwarnings()
        gdf_in_bbox = gpd.overlay(
            gdf_all_polys, gdf_bbox, how="intersection",
            keep_geom_type=True, make_valid=True)
    warnings.resetwarnings()

    if len(gdf_in_bbox) == 0:
        return dict_ret

    masked_bg_raw, _ = get_masked_array_from_vector(
        tiff_file, gdf_in_bbox.geometry, filled=False, crop=True, invert=False)
    # Invert: keep pixels OUTSIDE the polygons (the sea background)
    masked_bg_raw.mask = ~masked_bg_raw.mask

    if masked_bg_raw.mask.all():
        return dict_ret

    dict_ret["BG_STD"]     = float(np.ma.std(masked_bg_raw))
    dict_ret["BG_VAR"]     = float(np.ma.var(masked_bg_raw))
    dict_ret["BG_MIN"]     = float(np.ma.min(masked_bg_raw))
    dict_ret["BG_MAX"]     = float(np.ma.max(masked_bg_raw))
    dict_ret["BG_MEAN"]    = float(np.ma.mean(masked_bg_raw))
    dict_ret["BG_MEDIAN"]  = float(np.ma.median(masked_bg_raw))
    dict_ret["BG_VAR_COEF"] = (
        dict_ret["BG_STD"] / dict_ret["BG_MEAN"]
        if dict_ret["BG_MEAN"] != 0 else 0.0)
    dict_ret["FG_DARK_INTENS"] = dict_ret["BG_MEAN"] - dict_ret["FG_MEAN"]

    data_fg = masked_fg_raw.compressed()
    data_bg = masked_bg_raw.compressed()

    stat_ks, pval_ks = ks_2samp(data_fg, data_bg)
    dict_ret["FG_BG_KS_STAT"] = float(stat_ks)
    dict_ret["FG_BG_KS_RES"]  = "DIFF" if pval_ks < 0.05 else "CAN_BE_SAME"

    stat_mw, pval_mw = mannwhitneyu(data_fg, data_bg, alternative="two-sided")
    dict_ret["FG_BG_MW_STAT"] = float(stat_mw)
    dict_ret["FG_BG_MW_RES"]  = "DIFF" if pval_mw < 0.05 else "CAN_BE_SAME"

    dict_ret["FG_BG_RAT_ARI"] = _power_mean_ratio(data_fg, data_bg, 1)
    dict_ret["FG_BG_RAT_QUA"] = _power_mean_ratio(data_fg, data_bg, 2)

    return dict_ret
