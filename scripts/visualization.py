import geemap
import ee
from IPython.display import display

def visualize_map(layers):
    """
    Vytvoření interaktivní mapy.
    :param layers: Seznam tuple vrstvy a jejich vizualizačních parametrů.
    """
    Map = geemap.Map(basemap="Esri.WorldImagery")
    Map.add_basemap("Esri.WorldTopoMap")
    
    for layer in layers:
        image, vis_params, name = layer
        if isinstance(image, ee.Image):  # Ověření, že je image správný typ
            Map.addLayer(image, vis_params, name)
        else:
            print(f"⚠ Varování: Vrstva '{name}' není typu ee.Image a nebude zobrazena!")
    
    return Map

def vis_2sigma(image, band, region, scale, k=2.0, palette=None, clamp_to_pct=None):
    """
    Build visualization params for Map.addLayer() using a μ ± k·σ stretch.

    Parameters
    ----------
    image : ee.Image
        Source image.
    band : str
        Band name to visualize.
    region : ee.Geometry
        Region for statistics (masked pixels are ignored).
    scale : float
        Pixel scale (meters) for reduceRegion.
    k : float, optional
        Sigma multiplier; default 2.0.
    palette : list[str] | None
        Optional color palette for visualization.
    clamp_to_pct : tuple[int, int] | None
        Optional percentile clamp (e.g., (2, 98)) to make the stretch more robust.

    Returns
    -------
    dict
        Dictionary with 'bands', 'min', 'max' (and 'palette' if provided) for Map.addLayer().
    """
    img = image.select([band])

    # Mean and standard deviation over the region (masked pixels are ignored)
    stats = img.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=region, scale=scale, bestEffort=True, maxPixels=1e13, tileScale=4
    )
    mu  = ee.Number(stats.get(f"{band}_mean"))
    sig = ee.Number(stats.get(f"{band}_stdDev"))

    vmin = mu.subtract(sig.multiply(k))
    vmax = mu.add(sig.multiply(k))

    # Optionally clamp by percentiles for a more robust stretch
    if clamp_to_pct is not None:
        lo, hi = clamp_to_pct
        p = img.reduceRegion(
            reducer=ee.Reducer.percentile([lo, hi]),
            geometry=region, scale=scale, bestEffort=True, maxPixels=1e13, tileScale=4
        )
        pmin = ee.Number(p.get(f"{band}_p{lo}"))
        pmax = ee.Number(p.get(f"{band}_p{hi}"))
        vmin = vmin.max(pmin)
        vmax = vmax.min(pmax)

    params = {"bands": [band], "min": vmin.getInfo(), "max": vmax.getInfo()}
    if palette:
        params["palette"] = palette
    return params

def vis_2sigma_safe(image, band, region, scale, k=2.0, palette=None, clamp_to_pct=None):
    """
    Build visParams with min/max = μ ± k·σ.
    Robust to null stats: falls back to percentiles if mean/stdDev are missing or degenerate.
    """
    img = image.select([band])

    # Count valid pixels to catch empty/masked regions early
    n_valid = ee.Number(
        img.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region, scale=scale, bestEffort=True, maxPixels=1e13, tileScale=4
        ).get(band, 0)
    )

    # Compute mean+std; use defaults to avoid nulls
    stats = ee.Dictionary(
        img.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=region, scale=scale, bestEffort=True, maxPixels=1e13, tileScale=4
        )
    )
    mu  = ee.Number(stats.get(f"{band}_mean", 0))
    sig = ee.Number(stats.get(f"{band}_stdDev", 0))

    # Proposed min/max from μ ± k·σ
    vmin_sigma = mu.subtract(sig.multiply(k))
    vmax_sigma = mu.add(sig.multiply(k))

    # Fallback percentiles (robust)
    lo, hi = (2, 98) if clamp_to_pct is None else clamp_to_pct
    p = ee.Dictionary(
        img.reduceRegion(
            reducer=ee.Reducer.percentile([lo, hi]),
            geometry=region, scale=scale, bestEffort=True, maxPixels=1e13, tileScale=4
        )
    )
    pmin = ee.Number(p.get(f"{band}_p{lo}", 0))
    pmax = ee.Number(p.get(f"{band}_p{hi}", 1))

    # Use sigma-stretch only if we have data and a non-degenerate stdDev range
    use_sigma = n_valid.gt(0).And(vmax_sigma.neq(vmin_sigma)).And(sig.gt(0))

    vmin = ee.Number(ee.Algorithms.If(use_sigma, vmin_sigma, pmin))
    vmax = ee.Number(ee.Algorithms.If(use_sigma, vmax_sigma, pmax))

    # Bring to client (numbers, not ee.Number)
    params = {"bands": [band], "min": vmin.getInfo(), "max": vmax.getInfo()}
    if palette:
        params["palette"] = palette
    return params

