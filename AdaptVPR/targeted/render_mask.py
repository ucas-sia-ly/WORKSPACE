"""Bounded Euclidean dilation for edge/shadow/blend allowance, never coverage."""

import math
import numpy as np
from scipy import ndimage


def build_render_mask(core_mask, settings):
    core = np.asarray(core_mask)
    if core.ndim != 2 or core.dtype != np.bool_ or not all(core.shape):
        raise ValueError("Core must be a nonempty-sized 2D bool array")
    fraction, growth, cap = (settings[key] for key in
                             ("radius_fraction_of_short_side", "max_added_area_ratio", "max_image_area_fraction"))
    maximum = settings["max_radius_pixels"]
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (fraction, growth, cap))
            or not 0 <= fraction <= .05 or not 0 <= growth <= 1 or not 0 < cap <= 1
            or type(maximum) is not int or not 0 <= maximum <= 64):
        raise ValueError("Invalid bounded render expansion settings")
    requested = min(maximum, round(min(core.shape) * fraction))
    area = int(core.sum())
    diagnostic = dict(method="euclidean_dilation_integer_radius", requested_radius_pixels=requested,
                      effective_radius_pixels=0, core_area=area, render_area=area, added_area=0,
                      max_added_area_ratio=growth, max_image_area_fraction=cap,
                      core_contained=True, usage="edges/shadows/blending only; excluded from all coverage gates")
    if not area or area > math.floor(core.size * cap):
        diagnostic.update(status="empty_core" if not area else "core_exceeds_render_area_cap", feasible=False)
        return core.copy(), diagnostic
    ys, xs = np.nonzero(core)
    y0, y1 = max(0, int(ys.min())-requested), min(core.shape[0], int(ys.max())+requested+1)
    x0, x1 = max(0, int(xs.min())-requested), min(core.shape[1], int(xs.max())+requested+1)
    distance = ndimage.distance_transform_edt(~core[y0:y1, x0:x1])
    limit = min(area + math.floor(area * growth), math.floor(core.size * cap))
    for radius in range(requested, -1, -1):
        expanded = distance <= radius
        if int(expanded.sum()) <= limit:
            render = np.zeros_like(core)
            render[y0:y1, x0:x1] = expanded
            if np.any(core & ~render):
                raise RuntimeError("Render expansion lost core pixels")
            total = int(render.sum())
            diagnostic.update(status="bounded" if radius < requested else "expanded", feasible=True,
                              effective_radius_pixels=radius, render_area=total, added_area=total-area,
                              added_area_ratio=(total-area)/area)
            return render, diagnostic
    raise RuntimeError("Zero-radius render must satisfy the core area bound")
