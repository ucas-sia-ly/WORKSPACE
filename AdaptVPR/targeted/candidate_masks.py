"""Deterministic family footprints and token-mass coverage. No VLM or diffusion."""

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from scipy import ndimage

from .family_constraints import FAMILY_SHAPES, config_digest, validate_constraints
from .render_mask import build_render_mask


def components(mask):
    return int(ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))[1])


class CoverageContext:
    """Each token retains its original mass, irrespective of native pixel cell area."""
    def __init__(self, roi, weights, image_size):
        self.roi = np.asarray(roi)
        self.weights = np.asarray(weights)
        if self.roi.shape != (16,16) or self.roi.dtype != np.bool_ or not self.roi.any():
            raise ValueError("ROI must be a nonempty bool 16x16 token mask")
        if (self.weights.shape != self.roi.shape or self.weights.dtype.kind != "f"
                or not np.isfinite(self.weights).all() or (self.weights < 0).any()):
            raise ValueError("Weights must be finite nonnegative floating 16x16 token values")
        if len(image_size) != 2 or any(type(v) is not int or v < 16 for v in image_size):
            raise ValueError("image_size must be original (height,width), each >=16")
        self.shape = tuple(image_size)
        h, w = self.shape
        yy, xx = np.arange(h)*16//h, np.arange(w)*16//w
        self.token_ids = yy[:,None]*16 + xx[None,:]
        self.cell_areas = np.bincount(self.token_ids.ravel(), minlength=256).reshape(16,16)
        self.roi_pixels = self.roi[yy[:,None], xx[None,:]]
        self.roi_area = int(self.roi_pixels.sum())
        ry, rx = np.nonzero(self.roi_pixels)
        self.centroid = (float(rx.mean()+.5), float(ry.mean()+.5))
        self.total_weight = float(self.weights.sum(dtype=np.float64))
        self.roi_weight = float(self.weights[self.roi].sum(dtype=np.float64))

    def measure(self, core):
        if core.shape != self.shape or core.dtype != np.bool_:
            raise ValueError("Core must be native-resolution bool, without implicit resize")
        area = int(core.sum())
        overlap = int((core & self.roi_pixels).sum())
        occupancy = np.bincount(self.token_ids[core], minlength=256).reshape(16,16) / self.cell_areas
        captured = float((occupancy*self.weights).sum(dtype=np.float64))
        captured_roi = float((occupancy*self.weights*self.roi).sum(dtype=np.float64))
        cy, cx = np.nonzero(core)
        center = [float(cx.mean()+.5), float(cy.mean()+.5)] if area else None
        distance = math.dist(center, self.centroid) if area else None
        metric = dict(target_precision=overlap/area if area else 0., binary_roi_coverage=overlap/self.roi_area,
                      vulnerability_weighted_coverage=captured/self.total_weight if self.total_weight > 0 else None,
                      roi_conditioned_weighted_coverage=captured_roi/self.roi_weight if self.roi_weight > 0 else None,
                      weighted_coverage_defined=self.total_weight > 0,
                      binary_roi_token_coverage=float(occupancy[self.roi].mean()),
                      captured_token_weight=captured, total_token_weight=self.total_weight,
                      area_pixels=area, area_fraction=area/core.size, overlap_pixels=overlap,
                      roi_area_pixels=self.roi_area, centroid_xy=center, roi_centroid_xy=list(self.centroid),
                      centroid_distance_pixels=distance,
                      centroid_distance_normalized=distance/math.hypot(*self.shape) if area else None,
                      connectivity=8, num_components=components(core))
        return metric, occupancy


def facade_alignment(source, context):
    """Local dominant orthogonal edge frame, not a reconstructed facade plane."""
    image = np.asarray(source.convert("L"), dtype=np.float64)
    if image.shape != context.shape:
        raise ValueError("Source dimensions differ from native mask dimensions")
    ys, xs = np.nonzero(context.roi_pixels)
    pad = round(min(context.shape)*.1)
    y0,y1 = max(0,int(ys.min())-pad), min(image.shape[0],int(ys.max())+pad+1)
    x0,x1 = max(0,int(xs.min())-pad), min(image.shape[1],int(xs.max())+pad+1)
    gy,gx = np.gradient(image[y0:y1,x0:x1])
    magnitude = np.hypot(gx,gy)
    selected = (magnitude >= np.quantile(magnitude,.85)) & (magnitude > 0)
    if selected.sum() < 16:
        return dict(angle_degrees=0., method="image_axis_fallback", evidence_fraction=0., facade_plane_verified=False)
    # Edge tangent modulo 90 gives the horizontal axis of a local orthogonal frame.
    angle = (np.degrees(np.arctan2(gy[selected],gx[selected]))+90+45)%90-45
    strength = magnitude[selected]
    bins = np.arange(-45.5,46.5,1)
    histogram,_ = np.histogram(angle,bins=bins,weights=strength)
    smooth = np.convolve(histogram,np.ones(5),mode="same")
    best = int(np.argmax(smooth))
    axis = float((bins[best]+bins[best+1])/2)
    near = np.abs(angle-axis) <= 2.5
    fraction = float(strength[near].sum()/strength.sum())
    valid = fraction >= .15 and abs(axis) <= 30
    return dict(angle_degrees=float(np.average(angle[near],weights=strength[near])) if valid else 0.,
                method="local_gradient_orthogonal_frame" if valid else "image_axis_fallback",
                evidence_fraction=fraction, facade_plane_verified=False)


def candidate_centers(context, search):
    ys,xs = np.nonzero(context.roi_pixels)
    width,height = int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)
    cx,cy = context.centroid
    centers = {(round(cx+ox*width,6),round(cy+oy*height,6))
               for oy in search["center_offsets"] for ox in search["center_offsets"]}
    centers.add(((int(xs.min())+int(xs.max())+1)/2,(int(ys.min())+int(ys.max())+1)/2))
    order = sorted(map(tuple,np.argwhere(context.roi)),key=lambda p:(-float(context.weights[p]),*p))
    anchors = []
    for y,x in order:
        if len(anchors) >= search["roi_anchors"]:
            break
        if all(math.dist((y,x),point) >= search["anchor_separation_tokens"] for point in anchors):
            anchors.append((y,x))
            # Center of the actual integer native-pixel cell, not a resampled map.
            centers.add(((math.ceil(x*context.shape[1]/16)+math.ceil((x+1)*context.shape[1]/16))/2,
                         (math.ceil(y*context.shape[0]/16)+math.ceil((y+1)*context.shape[0]/16))/2))
    return sorted((x,y) for x,y in centers if 0 <= x < context.shape[1] and 0 <= y < context.shape[0])


def rasterize(shape, image_size, center, area_fraction, aspect, angle, phase):
    """Native pixel-center rasterization of a geometric occluder footprint."""
    factor = {"horizontal_compact": math.gamma(1.25)**2/math.gamma(1.5),
              "vertical_rectangle": 1., "facade_rectangle": 1., "cone_triangle": .5,
              "compact_irregular_blob": math.pi/4*(1+(.1**2+.06**2)/2)}[shape]
    nominal_area = image_size[0]*image_size[1]*area_fraction
    width,height = math.sqrt(nominal_area*aspect/factor),math.sqrt(nominal_area/aspect/factor)
    theta = math.radians(angle)
    c,s = math.cos(theta),math.sin(theta)
    margin = 1.17 if shape == "compact_irregular_blob" else 1.
    bx,by = margin*(abs(c)*width+abs(s)*height)/2,margin*(abs(s)*width+abs(c)*height)/2
    cx,cy = center
    x0,x1 = math.floor(cx-bx)-1, math.ceil(cx+bx)+1
    y0,y1 = math.floor(cy-by)-1, math.ceil(cy+by)+1
    x,y = np.arange(x0,x1)[None,:]+.5-cx,np.arange(y0,y1)[:,None]+.5-cy
    u,v = (c*x+s*y)/(width/2),(-s*x+c*y)/(height/2)
    if shape == "horizontal_compact":
        local = u**4+v**4 <= 1
    elif shape in ("vertical_rectangle","facade_rectangle"):
        local = (np.abs(u)<=1)&(np.abs(v)<=1)
    elif shape == "cone_triangle":
        local = (v>=-1)&(v<=1)&(np.abs(u)<=(v+1)/2)
    else:
        polar = np.arctan2(v,u)
        radius = 1+.1*np.cos(3*polar+phase)+.06*np.sin(5*polar-phase)
        local = np.hypot(u,v) <= radius
    core = np.zeros(image_size,dtype=bool)
    top,bottom,left,right = max(0,y0),min(image_size[0],y1),max(0,x0),min(image_size[1],x1)
    if top<bottom and left<right:
        core[top:bottom,left:right] = local[top-y0:bottom-y0,left-x0:right-x0]
    return core, dict(nominal_width_pixels=width,nominal_height_pixels=height,
                      unclipped_area_pixels=int(local.sum()),clipped_pixels=int(local.sum())-int(core.sum()))


def assess_thresholds(metrics, spec, geometry, config):
    reasons = []
    low,high = spec["area_range"]
    if not low <= metrics["area_fraction"] <= high:
        reasons.append("actual_core_area_outside_family_range")
    if geometry["clipped_pixels"]:
        reasons.append("core_footprint_clipped_at_image_boundary")
    if metrics["num_components"] != 1:
        reasons.append("core_not_one_8_connected_component")
    if metrics["target_precision"] < config["tau_target_precision"]:
        reasons.append("target_precision_below_threshold")
    if not metrics["weighted_coverage_defined"]:
        reasons.append("zero_total_weight_coverage_undefined")
    return {str(tau):dict(passes=not reasons and metrics["vulnerability_weighted_coverage"]>=tau,
                         rejection_reasons=reasons+(["weighted_coverage_below_threshold"]
                            if metrics["weighted_coverage_defined"] and metrics["vulnerability_weighted_coverage"]<tau else []))
            for tau in config["weighted_coverage_thresholds"]}


@dataclass
class Candidate:
    core_mask: np.ndarray
    render_mask: np.ndarray
    token_occupancy: np.ndarray
    diagnostic: dict


def generate_candidates(*, families, roi, weights, source, config, image_key, seed=0):
    """Yield EVERY configured hypothesis, including failed gates; never pick a winner."""
    validate_constraints(config)
    if type(seed) is not int or seed < 0 or len(set(families)) != len(families) or any(f not in FAMILY_SHAPES for f in families):
        raise ValueError("Require nonnegative seed and unique closed-taxonomy families")
    context = CoverageContext(roi,weights,(source.height,source.width))
    centers = candidate_centers(context,config["search"])
    alignment = facade_alignment(source,context) if any(FAMILY_SHAPES[f]=="facade_rectangle" for f in families) else None
    config_hash = config_digest(config)
    for family in sorted(families):
        spec = config["families"][family]
        phase = int.from_bytes(hashlib.sha256(f"blob:{seed}:{image_key}:{family}".encode()).digest()[:8],"big")/2**64*2*math.pi
        base_angle = alignment["angle_degrees"] if spec["shape"]=="facade_rectangle" else 0.
        for fraction in np.linspace(*spec["area_range"],spec["area_samples"]):
            for aspect in sorted(set(spec["aspect_ratios"])):
                for offset in sorted(set(spec["angle_offsets_degrees"])):
                    for center in centers:
                        core,geometry = rasterize(spec["shape"],context.shape,center,float(fraction),aspect,base_angle+offset,phase)
                        metrics,occupancy = context.measure(core)
                        render,render_diagnostic = build_render_mask(core,config["render"])
                        parameters = dict(family=family,shape=spec["shape"],requested_area_fraction=float(fraction),
                                          aspect_ratio=aspect,angle_degrees=base_angle+offset,center_xy=list(center),
                                          seed=seed,config_sha256=config_hash,image_key=image_key)
                        cid = hashlib.sha256(json.dumps(parameters,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()[:24]
                        diagnostic = dict(candidate_id=cid,parameters=parameters,metrics=metrics,geometry=geometry,
                                          facade_alignment=alignment if spec["shape"]=="facade_rectangle" else None,
                                          thresholds=assess_thresholds(metrics,spec,geometry,config),
                                          render=render_diagnostic,core_semantics="occluder footprint hypothesis; no blend allowance",
                                          support_contact_verified=False,selected=False)
                        yield Candidate(core,render,occupancy,diagnostic)
