"""Schema-hardened full-place night planning; preserves the v1 pilot evidence."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import night_place_cohort as engine

ROOT = engine.common.WORKSPACE_ROOT/'outputs/night_place_cohort_v2'
VERSION = 'night_place_cohort_v2'
_capabilities = engine.policy.capabilities
_dependencies = engine.dependencies


def canonical_evidence(value):
    """Accept explicit box-name aliases, never invent regions or coordinates."""
    value = copy.deepcopy(value)
    for region in value.get('protected_regions', []):
        keys = [key for key in ('box', 'bbox', 'bounding_box', 'region') if key in region]
        if len(keys) > 1 and any(region[key] != region[keys[0]] for key in keys[1:]):
            raise ValueError('Conflicting explicit landmark coordinates')
        if keys:
            region['box'] = region[keys[0]]
            for key in keys:
                if key != 'box': del region[key]
    return _capabilities(value)


def analysis_request():
    return ('Inspect only this original image, not a proposed edit. Return exactly one JSON object '
        'with this schema: {"coordinate_system":"0_1000","scene_summary":"visible scene",'
        '"protected_regions":[{"description":"visible permanent landmark",'
        '"box":[0,0,1000,1000]}],"features":{...}}. '
        'Replace the example coordinates with the actual tight xyxy box in native 0..1000 coordinates. '
        'Use the exact key "box". Identify 1-3 actual permanent landmarks: a building, facade, '
        'entrance, fixed wall, or visible road layout. Do not use movable vehicles or people as landmarks. '
        'If no permanent landmark is visible, an empty list is allowed and the image will be skipped. '
        'The features object must include EVERY key '+engine.json.dumps(engine.policy.FEATURES)+
        ', each mapping to {"status":"present|absent|uncertain","evidence":"specific visible evidence"}. '
        'An ordinary window is not a shop; a road alone does not prove active traffic. '
        'Missing sky does not imply indoors. No invented lamps, signs or object-placement space. '
        'No markdown or extra text.')


def dependencies(model):
    return {**_dependencies(model), str(Path(__file__).resolve()):engine.common.file_sha256(__file__)}


def configure():
    engine.ROOT, engine.VERSION = ROOT, VERSION
    engine.analysis_request = analysis_request
    engine.policy.capabilities = canonical_evidence
    engine.dependencies = dependencies


def generation_resource_plan(directory):
    configure()
    return engine.generation_resource_plan(directory)


if __name__ == '__main__':
    configure()
    engine.main()
