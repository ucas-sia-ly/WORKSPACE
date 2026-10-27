"""Bulk night prompts: source evidence plus a full-frame preservation contract.

The preservation envelope is a policy, NOT a predicted landmark, semantic mask,
geometric evidence, or local-distillation supervision. Dynamic actors/wetness
remain allowed by the night policy. No VLM bounding-box prediction is required.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import night_place_cohort as engine

ROOT=engine.common.WORKSPACE_ROOT/'outputs/night_place_cohort_v4'
VERSION='night_place_cohort_v4'
_capabilities=engine.policy.capabilities
_dependencies=engine.dependencies
_compose_request=engine.compose_request
_planner=engine.BatchedPlanner


def source_evidence(value):
    value=copy.deepcopy(value)
    value['coordinate_system']='0_1000'
    value['region_policy']='full_frame_structural_preservation_contract_not_detection_or_supervision'
    value['protected_regions']=[dict(description=(
        'Preserve existing permanent structure throughout the original frame; '
        'this is a policy envelope, not a detected landmark or semantic mask. '
        'Plausible vehicle/person changes and wet/dry changes remain allowed.'),
        box=[0,0,1000,1000],origin='deterministic_preservation_contract',
        usable_as_correspondence_or_semantic_evidence=False)]
    return _capabilities(value)


def analysis_request():
    return ('Inspect this original photograph for visual place recognition. '
        'Describe the source as it actually appears, whether day or night. '
        'Return one JSON object with scene_summary and features. '
        'Write a specific scene_summary of at least six English words. '
        'Do not return bounding boxes or protected_regions. '
        'The features object must include every one of these exact keys: '+
        engine.json.dumps(engine.policy.FEATURES)+'. '
        'Each key must map to an object with status and evidence. '
        'status must be present, absent, or uncertain. evidence must contain '
        'at least two words describing actual visible evidence or its absence. '
        'A window is not automatically a shop. A road is not proof of active traffic. '
        'An empty sidewalk can provide safe person space; no person need already exist. '
        'No sky does not mean indoors. Do not assume lamps, shops, signs or structures '
        'because they would be useful for a future night edit. No markdown.')


def compose_request(source,style,feedback=''):
    return (_compose_request(source,style,feedback)+
        ' lighting and exposure must each have at least two descriptive English words. '
        'The full-frame box is only a permanent-structure preservation contract; '
        'it does not ban plausible people, vehicle or pavement-wetness changes. '
        'Do not copy placeholders. Write the complete original 40-160 word appearance_prompt.')


class Planner(_planner):
    def ask(self,directory,sources,requests,phase,attempt,cfg):
        replies=super().ask(directory,sources,requests,phase,attempt,cfg)
        if phase!='analysis' or attempt!=0: return replies
        invalid=[]
        for i,raw in enumerate(replies):
            try: source_evidence(engine.policy.parse_json(raw))
            except (ValueError,KeyError,TypeError) as error: invalid.append((i,str(error)))
        if invalid:
            repaired=super().ask(directory,[sources[i] for i,_ in invalid],
                [requests[i]+' Validation failed: '+error+
                 '. Inspect the image again and return the complete corrected JSON.' for i,error in invalid],
                phase,1,cfg)
            for (i,_),raw in zip(invalid,repaired,strict=True): replies[i]=raw
        return replies


def dependencies(model):
    return {**_dependencies(model),str(Path(__file__).resolve()):engine.common.file_sha256(__file__)}


def configure():
    engine.ROOT,engine.VERSION=ROOT,VERSION
    engine.analysis_request=analysis_request
    engine.compose_request=compose_request
    engine.policy.capabilities=source_evidence
    engine.BatchedPlanner=Planner
    engine.dependencies=dependencies


def generation_resource_plan(directory):
    configure()
    return engine.generation_resource_plan(directory)


if __name__=='__main__':
    configure()
    engine.main()
