"""Full-cohort planner with bounded source-schema repair, never template fallback."""
from __future__ import annotations

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import night_place_cohort as engine
from experiments.qwen_curriculum import night_place_cohort_v2 as schema

ROOT = engine.common.WORKSPACE_ROOT/'outputs/night_place_cohort_v3'
VERSION = 'night_place_cohort_v3'
_analysis_request = engine.analysis_request
_compose_request = engine.compose_request
_dependencies = engine.dependencies
_planner = engine.BatchedPlanner


def analysis_request():
    return (_analysis_request()+
        ' Use the EXACT region keys description and box. Each box is four numbers [x1,y1,x2,y2]. '
        'Write an original scene_summary of at least six words about THIS image. '
        'Describe actual visible permanent buildings, walls, facades, entrances or road layout. '
        'Do not copy placeholder text. Do not describe the future night edit. '
        'Use an empty protected_regions list only if no permanent landmark is actually visible.')


def compose_request(source, style, feedback=''):
    return (_compose_request(source,style,feedback)+
        ' Both lighting and exposure must contain at least two descriptive English words. '
        'The appearance_prompt must be your complete original 40-160 word instruction.')


class Planner(_planner):
    def ask(self,directory,sources,requests,phase,attempt,cfg):
        replies=super().ask(directory,sources,requests,phase,attempt,cfg)
        if phase!='analysis' or attempt!=0:
            return replies
        invalid=[]
        for i,raw in enumerate(replies):
            try: schema.canonical_evidence(engine.policy.parse_json(raw))
            except (ValueError,KeyError,TypeError) as error:
                invalid.append((i,str(error)))
        if invalid:
            repaired=super().ask(directory,[sources[i] for i,_ in invalid],
                [requests[i]+' Your previous response failed validation: '+error+
                 '. Inspect the image again and return a complete corrected JSON object. '
                 'Do not invent a landmark if none is visible.' for i,error in invalid],
                phase,1,cfg)
            for (i,_),raw in zip(invalid,repaired,strict=True): replies[i]=raw
        return replies


def dependencies(model):
    extra=(Path(__file__).resolve(),Path(schema.__file__).resolve())
    return {**_dependencies(model),**{str(p):engine.common.file_sha256(p) for p in extra}}


def configure():
    engine.ROOT,engine.VERSION=ROOT,VERSION
    engine.analysis_request=analysis_request
    engine.compose_request=compose_request
    engine.policy.capabilities=schema.canonical_evidence
    engine.BatchedPlanner=Planner
    engine.dependencies=dependencies


def generation_resource_plan(directory):
    configure()
    return engine.generation_resource_plan(directory)


if __name__=='__main__':
    configure()
    engine.main()
