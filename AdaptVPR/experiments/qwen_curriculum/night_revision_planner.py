"""Lazy, journaled Qwen3-VL inference for the revised source-grounded policy.

This adapter never changes a frozen cohort module or starts an image backend.
An interrupted *text inference* may be recomputed with its original seed after
archiving the exact reservation. Saved model responses are always reused first.
"""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import time

from PIL import Image

from experiments.qwen_curriculum import common, run, scene_prompting

VERSION = 'night-revision-planner-v1'
ANALYSIS_SYSTEM = (
    'You inspect an original photograph for visual place recognition. '
    'Describe only visible source evidence, whether the source is DAY or NIGHT. '
    'Do not assume objects because they would help a future night edit. '
    'Distinguish building/residential windows from vehicle windows, shop '
    'display glazing from ordinary glass doors, and a roofed passage from '
    'an ordinary entrance. Do not perform an edit during source analysis.')


def sealed_read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value != run.sealed(value):
        raise ValueError('Changed revision inference receipt: '+str(path))
    return value


class Planner:
    """Reuse the installed BF16/SDPA loader, loading it only on a cache miss.

    ``ask`` has the original BatchedPlanner signature. The caller owns the
    cohort lock, source selection, policy validation and atomic group commits.
    New output directories and config fingerprints isolate this revised policy
    from historical prompt caches. No old-policy module is monkeypatched.
    """
    def __init__(self, model_path, *, policy=None, allocator_gib=42):
        if not 0 < allocator_gib <= 42:
            raise ValueError('Allocator budget must be positive and at most 42 GiB')
        self.model_path = Path(model_path).resolve()
        self.policy = policy
        self.allocator_gib = allocator_gib
        self._backend = None

    def _policy(self):
        if self.policy is None:
            self.policy = importlib.import_module(
                'experiments.qwen_curriculum.night_diversity_policy_v2')
        return self.policy

    def ensure_model(self):
        if self._backend is None:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('Revision planning requires an available CUDA device')
            total = torch.cuda.get_device_properties(0).total_memory
            cap = min(self.allocator_gib*1024**3, .95*total)
            torch.cuda.set_per_process_memory_fraction(cap/total)
            self._backend = scene_prompting.LocalVisionPlanner(self.model_path)
        return self._backend

    @property
    def torch(self):
        return self.ensure_model().torch

    @property
    def processor(self):
        return self.ensure_model().processor

    def _identity(self, sources, requests, phase, attempt, cfg):
        if phase not in ('analysis', 'compose') or type(attempt) is not int or attempt < 0:
            raise ValueError('Require analysis/compose and nonnegative integer draft')
        if (not isinstance(sources, (list, tuple)) or not 1 <= len(sources) <= 64 or
                len(sources) != len(requests)):
            raise ValueError('Require one request per source in a bounded batch of 1..64')
        if any(not isinstance(r, str) or not r.strip() for r in requests):
            raise ValueError('Every revision inference request must be nonempty text')
        if not isinstance(cfg.get('fingerprint'), str) or not cfg['fingerprint'] or type(cfg.get('seed')) is not int:
            raise ValueError('Require an explicit cohort fingerprint and integer seed')
        source_inputs = []
        for source in sources:
            if any(not isinstance(source.get(k), str) or not source[k]
                   for k in ('sample_id', 'source_path', 'source_sha256')):
                raise ValueError('Every source requires exact ID, path and SHA256')
            if common.file_sha256(source['source_path']) != source['source_sha256']:
                raise ValueError('Revision source bytes changed: '+source['sample_id'])
            source_inputs.append({k: source[k] for k in
                                  ('sample_id', 'source_path', 'source_sha256')})
        if len({s['sample_id'] for s in sources}) != len(sources):
            raise ValueError('A revision inference batch must not repeat a source ID')
        token_field = 'analysis_tokens' if phase == 'analysis' else 'compose_tokens'
        max_tokens = cfg.get(token_field)
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError('Explicit bounded '+token_field+' is required')
        system = self._policy().SYSTEM if phase == 'compose' else ANALYSIS_SYSTEM
        identity=dict(version=VERSION, cohort_fingerprint=cfg['fingerprint'],
            phase=phase, attempt=attempt, source_inputs=source_inputs,
            requests=list(requests), system=system, model_path=str(self.model_path),
            max_tokens=max_tokens, thumbnail_max=[672, 672],
            decoding=(dict(do_sample=True, temperature=.8, top_p=.9, top_k=30)
                      if phase == 'compose' else dict(do_sample=False)))
        if cfg.get('execution_patch_sha256'):
            identity['execution_patch_sha256']=cfg['execution_patch_sha256']
        return identity

    def _infer(self, sources, requests, identity, seed):
        backend = self.ensure_model()
        conversations = []
        for source, request in zip(sources, requests, strict=True):
            with Image.open(source['source_path']) as opened:
                image = opened.convert('RGB')
                image.thumbnail((672, 672))
            conversations.append([
                dict(role='system', content=[dict(type='text', text=identity['system'])]),
                dict(role='user', content=[dict(type='image', image=image),
                                           dict(type='text', text=request)])])
        backend.processor.tokenizer.padding_side = 'left'
        inputs = backend.processor.apply_chat_template(conversations, tokenize=True,
            padding=True, add_generation_prompt=True, return_dict=True,
            return_tensors='pt').to(backend.device)
        torch = backend.torch
        with torch.inference_mode(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            generated = backend.model.generate(**inputs, max_new_tokens=identity['max_tokens'],
                use_cache=True, **identity['decoding'])
        responses = backend.processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return responses

    def ask(self, directory, sources, requests, phase, attempt, cfg):
        identity = self._identity(sources, requests, phase, attempt, cfg)
        key = common.fingerprint(identity)
        seed = common.candidate_seed(cfg['seed'], key, attempt)
        path = Path(directory)/'batches'/f'{key}.json'
        reservation_path = path.with_suffix('.request.json')
        expected = dict(key=key, identity=identity, seed=seed)
        if path.exists():
            cached = sealed_read(path)
            if any(cached.get(k) != v for k, v in expected.items()):
                raise ValueError('Saved revision response differs from exact inputs')
            replies = cached.get('responses')
            if (not isinstance(replies, list) or len(replies) != len(sources) or
                    any(not isinstance(v, str) for v in replies)):
                raise ValueError('Saved revision response has invalid cardinality/types')
            if reservation_path.exists():
                request = sealed_read(reservation_path)
                if cached.get('request_sha256') != request['result_sha256']:
                    raise ValueError('Saved revision response and reservation disagree')
            return copy.deepcopy(replies)
        inference_attempt = 0
        lineage = None
        if reservation_path.exists():
            prior = sealed_read(reservation_path)
            if (any(prior.get(k) != v for k, v in expected.items()) or
                    type(prior.get('inference_attempt')) is not int or prior['inference_attempt'] < 0):
                raise ValueError('Unresolved revision reservation differs from exact inputs')
            archive = Path(directory)/'inference_recovery'/key/f'{prior["inference_attempt"]:06d}_{prior["result_sha256"]}.request.json'
            if archive.exists() and sealed_read(archive) != prior:
                raise ValueError('Revision recovery archive was changed')
            if not archive.exists():
                common.write_json(archive, prior)
            inference_attempt = prior['inference_attempt']+1
            lineage = dict(prior_request_sha256=prior['result_sha256'], archive=str(archive),
                reason='Interrupted text-only VLM inference; deterministic recomputation authorized by revision protocol')
        request = run.sealed(dict(**expected, inference_attempt=inference_attempt,
            recovery=lineage, no_image_generation=True))
        common.write_json(reservation_path, request)
        started = time.monotonic()
        responses = self._infer(sources, requests, identity, seed)
        if (not isinstance(responses, list) or len(responses) != len(sources) or
                any(not isinstance(v, str) for v in responses)):
            raise RuntimeError('Revision VLM lost exact image/response correspondence')
        common.write_json(path, run.sealed(dict(**expected, responses=responses,
            request_sha256=request['result_sha256'], inference_attempt=inference_attempt,
            inference_seconds=time.monotonic()-started, no_image_generation=True)))
        return responses

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None
