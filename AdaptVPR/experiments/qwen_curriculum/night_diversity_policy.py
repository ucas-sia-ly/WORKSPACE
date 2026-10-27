"""Night-only source grounding, open prose and deterministic coverage allocation."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

from PIL import Image

from experiments.qwen_curriculum import common, run, scene_prompting as legacy

VERSION = 'scene_balanced_night_v1'
CATALOG_PATH = Path(__file__).with_name('night_style_catalog.json')
CATALOG = json.loads(CATALOG_PATH.read_text())
STYLES = {v['id']: v for v in CATALOG['styles']}
FEATURES = ('outdoors', 'drivable_road', 'walkable_ground', 'paved_ground', 'visible_lamps',
            'windows', 'shopfronts', 'signs', 'entrances', 'covered_passage', 'industrial_site',
            'visible_traffic', 'safe_vehicle_space', 'safe_person_space')
SNAPSHOT = {}
STATE = None

SYSTEM = '''You plan image-conditioned augmentation for visual place recognition.
These are open-ended descriptions, not a fixed list of prompt templates.
The supplied style categories describe lighting mechanisms and coverage goals,
not text to copy. Inspect the actual source image and write original, specific
editing instructions for exactly that location, using the selected mechanisms.
Only NIGHT is requested. Surface wetness, modest haze and photographic response
can vary within night. Do not turn this into daytime, a rain/snow/fog domain task,
or a daylight facade below an artificially black sky. When little sky is visible,
establish night through surface illumination, local light falloff and shadows.

Preserve permanent place identity: viewpoint, building shapes, facade layout,
window/door positions, sign shapes and lettering, road topology and fixed landmarks.
Vehicles and people MAY be added, removed or changed in physically plausible
locations. Dry pavement MAY become damp or wet, and vice versa. These are allowed
augmentation, not structural errors. Keep enough permanent landmarks visible.
Do not invent a shop, sign, window, streetlamp pole or architectural element to
make a lighting idea fit. Light actual existing openings and signs, or use
plausible off-frame illumination without drawing new fixtures. A vehicle light
must belong to an existing or explicitly introduced plausible vehicle. If the
image cannot support a mechanism, do not fabricate its structural prerequisite.
Make selected variants visibly different, with natural photographic night light.
The final complete English prompt is YOUR source-grounded prose, not a template.'''

CONTRACT = ('Keep this exact viewpoint, permanent building and facade geometry, window and door layout, '
            'existing sign shapes and lettering, road topology and fixed landmarks. '
            'Produce a real-looking NIGHT photograph. Plausible changes to vehicles, pedestrians and '
            'pavement wetness are allowed, provided useful permanent place evidence remains visible. '
            'Do not invent shops, signs, architecture or fixed light fixtures to justify illumination.')

parse_json = legacy.parse_json
text = legacy.text
box = legacy.box


def definition():
    return dict(version=VERSION, system=SYSTEM, contract=CONTRACT, catalog=CATALOG,
                allocation='source eligibility, then least saved-minus-mismatched + pending + local reservation; deterministic ties',
                coverage_snapshot=SNAPSHOT, conditions=['night'],
                transient_policy='plausible additions/removals and wet/dry changes allowed; no exact count or box requirement')


def capabilities(value):
    text(value.get('scene_summary'), 'scene summary', 3)
    features = value.get('features')
    if not isinstance(features, dict) or set(features) != set(FEATURES):
        raise ValueError('Every source capability requires explicit present/absent/uncertain evidence')
    for name, item in features.items():
        if not isinstance(item, dict) or item.get('status') not in ('present', 'absent', 'uncertain'):
            raise ValueError('Invalid source capability '+name)
        text(item.get('evidence'), name+' evidence', 2)
    if value.get('coordinate_system') != '0_1000':
        raise ValueError('Source landmarks require explicit native coordinates')
    regions = value.get('protected_regions')
    if not isinstance(regions, list) or not 1 <= len(regions) <= 3:
        raise ValueError('Identify one to three actual permanent landmarks')
    for region in regions:
        text(region.get('description'), 'landmark description', 2)
        raw = region.get('box')
        if not isinstance(raw, list) or len(raw) != 4:
            raise ValueError('Invalid native landmark box')
        box([v/1000 if type(v) in (int, float) else v for v in raw])
    return value


def eligible_styles(source):
    capabilities(source)
    present = {key for key, value in source['features'].items() if value['status'] == 'present'}
    eligible, excluded = [], {}
    for key, style in STYLES.items():
        missing = [clause for clause in style['requires'] if not present.intersection(clause)]
        if 'outdoors' not in present:
            missing = [['outdoors'], *missing]
        if missing:
            excluded[key] = dict(missing_confirmed_alternatives=missing,
                                 reason='absent or uncertain; do not invent a supporting object/structure')
        else:
            eligible.append(key)
    return eligible, excluded


def select_styles(eligible, count, coverage, local=(), seed=42, source_key=''):
    if type(count) is not int or count < 1 or len(set(eligible)) != len(eligible):
        raise ValueError('Invalid bounded style allocation')
    if any(key not in STYLES for key in eligible) or len(eligible) < count:
        raise ValueError('Insufficient source-compatible styles; do not force an unsuitable style')
    local_counts = Counter(local)
    def priority(key):
        stats = coverage.get(key, {})
        fields = ('saved', 'mismatch', 'pending')
        if any(type(stats.get(f, 0)) is not int or stats.get(f, 0) < 0 for f in fields):
            raise ValueError('Coverage counts must be nonnegative integers')
        if stats.get('mismatch', 0) > stats.get('saved', 0):
            raise ValueError('Mismatches exceed saved images')
        credit = stats.get('saved', 0)-stats.get('mismatch', 0)+stats.get('pending', 0)+local_counts[key]
        return credit, common.fingerprint([seed, source_key, key])
    return sorted(eligible, key=priority)[:count]


def normalize_plan(value):
    value = copy.deepcopy(value)
    if value.get('coordinate_system') != '0_1000':
        raise ValueError('Explicit 0-to-1000 landmark coordinates required')
    for region in value.get('protected_regions', []):
        raw = region['box']
        if any(type(v) not in (int, float) for v in raw):
            raise ValueError('Landmark coordinates must be numeric')
        region['box'] = [v/1000 for v in raw]
    value['coordinate_system'] = 'normalized_0_1'
    return value


def validate_plan(value, variants):
    if type(value.get('schema_version')) is not int or value['schema_version'] != 1 or value.get('coordinate_system') != 'normalized_0_1':
        raise ValueError('Unsupported balanced-night schema/coordinates')
    if not isinstance(value.get('source_evidence'), dict) or not isinstance(value.get('protected_regions'), list):
        raise ValueError('Missing source evidence or permanent landmarks')
    source = {**value['source_evidence'], 'protected_regions': [
        {**r, 'box': [v*1000 for v in r['box']]} for r in value['protected_regions']]}
    eligible, _ = eligible_styles(source)
    entries = value.get('variants')
    if not isinstance(entries, list) or len(entries) != variants:
        raise ValueError('Wrong variant count')
    keys, prose = set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Each variant must be an object')
        key = entry.get('style_id')
        if key not in eligible or key in keys:
            raise ValueError('Duplicate, unknown or source-incompatible style')
        if entry.get('weather') != 'night':
            raise ValueError('Only condition=night is enabled')
        text(entry.get('style_name'), 'style name')
        text(entry.get('lighting'), 'lighting', 3)
        text(entry.get('exposure'), 'exposure', 2)
        prompt = text(entry.get('appearance_prompt'), 'complete source-conditioned prompt', 25)
        if prompt.casefold() in prose:
            raise ValueError('Duplicate prompt prose')
        if entry.get('surface_state') not in ('dry', 'wet', 'mixed', 'unspecified'):
            raise ValueError('Record intended surface appearance separately from condition')
        keys.add(key); prose.add(prompt.casefold())
    if set(value.get('requested_style_ids', [])) != keys:
        raise ValueError('Composer changed the scheduled style categories')
    return value


def render_prompt(variant, with_transient=False):
    if with_transient:
        raise ValueError('Balanced night uses integrated augmentation, not exact transient pairs')
    return text(variant['appearance_prompt'], 'source-conditioned prompt')+'\n\n'+CONTRACT


def coverage_snapshot(states):
    counts = {key: dict(saved=0, pending=0, confirmed=0, mismatch=0) for key in STYLES}
    receipts, seen, files, unclassified = {}, set(), {}, set()
    for state in states:
        state = Path(state).resolve()
        for filename in ('plan.jsonl', 'generated.jsonl', 'results.jsonl', 'quality_results.jsonl'):
            path = state/filename
            if not path.exists():
                continue
            files[str(path)] = common.file_sha256(path)
            for row in common.read_jsonl(path):
                if row.get('condition') != 'night':
                    continue
                sid = row['sample_id']
                if sid in receipts and receipts[sid].get('record_sha256') != row.get('record_sha256'):
                    raise ValueError('Coverage states disagree on the same job')
                receipts[sid] = {**receipts.get(sid, {}), **row, '_state': str(state)}
    for row in receipts.values():
        key = row.get('style_id', row.get('appearance', {}).get('style_id', row.get('prompt_variant')))
        if key not in STYLES:
            unclassified.add(row['sample_id'])
            continue  # Never guess a category from a free style name.
        digest = row.get('output_sha256')
        if not digest:
            counts[key]['pending'] += 1
            continue
        if digest in seen:
            continue
        run.validate_image(row)
        seen.add(digest); counts[key]['saved'] += 1
        audit_file = Path(row['_state'])/'audit'/f"{row['sample_id']}.json"
        if audit_file.exists():
            files[str(audit_file)] = common.file_sha256(audit_file)
            verdict = json.loads(audit_file.read_text())
            if verdict != run.sealed(verdict) or verdict.get('output_sha256') != digest:
                raise ValueError('Style audit differs from its image')
            verdict = verdict['verdict']
            if verdict.get('style_matched') is False:
                counts[key]['mismatch'] += 1
            if verdict.get('style_matched') is True and verdict.get('status') == 'candidate':
                counts[key]['confirmed'] += 1
    return dict(version=1, counts=counts, observed_files_sha256=files,
                unclassified_night_records=len(unclassified),
                interpretation='saved/requested categories are provisional exposure, not verified lamp identity; old unknown styles not guessed')


class LocalVisionPlanner(legacy.LocalVisionPlanner):
    """Two bounded phases: read source capabilities, then compose scheduled styles."""
    def _ask(self, image_path, request, seed, sampling):
        with Image.open(image_path) as opened:
            image = opened.convert('RGB'); image.thumbnail((672, 672))
        messages = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
                    {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': request}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt').to(self.device)
        with self.torch.inference_mode(), self.torch.random.fork_rng(devices=[self.torch.cuda.current_device()]):
            self.torch.manual_seed(seed)
            options = dict(do_sample=True, temperature=.8, top_p=.9, top_k=30) if sampling else dict(do_sample=False)
            ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens, **options)
        return self.processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]

    def read(self, image_path, variants, seed, previous_styles=(), feedback=''):
        digest = common.file_sha256(image_path)
        path = STATE/'source_analysis'/f'{digest}.json'
        if path.exists():
            analysis = json.loads(path.read_text())
            if analysis != run.sealed(analysis) or analysis['source_sha256'] != digest:
                raise ValueError('Source analysis differs from input')
        else:
            request_file = path.with_suffix('.request.json')
            if request_file.exists():
                raise ValueError('Unresolved source-analysis call; inspect its raw evidence, do not silently retry')
            common.write_json(request_file, dict(source_sha256=digest, seed=seed))
            request = ('Inspect only visible source evidence. Return JSON: coordinate_system="0_1000", '
                'scene_summary, protected_regions (1-3 actual permanent landmarks, description and box), '
                'features: every key '+json.dumps(FEATURES)+' maps to {"status":"present|absent|uncertain", '
                '"evidence":"specific visible evidence"}. Do not infer a shop from an ordinary window, '
                'or traffic from a road. Missing sky does not imply indoors. Existing lamps may be outside '
                'the crop; do not mark them visible without evidence. Safe space means visible legal free '
                'ground with plausible object scale; uncertain is allowed. Use 0-to-1000 boxes, JSON only.')
            raw = self._ask(image_path, request, seed, False)
            common.write_json(path.with_suffix('.raw.json'), dict(response=raw))
            source = capabilities(parse_json(raw))
            eligible, excluded = eligible_styles(source)
            local = [name.split(':', 1)[0] for name in previous_styles]
            selected = select_styles(eligible, variants, SNAPSHOT['counts'], local, seed, digest)
            analysis = run.sealed(dict(source_sha256=digest, source=source, selected=selected, excluded=excluded))
            common.write_json(path, analysis)
        source, selected = analysis['source'], analysis['selected']
        if len(selected) != variants:
            raise ValueError('Source-analysis variant budget changed')
        instructions = [STYLES[key] for key in selected]
        request = ('Write exactly one variant for EACH supplied style_id, retaining those IDs. '
            'Use the actual input image and source evidence: '+json.dumps(source)+
            '\nSelected lighting mechanisms (NOT complete prompts): '+json.dumps(instructions)+
            '\nReturn JSON {"variants":[{"style_id":"selected ID","style_name":"original descriptive name",'
            '"weather":"night","surface_state":"dry|wet|mixed|unspecified",'
            '"lighting":"source-grounded mechanism","exposure":"night photographic response",'
            '"appearance_prompt":"complete original English prompt, 40-160 words"}]}. '
            'Vehicle/person additions/removals and wet/dry changes are permitted, with plausible placement. '
            'Do not invent structural prerequisites. All variants must depict night even with little sky. '+feedback)
        raw = self._ask(image_path, request, seed, True)
        try:
            value = parse_json(raw)
        except ValueError:
            return raw  # Existing bounded composer-repair budget handles malformed JSON.
        if not isinstance(value.get('variants'), list) or any(not isinstance(v, dict) for v in value['variants']):
            return raw
        for variant in value.get('variants', []):
            variant['style_name'] = str(variant.get('style_id'))+': '+str(variant.get('style_name', ''))
        value.update(schema_version=1, coordinate_system='0_1000', scene_summary=source['scene_summary'],
            protected_regions=source['protected_regions'], source_evidence=source,
            requested_style_ids=selected, variants=value.get('variants', []))
        return json.dumps(value)

    def audit(self, source_path, candidate_path, authorization):
        request = '''Compare Image 1 (source) and Image 2 (edited).
Judge NIGHT appearance and permanent PLACE STRUCTURE. Vehicle/person addition,
removal or motion, and wet/dry pavement changes, are ALLOWED and are not grounds
for rejection. They must not hide nearly all useful permanent landmarks.
Check viewpoint, road topology, facade/window/door layout, existing sign shape
and lettering; new shops, architecture or fixed fixtures are NOT allowed.
Judge night using surfaces and the lighting field even if little sky is visible;
a black sky alone does not prove night. Uncertain is valid; do not copy the plan.
Separately judge the requested lighting mechanism. A valid night may fail that
style: report style_matched=false rather than pretending diversity was achieved.
Return JSON only: {"structure_preserved":true|false|null,"is_night":true|false|null,
"landmarks_visible":true|false|null,"style_matched":true|false|null,
"surface_state":"dry|wet|mixed|uncertain","transient_changes":[],
"permanent_changes":[],"evidence":"brief specific comparison"}.
Authorization: '''+json.dumps(authorization)
        images = []
        for path in (source_path, candidate_path):
            with Image.open(path) as opened:
                image = opened.convert('RGB'); image.thumbnail((672, 672)); images.append(image)
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': im} for im in images]+
                     [{'type': 'text', 'text': request}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=700, do_sample=False)
        return self.processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
