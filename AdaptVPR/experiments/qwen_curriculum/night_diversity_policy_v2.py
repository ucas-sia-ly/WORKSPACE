"""Versioned, typed source grounding for the post-mercury night revision.

This module never patches the original policy or its catalog.  The full-frame
region is a preservation contract, not a detection, mask or correspondence.
Style eligibility is a logical gate on visible evidence, not proof that an
edited image actually implements the requested light; that still needs audit.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import re
from pathlib import Path

from experiments.qwen_curriculum import common, night_diversity_policy as original

legacy = original.legacy
text = legacy.text
VERSION = 'scene_balanced_night_v2'
CATALOG_PATH = Path(__file__).with_name('night_style_catalog_v2.json')
CATALOG = json.loads(CATALOG_PATH.read_text())
STYLES = {style['id']: style for style in CATALOG['styles']}
RETIRED_STYLES = frozenset(CATALOG['retired_style_ids'])
FEATURE_DEFINITIONS = {
    'outdoors': 'Visible outdoor scene, including exterior views with little or no sky.',
    'drivable_road': 'Actual visible road or vehicle lane; not proof of vehicles or available space.',
    'walkable_ground': 'Visible walkable ground; not an invented walkway.',
    'paved_ground': 'Actual paved surface whose wetness may change without changing its geometry.',
    'building_windows': 'Visible architectural windows in a BUILDING facade; NEVER vehicle glazing or a glass door.',
    'residential_windows': 'Visible BUILDING windows with evidence of residential use; requires building_windows present. A vehicle window never qualifies.',
    'vehicle_windows': 'Visible car/bus/truck glazing or windshield; NEVER evidence of building or residential windows.',
    'visible_fixed_luminaires': 'Actual visible FIXED lighting fixtures or lamp heads; not inferred light, reflection, windows or vehicle lamps.',
    'shopfront_display_windows': 'Visible existing SHOP DISPLAY glazing and independent evidence of an actual shop; not an ordinary glass door, house window or billboard.',
    'ordinary_glass_doors': 'Plain glazed entrance doors; not by themselves shop display windows or residential windows.',
    'billboards': 'Existing freestanding or facade-mounted advertising panels; not evidence of a shop or its display glazing.',
    'existing_signs': 'Actual location/business/road sign face with visible shape and content; a billboard alone does not establish a business frontage.',
    'building_entrances': 'Actual doorway or entrance opening; not by itself a covered passage or a visible lamp.',
    'covered_passage': 'Actual visible roofed passage with overhead cover and passage depth; an ordinary doorway or entrance is insufficient.',
    'industrial_site': 'Visible industrial buildings/equipment/yard with source evidence, not a generic wall.',
    'visible_vehicles': 'Actual vehicles visible in the original image; a road alone is insufficient.',
    'visible_headlight_faces': 'Visible front headlamp faces attached to actual vehicles, with a compatible forward orientation; they need not already be lit.',
    'visible_taillight_faces': 'Visible rear lamp faces attached to actual vehicles, with a compatible rear orientation; they need not already be lit.',
    'safe_vehicle_space': 'Visibly free, plausible road space for a new vehicle at the correct scale, without hiding key permanent landmarks.',
    'safe_person_space': 'Visibly free, plausible walkable space for people at the correct scale, without hiding key permanent landmarks.',
}
FEATURES = tuple(FEATURE_DEFINITIONS)
ORIGINS = frozenset(('visible_fixture', 'off_frame', 'existing_opening', 'existing_sign',
                     'existing_vehicle', 'introduced_vehicle', 'ambient'))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Ambiguous duplicate key in source JSON quote repair: '+key)
        result[key] = value
    return result


def _unescaped_quotes(value):
    positions = []
    backslashes = 0
    for index, char in enumerate(value):
        if char == '"' and backslashes % 2 == 0:
            positions.append(index)
        backslashes = backslashes+1 if char == '\\' else 0
    return positions


def parse_json_with_diagnostics(raw):
    """Parse JSON, repairing only paired quotes inside complete source text lines.

    No field, delimiter, status, closing brace or evidence text is supplied.
    The fallback accepts only the complete typed source schema and leaves every
    character unchanged except an inserted escape before an inner double quote.
    Composition responses and ambiguous inline objects never use this fallback.
    """
    try:
        return legacy.parse_json(raw), []
    except json.JSONDecodeError as original_error:
        body = raw.strip()
        if body.startswith('```'):
            if not body.endswith('```') or '\n' not in body:
                raise original_error
            body = body.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        if not body.startswith('{') or not body.endswith('}') or '\n' not in body:
            raise original_error
        lines = body.splitlines(keepends=True)
        changes = []
        for index, line in enumerate(lines):
            # A field value must occupy a whole physical line. Greedy content
            # capture retains the original outer closing quote and comma.
            match = re.fullmatch(r'([ \t]*"(scene_summary|evidence)"[ \t]*:[ \t]*")'
                                 r'(.*)("[ \t]*,?[ \t]*)(\r?\n)?', line)
            if match is None:
                continue
            content = match[3]
            inner = _unescaped_quotes(content)
            if not inner:
                continue
            if (len(inner) % 2 or any(char in content for char in '{}[]')
                    or any(re.match(r'"[ \t]*:', content[position:]) for position in inner)):
                raise ValueError('Ambiguous source text; refusing structural JSON quote repair') from original_error
            positions = set(inner)
            escaped = ''.join(('\\' if pos in positions else '')+char
                              for pos, char in enumerate(content))
            # Detect invalid original escapes rather than correcting them.
            json.loads('"'+escaped+'"')
            lines[index] = match[1]+escaped+match[4]+(match[5] or '')
            changes.append(dict(kind='escape_unescaped_source_text_quotes', field=match[2],
                                line=index+1, escaped_quote_count=len(inner)))
        if not changes:
            raise original_error
        value = json.loads(''.join(lines), object_pairs_hook=_unique_object)
        if (not isinstance(value, dict) or set(value) != {'scene_summary', 'features'}
                or not isinstance(value['scene_summary'], str)
                or not isinstance(value['features'], dict) or set(value['features']) != set(FEATURES)):
            raise ValueError('Quote repair requires a complete source analysis object') from original_error
        for item in value['features'].values():
            if (not isinstance(item, dict) or set(item) != {'status', 'evidence'}
                    or item['status'] not in ('present', 'absent', 'uncertain')
                    or not isinstance(item['evidence'], str)):
                raise ValueError('Quote repair cannot repair source fields or statuses') from original_error
        return value, changes


def parse_json(raw):
    return parse_json_with_diagnostics(raw)[0]


def _source_text(value, name, minimum):
    """Retain English word limits while counting unspaced Han source evidence."""
    if isinstance(value, str) and len(re.findall(r'[\u3400-\u9fff]', value)) >= 2*minimum:
        return value.strip()
    return text(value, name, minimum)


PRESERVATION = (
    'Original viewpoint and camera geometry; building outlines and facade details; '
    'the exact positions, shapes and layout of existing doors and windows; road topology; '
    'existing sign outlines and lettering; fixed facilities and their positions and shapes; '
    'permanent structural details used to identify this place. Do not create shops, building '
    'openings, signs, fixed lighting fixtures or architecture to justify a lighting idea.'
)
ALLOWED = (
    'Light distribution, exposure and colour temperature may change. Physically plausible '
    'people and vehicles may be added, removed or changed while leaving useful permanent '
    'landmarks visible. Ground wetness may change ONLY to the declared surface state below.'
)
CONTRACT = 'MUST PRESERVE\n'+PRESERVATION+'\n\nMAY CHANGE\n'+ALLOWED
SYSTEM = '''You plan image-conditioned augmentation for visual place recognition.
These are open-ended descriptions, not a fixed list of prompt templates.
Only NIGHT is requested. Inspect the actual source image and compose original,
image-specific prose for a compatible lighting mechanism, rather than forcing
objects into the image to satisfy a category. A black sky over a daytime facade
is not night; with little sky, establish night through the light field, surface
luminance, directional falloff and dark intervals. Keep natural material colours.

Distinguish BUILDING windows, residential BUILDING windows and VEHICLE windows.
A car windshield never authorizes residential window glow. An ordinary glass
door or an advertising billboard never establishes a shop display window.
An ordinary entrance never establishes a roofed passage. Only positively visible
fixed luminaires may be described as in-frame fixtures. Otherwise place the
source explicitly OUTSIDE THE FRAME and describe illuminated existing surfaces,
incidence direction and falloff, without asking for a row of lamps on a wall.
Do not state one clause requesting new fixtures and another forbidding them.

Separate MUST PRESERVE structure from MAY CHANGE appearance. Keep viewpoint,
permanent building/facade geometry, door/window layout, road topology, sign
content and fixed facilities. Reasonable vehicle/person additions/removals and
DECLARED dry/wet changes are allowed. Do not invent shops, signs or architecture.
Style coverage is subordinate to scene suitability; unsupported or uncertain
prerequisites must not be manufactured. Mercury-vapor lighting is retired.
The final English appearance prose must be specific to this exact photograph.'''


def definition():
    return dict(version=VERSION, system=SYSTEM, contract=CONTRACT, catalog=CATALOG,
        conditions=['night'], source_schema_version=2,
        allocation='positive typed source evidence, then least saved-minus-mismatched + pending + local reservation; deterministic ties',
        structural_contract_is_semantic_evidence=False,
        retired_styles=sorted(RETIRED_STYLES))


def source_evidence(value):
    """Validate new visible evidence and attach an explicitly non-semantic envelope."""
    if not isinstance(value, dict):
        raise ValueError('Source evidence must be an object')
    result = copy.deepcopy(value)
    _source_text(result.get('scene_summary'), 'specific visible source summary', 6)
    features = result.get('features')
    if not isinstance(features, dict) or set(features) != set(FEATURES):
        raise ValueError('Every v2 typed source capability requires explicit present/absent/uncertain evidence')
    for name, item in features.items():
        if not isinstance(item, dict) or item.get('status') not in ('present', 'absent', 'uncertain'):
            raise ValueError('Invalid source capability '+name)
        evidence = _source_text(item.get('evidence'), name+' visible evidence', 2)
        if re.search(r'\b(?:REPLACE with|placeholder|INSERT EVIDENCE)\b', evidence, re.I):
            raise ValueError('Source evidence still contains a response-shape placeholder: '+name)
    if features['residential_windows']['status'] == 'present' and features['building_windows']['status'] != 'present':
        raise ValueError('Residential windows require positively visible BUILDING windows, not vehicle windows')
    if features['shopfront_display_windows']['status'] == 'present' and features['building_windows']['status'] != 'present':
        raise ValueError('Shop display windows require positively visible BUILDING windows')
    for key in ('visible_headlight_faces', 'visible_taillight_faces'):
        if features[key]['status'] == 'present' and features['visible_vehicles']['status'] != 'present':
            raise ValueError('Visible vehicle lamp faces require actual visible vehicles')
    result.update(source_schema_version=2, coordinate_system='0_1000',
        region_policy='full_frame_structural_preservation_contract_not_detection_or_supervision',
        protected_regions=[dict(description='Preserve the existing permanent place structure throughout the original frame.',
            box=[0, 0, 1000, 1000], origin='deterministic_preservation_contract',
            usable_as_correspondence_or_semantic_evidence=False)])
    return result


capabilities = validate_source = source_evidence


def eligible_styles(source):
    source = source_evidence(source)
    present = {key for key, value in source['features'].items() if value['status'] == 'present'}
    eligible, excluded = [], {}
    for key, style in STYLES.items():
        missing = [clause for clause in style['requires'] if not present.intersection(clause)]
        if 'outdoors' not in present:
            missing = [['outdoors'], *missing]
        if missing:
            excluded[key] = dict(missing_confirmed_alternatives=missing,
                reason='absent or uncertain typed evidence; do not manufacture a supporting structure')
        else:
            eligible.append(key)
    return eligible, excluded


def select_styles(eligible, count, coverage, local=(), seed=42, source_key=''):
    if type(count) is not int or count < 1 or len(set(eligible)) != len(eligible):
        raise ValueError('Invalid bounded style allocation')
    if any(key not in STYLES for key in eligible) or len(eligible) < count:
        raise ValueError('Insufficient compatible styles; never force an unsuitable or retired category')
    local_counts = Counter(local)
    def priority(key):
        stats = coverage.get(key, {})
        if any(type(stats.get(field, 0)) is not int or stats.get(field, 0) < 0
               for field in ('saved', 'mismatch', 'pending')):
            raise ValueError('Coverage counts must be nonnegative integers')
        if stats.get('mismatch', 0) > stats.get('saved', 0):
            raise ValueError('Mismatches exceed saved images')
        credit = stats.get('saved', 0)-stats.get('mismatch', 0)+stats.get('pending', 0)+local_counts[key]
        return credit, common.fingerprint([VERSION, seed, source_key, key])
    return sorted(eligible, key=priority)[:count]


def analysis_request():
    shape = dict(scene_summary='REPLACE with at least six specific English words about this source',
        features={key: dict(status='CHOOSE present OR absent OR uncertain',
            evidence='REPLACE with two to eight words of actual visible evidence') for key in FEATURES})
    return ('Inspect this ORIGINAL photograph as visible, whether day or night. Return ONE JSON OBJECT '
        'with BOTH top-level keys scene_summary AND features. Keep scene_summary in every reply, including '
        'repairs. Do not return only the features map. Do not return bounding boxes, protected_regions or '
        'a proposed edit. ALL scene_summary and evidence strings must be ENGLISH, including every repair. '
        'Use concise descriptions of visible objects rather than transcribing sign lettering. Do not put '
        'quoted OCR text inside evidence strings; describe the sign face instead. Every string must be '
        'valid JSON: escape any unavoidable internal double quote with a backslash. '
        'Every features value MUST be a nested OBJECT with exactly status and evidence. '
        'features:{"outdoors":"present"} is INVALID: a status STRING alone is forbidden. '
        'For EVERY feature choose one status literal, present, absent or uncertain, and write two to eight '
        'source-specific evidence words, including visible evidence or why it is absent/ambiguous. '
        'Replace ALL placeholder text in the JSON skeleton below; never copy placeholder evidence. '
        'Use uncertain when the object type is ambiguous; do not infer a future light source from a road, '
        'window reflection or desired night style. In particular VEHICLE windows are not BUILDING windows; '
        'residential_windows and shopfront_display_windows require building_windows present. '
        'Do not infer shop displays from a glass door, a billboard, or a house window; do not infer a covered '
        'passage from an ordinary doorway. Existing luminaires outside the crop are NOT visible luminaires. '
        'Feature meanings: '+json.dumps(FEATURE_DEFINITIONS)+
        '\nREQUIRED COMPLETE RESPONSE SHAPE (replace all placeholders, retain every key and nested object):\n'+
        json.dumps(shape)+'\nReturn exactly this complete nested JSON shape, no markdown or commentary.')


def _composition_origins(source, style):
    """Describe feasible origins without changing eligibility or validation rules."""
    present = lambda name: source['features'][name]['status'] == 'present'
    available = {'off_frame', 'ambient'}
    if present('visible_fixed_luminaires'):
        available.add('visible_fixture')
    if present('building_windows'):
        available.add('existing_opening')
    if present('existing_signs'):
        available.add('existing_sign')
    if present('visible_vehicles'):
        available.add('existing_vehicle')
    if present('drivable_road') and present('safe_vehicle_space'):
        available.add('introduced_vehicle')
    mechanism = STYLES[style]
    feasible = [origin for origin in mechanism['allowed_origins'] if origin in available]
    required = [[origin for origin in group if origin in feasible]
                for group in mechanism['required_origin_groups']]
    if not feasible or any(not group for group in required):
        raise ValueError('No source-supported origin can satisfy the assigned mechanism')
    return feasible, required


def compose_request(source, style, feedback=''):
    source = source_evidence(source)
    if style not in eligible_styles(source)[0]:
        raise ValueError('Cannot compose an unsupported or retired night style')
    mechanism = {key: value for key, value in STYLES[style].items() if key not in ('name', 'references')}
    feasible, required = _composition_origins(source, style)
    suggested = list(dict.fromkeys(group[0] for group in required)) or [feasible[0]]
    shape = dict(variants=[dict(style_id=style, style_name='REPLACE: original English name, at least two words',
        weather='night', surface_state='CHOOSE dry OR wet OR mixed',
        lighting='REPLACE: source-specific lighting description, at least three English words',
        exposure='REPLACE: source-specific nighttime camera response, at least two English words',
        illumination=[dict(origin=origin,
            evidence='REPLACE: actual source anchor or explicit off-frame position',
            surfaces='REPLACE: actual existing surfaces receiving this light',
            direction='REPLACE: light incidence direction for this image',
            falloff='REPLACE: spatial light decay or diffuse night field') for origin in suggested],
        appearance_prompt='REPLACE: original image-specific English night editing prose, 40-160 words')])
    return ('Write exactly ONE image-specific NIGHT variant. All prose strings must be ENGLISH. '
        'Replace every REPLACE/CHOOSE placeholder with your actual answer; never copy schema labels. '
        'style_name must have at least TWO English words; lighting at least THREE; exposure at least TWO. '
        'appearance_prompt must contain 40 to 200 English words. EVERY illumination evidence, surfaces, '
        'direction and falloff value must contain at least TWO whitespace-separated English words of '
        'actual descriptive prose. Only origin is an enum code: never copy off_frame, ambient or another '
        'origin code into evidence, surfaces, direction or falloff. Describe the actual source location, '
        'receiving surfaces, light incidence and decay instead of using a code or placeholder. '
        'Preserve permanent structure while changing '
        'daytime illumination to the assigned night mechanism, not just darkening the sky. '
        'Visible source evidence: '+json.dumps(source)+
        '\nAssigned mechanism, NOT a prompt template: '+json.dumps(mechanism)+
        '\nAllowed illumination.origin values for THIS style ONLY: '+json.dumps(mechanism['allowed_origins'])+
        '. Do not use any other origin, even if that other source object exists. '
        '\nSource-supported illumination.origin values for THIS photograph: '+json.dumps(feasible)+
        '\nMandatory source-supported origin groups: '+json.dumps(required)+
        '. Include at least ONE origin from EACH group; groups are simultaneous requirements, while '
        'members within a group are alternatives. Use one to three illumination entries. The response '
        'shape below suggests a source-supported origin for every required group; you may choose another '
        'source-supported member of that group. Ambient light may supplement a required artificial '
        'source, but ambient NEVER replaces off_frame or visible_fixture when that group is required. '
        'Do not invent a required source object. Existing unlit lamp/headlight/taillight faces in a '
        'DAY photograph MAY be switched on for the NIGHT edit; preserve the actual fixture or vehicle '
        'geometry, not its daytime unlit state. Plausible vehicle/person motion is permitted. '
        'Where no fixed lamp is visible and the assigned mechanism requires artificial illumination, '
        'use off_frame IF it is in the source-supported group. Keep that emitter outside the frame; '
        'describe the actual receiving surfaces, light direction and decay. The absence of a visible '
        'fixture does not forbid off-frame artificial light and does not authorize a new fixture. '
        'For ambient mechanisms, use ambient only when consistent with their assigned groups. existing_opening means '
        'BUILDING windows; vehicle glazing is not residential lighting. A glass door is not shop display '
        'glazing; an ordinary entrance is not a covered passage. Declare dry/wet/mixed ground consistently. '
        'The renderer appends separate MUST PRESERVE and MAY CHANGE blocks; keep useful permanent '
        'landmarks visible and do not add architecture or fixed fixtures. '
        '\nReturn this complete JSON shape with every placeholder replaced:\n'+json.dumps(shape)+
        '\nJSON only, no markdown. '+feedback)


def compose_feedback(raw, source, style, error):
    """Explain repairable output errors without changing source facts or labels."""
    source = source_evidence(source)
    if style not in eligible_styles(source)[0]:
        raise ValueError('Cannot repair an unsupported or retired night style')
    feasible, required = _composition_origins(source, style)
    entry = {}
    parse_error = None
    try:
        value = parse_json(raw)
        variants = value.get('variants') if isinstance(value, dict) else None
        if isinstance(variants, list) and len(variants) == 1 and isinstance(variants[0], dict):
            entry = variants[0]
    except (ValueError, TypeError, KeyError) as exc:
        parse_error = str(exc)[:400]
    fields = ('style_id', 'style_name', 'weather', 'surface_state', 'lighting',
              'exposure', 'illumination', 'appearance_prompt')
    illumination = entry.get('illumination')
    origins = [item['origin'] for item in illumination
               if isinstance(item, dict) and isinstance(item.get('origin'), str)] \
              if isinstance(illumination, list) else []
    field_requirements = dict(style_name=dict(min_words=2), lighting=dict(min_words=3),
        exposure=dict(min_words=2), appearance_prompt=dict(min_words=40, max_words=200),
        illumination_text=dict(fields=['evidence', 'surfaces', 'direction', 'falloff'], min_words=2))
    invalid_fields = []
    def inspect_field(path, content, minimum, maximum=None):
        words = len(content.strip().split()) if isinstance(content, str) else 0
        try:
            _english_text(content, path, minimum)
            if maximum is not None and words > maximum:
                raise ValueError('Exceeds maximum of '+str(maximum)+' words')
        except (ValueError, TypeError) as exc:
            invalid_fields.append(dict(field=path, value=content[:300] if isinstance(content, str) else content,
                word_count=words, min_words=minimum, max_words=maximum, reason=str(exc)))
    for name in ('style_name', 'lighting', 'exposure', 'appearance_prompt'):
        limits = field_requirements[name]
        inspect_field(name, entry.get(name), limits['min_words'], limits.get('max_words'))
    if isinstance(illumination, list):
        for index, item in enumerate(illumination):
            if isinstance(item, dict):
                for name in field_requirements['illumination_text']['fields']:
                    inspect_field('illumination['+str(index)+'].'+name, item.get(name), 2)
    diagnostics = dict(assigned_style=style,
        missing_fields=[name for name in fields if not entry.get(name)],
        current_origins=origins, source_supported_origins=feasible,
        mandatory_origin_groups=required,
        missing_origin_groups=[group for group in required if not set(origins).intersection(group)],
        unsupported_origins=sorted(set(origins)-set(feasible)),
        field_requirements=field_requirements, invalid_fields=invalid_fields)
    if parse_error:
        diagnostics['json_error'] = parse_error
    prior = raw if isinstance(raw, str) else repr(raw)
    diagnostics['previous_response_sha256'] = hashlib.sha256(prior.encode('utf-8')).hexdigest()
    corrections = []
    for item in invalid_fields:
        instruction = ('Rewrite '+item['field']+': current value '+json.dumps(item['value'])+' has '+
            str(item['word_count'])+' whitespace-separated word(s); minimum '+str(item['min_words'])+
            ' English words'+(' and maximum '+str(item['max_words']) if item['max_words'] is not None else '')+
            '. Validation reason: '+item['reason']+'.')
        if item['field'].endswith('.evidence'):
            instruction += ' Write actual descriptive source-location evidence, not an origin enum.'
        corrections.append(instruction)
    return ('Previous response rejected: '+str(error)[:600]+'. '
        'Regenerate the COMPLETE corrected JSON, retaining valid semantics and source-specific grounding. '
        'Keep the assigned style and confirmed source facts unchanged; do not invent objects to satisfy '
        'a missing origin. ALL prose must remain English and valid JSON. In particular, an ambient-only '
        'entry cannot satisfy an off_frame/visible_fixture group. '
        'Repair diagnostics: '+json.dumps(diagnostics)+
        '\nMANDATORY CORRECTIONS: '+(' '.join(corrections) if corrections else str(error)[:600])+'. '
        'Replace EVERY invalid field before returning the complete JSON. Do not repeat the rejected '
        'response or an invalid enum-valued evidence string.')


def normalize_plan(value):
    value = copy.deepcopy(value)
    if value.get('coordinate_system') != '0_1000':
        raise ValueError('Explicit native coordinates required before normalization')
    for region in value.get('protected_regions', []):
        raw = region.get('box')
        if not isinstance(raw, list) or len(raw) != 4 or any(type(v) not in (int, float) for v in raw):
            raise ValueError('Invalid preservation envelope coordinates')
        region['box'] = [v/1000 for v in raw]
    value['coordinate_system'] = 'normalized_0_1'
    return value


_PLACEHOLDER_STRINGS = frozenset((
    'original descriptive name', 'source-grounded lighting mechanism',
    'specific nighttime camera response', 'specific source anchor or explicit off-frame position',
    'actual existing surfaces receiving this light', 'source-specific incidence direction',
    'source-specific decay or spatially diffuse night field',
))


def _english_text(value, name, minimum=2):
    value = text(value, name, minimum)
    if value.casefold() in _PLACEHOLDER_STRINGS or re.search(r'\b(?:REPLACE|CHOOSE|placeholder)\b', value, re.I):
        raise ValueError('Unreplaced composition placeholder: '+name)
    if re.search(r'[\u3400-\u9fff]', value) or len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", value)) < minimum:
        raise ValueError('English source-specific text required: '+name)
    return value


def _validate_illumination(entry, source, style):
    values = entry.get('illumination')
    if not isinstance(values, list) or not 1 <= len(values) <= 3:
        raise ValueError('One to three explicit illumination origins required')
    features = source['features']
    present = lambda key: features[key]['status'] == 'present'
    origins = set()
    for item in values:
        if not isinstance(item, dict) or item.get('origin') not in ORIGINS:
            raise ValueError('Invalid illumination origin')
        origin = item['origin']
        if origin not in style['allowed_origins']:
            raise ValueError('Illumination origin conflicts with the assigned mechanism')
        for field in ('evidence', 'surfaces', 'direction', 'falloff'):
            _english_text(item.get(field), 'illumination '+field, 2)
        if origin == 'visible_fixture' and not present('visible_fixed_luminaires'):
            raise ValueError('No positively visible fixed fixture: keep the source off frame')
        if origin == 'existing_opening' and not present('building_windows'):
            raise ValueError('Light-emitting existing openings require BUILDING windows, not vehicle glazing or doors')
        if origin == 'existing_sign' and not present('existing_signs'):
            raise ValueError('A sign light requires an actually evidenced existing sign')
        if origin == 'existing_vehicle' and not present('visible_vehicles'):
            raise ValueError('Existing vehicle light requires an actual source vehicle')
        if origin == 'introduced_vehicle' and not (present('drivable_road') and present('safe_vehicle_space')):
            raise ValueError('A new vehicle requires positively evidenced free road space')
        origins.add(origin)
    if any(not origins.intersection(group) for group in style['required_origin_groups']):
        raise ValueError('Selected style lacks its required source-grounded illumination mechanism')


# These guards catch explicit metadata/prose contradictions, not arbitrary
# hallucinations. They are deliberately conservative about object assertions:
# lit existing surfaces are valid even when the emitting lamp is off frame.
_NEGATION = re.compile(r"\b(?:do not|don't|never|without|avoid|must not|should not|no|not)\b", re.I)
_OFF_FRAME = re.compile(r'\b(?:off[\s-]?frame|out of view|outside (?:the )?(?:image|frame|crop)|beyond (?:the )?(?:frame|crop))\b', re.I)
_FIXTURE_ASSERTIONS = tuple(re.compile(pattern, re.I) for pattern in (
    r'\b(?:add|create|install|introduce|place|draw|mount)\b.{0,45}\b(?:lamps?|luminaires?|light[\s-]?fixtures?|floodlights?|fluorescent tubes?|streetlights?)\b',
    r'\b(?:wall[\s-]?mounted|ceiling[\s-]?mounted|pole[\s-]?mounted|hanging|suspended|rows? of|series of|line of)\b.{0,35}\b(?:lamps?|lights?|luminaires?|fixtures?|fluorescent tubes?)\b',
    r'\b(?:lamps?|luminaires?|fixtures?|fluorescent tubes?|streetlights?)\b.{0,30}\b(?:along|mounted on|attached to|lining|fixed to)\b',
    r'\b(?:wall[\s-]lights?|existing (?:fixed )?(?:lamps?|luminaires?|fixtures?|floodlights?))\b',
))
_WINDOW_ASSERTIONS = tuple(re.compile(pattern, re.I) for pattern in (
    r'\b(?:glowing|illuminated|lit|luminous)\s+(?:\w+[\s-]){0,3}windows?\b',
    r'\bwindows?\s+(?:\w+\s+){0,4}(?:glow(?:s|ing)?|emit(?:s|ting)?|spill(?:s|ing)?|shine(?:s)?|illuminated|lit|light up)\b',
    r'\bwindow[\s-](?:glow|light|illumination|spill)\b',
))
_SHOP_ASSERTIONS = tuple(re.compile(pattern, re.I) for pattern in (
    r'\b(?:glowing|illuminated|lit|bright|luminous)\s+(?:\w+\s+){0,2}(?:shops?|storefronts?|shopfronts?|shop windows?|display windows?)\b',
    r'\b(?:shop(?:front)?|storefront|store|display)[\s-](?:windows?|glazing|lights?|glow|illumination)\b',
    r'\b(?:shopfronts?|storefronts?|shop windows?|display windows?)\s+(?:\w+\s+){0,3}(?:glow|illuminate|light|spill|emit)\w*\b',
))
_PASSAGE_ASSERTIONS = tuple(re.compile(pattern, re.I) for pattern in (
    r'\b(?:covered|roofed)\s+(?:passage|corridor|walkway|arcade|tunnel)\b',
    r'\b(?:fluorescent|tube)[\s-](?:lit|lighting)\s+(?:passage|corridor|walkway)\b',
))


_WET_GROUND_ASSERTIONS = tuple(re.compile(pattern, re.I) for pattern in (
    r'\b(?:wet|damp|rain[\s-]soaked|waterlogged)\s+(?:\w+\s+){0,2}(?:pavement|road|asphalt|ground|street|sidewalk|tarmac)\b',
    r'\b(?:pavement|road|asphalt|ground|street|sidewalk|tarmac)\s+(?:is|looks|remains|stays|becomes)\s+(?:visibly |slightly |fully )?(?:wet|damp)\b',
    r'\b(?:wet|watery|puddle)[\s-]reflections?\b',
    r'\bpuddles?\s+(?:on|across|along)\s+(?:the )?(?:pavement|road|asphalt|ground|street|sidewalk|tarmac)\b',
))
_DRY_GROUND_ASSERTIONS = tuple(re.compile(pattern, re.I) for pattern in (
    r'\bdry\s+(?:\w+\s+){0,2}(?:pavement|road|asphalt|ground|street|sidewalk|tarmac)\b',
    r'\b(?:pavement|road|asphalt|ground|street|sidewalk|tarmac)\s+(?:(?:is|looks|remains|stays)\s+)?(?:entirely |fully |visibly )?dry\b',
))


def _ground_assertion(prose, patterns, declared):
    for clause, match in _assertions(prose, patterns):
        # A policy permission is not a statement that the edited ground is wet.
        if re.search(r'\b(?:may|can|could|permitted|allowed|optional)\b', clause, re.I):
            continue
        # An explicit dry-to-wet edit can mention the original dry surface.
        if declared == 'wet' and re.search(r'\b(?:make|turn|change|transform|convert)\b.*\b(?:wet|damp)\b', clause, re.I):
            continue
        return match
    return None


_DAYLIGHT_ASSERTIONS = tuple(re.compile(pattern, re.I) for pattern in (
    r'\b(?:keep|retain|preserve|maintain|leave)\b.{0,40}\b(?:daylight|daytime|sunlit)\s+(?:appearance|lighting|illumination|exposure|brightness|look|facade|surfaces?)\b',
    r'\b(?:remain|stay|still|unchanged)\b.{0,20}\b(?:daylight|daytime|sunlit)\b',
    r'\b(?:daytime|daylight)\s+(?:lighting|illumination|exposure)\b',
    r'\b(?:only|just|simply)\s+(?:darken|blacken)\s+(?:the )?sky\b',
))


def _validate_night_effect(entry, prose):
    for clause, _ in _assertions(prose, _DAYLIGHT_ASSERTIONS):
        if re.search(r'\b(?:change|convert|transform|replace|turn)\b.*\b(?:night|nocturnal|nighttime)\b', clause, re.I):
            continue
        raise ValueError('Prose retains daylight or only darkens the sky instead of creating NIGHT illumination')
    if re.search(r'\b(?:no|omit|without|do not (?:apply|use|generate))\s+(?:the )?(?:requested|assigned|selected)\s+(?:effect|mechanism|lighting|night|style)(?: effect| mechanism)?\b', prose, re.I):
        raise ValueError('Prose explicitly removes the requested night lighting effect')
    if entry['style_id'] == 'short_light_trails' and re.search(
            r'\b(?:no|without|omit|avoid|do not (?:add|show|create|introduce|generate))\s+(?:any |short |visible )?light[\s-]trails?\b', prose, re.I):
        raise ValueError('Prose explicitly removes the requested short light trails')


def _assertions(prose, patterns):
    # A prohibition only negates its own clause, not a later affirmative clause.
    for clause in re.split(r'[.!?;,\n]+|\b(?:but|instead)\b', prose, flags=re.I):
        for pattern in patterns:
            for match in pattern.finditer(clause):
                if _NEGATION.search(clause[:match.end()]):
                    continue
                yield clause, match


def validate_prose_consistency(entry, source):
    """Reject clear unsupported object claims; never promote uncertain evidence."""
    features = source['features']
    present = lambda key: features[key]['status'] == 'present'
    prose = entry['appearance_prompt']+'; '+entry['lighting']
    _validate_night_effect(entry, prose)
    if not present('visible_fixed_luminaires'):
        for clause, match in _assertions(prose, _FIXTURE_ASSERTIONS):
            # Explicitly outside-frame emitters do not require a visible fixture.
            # Do not excuse a separate new fixture elsewhere in the same sentence.
            nearby = clause[max(0, match.start()-35):min(len(clause), match.end()+35)]
            if not _OFF_FRAME.search(nearby):
                raise ValueError('Prose requests an unsupported in-frame fixed fixture; describe lit surfaces from off-frame light')
    for clause, match in _assertions(prose, _WINDOW_ASSERTIONS):
        vehicle = re.search(r'\b(?:vehicle|car|bus|truck)\s+(?:\w+\s+){0,2}windows?\b', clause, re.I)
        architecture = re.search(r'\b(?:building|residential|house|apartment|facade)\b', clause, re.I)
        if not present('building_windows') and not (vehicle and not architecture):
            raise ValueError('Prose requests unsupported BUILDING window illumination; vehicle glazing is not a substitute')
        if re.search(r'\b(?:residential|house|apartment)\b', match.group()+clause[max(0, match.start()-25):match.start()], re.I) and not present('residential_windows'):
            raise ValueError('Prose requests residential window light without residential BUILDING evidence')
    if not present('shopfront_display_windows') and next(_assertions(prose, _SHOP_ASSERTIONS), None):
        raise ValueError('Prose requests unsupported shop/display lighting; glass doors and billboards are not shop evidence')
    if not present('covered_passage') and next(_assertions(prose, _PASSAGE_ASSERTIONS), None):
        raise ValueError('Prose requests an unsupported covered passage; an ordinary entrance is insufficient')
    state = entry.get('surface_state')
    if state == 'dry' and _ground_assertion(prose, _WET_GROUND_ASSERTIONS, state):
        raise ValueError('Prose requests wet ground/reflections despite the declared dry surface state')
    if state == 'wet' and _ground_assertion(prose, _DRY_GROUND_ASSERTIONS, state):
        raise ValueError('Prose requests dry ground despite the declared wet surface state')
    return entry


def validate_plan(value, variants=1):
    if type(value.get('schema_version')) is not int or value['schema_version'] != 2:
        raise ValueError('Unsupported v2 grounded-night plan schema')
    if value.get('coordinate_system') != 'normalized_0_1':
        raise ValueError('Normalized preservation envelope required')
    source = source_evidence(value.get('source_evidence'))
    if value.get('protected_regions') != [{**source['protected_regions'][0], 'box': [0., 0., 1., 1.]}]:
        raise ValueError('Only the declared non-semantic preservation envelope is permitted')
    eligible, _ = eligible_styles(source)
    entries = value.get('variants')
    if not isinstance(entries, list) or len(entries) != variants:
        raise ValueError('Wrong night variant count')
    keys, prose = set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Each variant must be an object')
        key = entry.get('style_id')
        if key not in eligible or key in keys:
            raise ValueError('Duplicate, unknown, retired or source-incompatible style')
        if entry.get('weather') != 'night':
            raise ValueError('Only condition=night is enabled')
        _english_text(entry.get('style_name'), 'original style name', 2)
        _english_text(entry.get('lighting'), 'source-grounded lighting', 3)
        _english_text(entry.get('exposure'), 'night exposure', 2)
        prompt = _english_text(entry.get('appearance_prompt'), 'original source-grounded prose', 40)
        if len(prompt.split()) > 200:
            raise ValueError('Night prose exceeds the bounded composition budget')
        if prompt.casefold() in prose:
            raise ValueError('Duplicate prompt prose')
        if entry.get('surface_state') not in ('dry', 'wet', 'mixed'):
            raise ValueError('Explicit dry/wet/mixed ground state required')
        style = STYLES[key]
        if entry['surface_state'] not in style.get('required_surface_states', ('dry', 'wet', 'mixed')):
            raise ValueError('Ground declaration conflicts with the selected wetness mechanism')
        _validate_illumination(entry, source, style)
        validate_prose_consistency(entry, source)
        keys.add(key); prose.add(prompt.casefold())
    requested = value.get('requested_style_ids')
    if not isinstance(requested, list) or len(requested) != len(keys) or set(requested) != keys:
        raise ValueError('Composer changed the scheduled style IDs')
    return value


def parse_plan(raw, source, style):
    source = source_evidence(source)
    value = parse_json(raw) if isinstance(raw, str) else copy.deepcopy(raw)
    if not isinstance(value, dict):
        raise ValueError('Night composition must be a JSON object')
    entries = value.get('variants')
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise ValueError('Exactly one night variant required')
    value.update(schema_version=2, coordinate_system='0_1000', scene_summary=source['scene_summary'],
        source_evidence=source, protected_regions=source['protected_regions'], requested_style_ids=[style])
    return validate_plan(normalize_plan(value), 1)


def render_prompt(variant, with_transient=False):
    if with_transient:
        raise ValueError('Use integrated plausible transient changes, not exact transient pairs')
    if variant.get('style_id') not in STYLES:
        raise ValueError('Cannot render a retired or unknown night mechanism')
    if variant.get('surface_state') not in ('dry', 'wet', 'mixed'):
        raise ValueError('Explicit ground declaration required')
    prompt = text(variant.get('appearance_prompt'), 'original appearance prose', 40)
    fields = []
    for source in variant['illumination']:
        prefix = ('OUTSIDE THE FRAME; do not insert an in-frame emitter' if source['origin'] == 'off_frame'
                  else source['origin'])
        fields.append(prefix+': '+source['evidence']+'. Illuminate '+source['surfaces']+
            '; direction: '+source['direction']+'; falloff: '+source['falloff']+'.')
    return (prompt+'\n\nLIGHTING SOURCE CONTRACT\n'+'\n'.join(fields)+'\n\n'+CONTRACT+
        '\nDeclared ground surface state: '+variant['surface_state']+
        '. Do not introduce an undeclared wet/dry change. Keep this photograph clearly NIGHT.')


def coverage_snapshot(states):
    snapshot = original.coverage_snapshot(states)
    retired = {key: snapshot['counts'].pop(key) for key in RETIRED_STYLES if key in snapshot['counts']}
    snapshot.update(version=2, policy_version=VERSION, retired_style_counts=retired,
        interpretation='Only the 29 active categories participate in allocation; requested/saved style is provisional until independently audited')
    return snapshot
