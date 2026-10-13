from pathlib import Path
import sys
import tempfile
import json
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, mine_all_cities as all_cities
from experiments.qwen_curriculum import all_city_campaign, finetune_all_cities, finetune_pretrained, plan, run
from PIL import Image


class AllCitySelectionTests(unittest.TestCase):
    def test_discovers_every_csv_and_rejects_incomplete_city_installation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "Dataframes").mkdir()
            for city in ("A", "B", "C"):
                (root / "Dataframes" / f"{city}.csv").touch()
                (root / "Images" / city).mkdir(parents=True)
            self.assertEqual(all_cities.discover_cities(root, 3), ["A", "B", "C"])
            self.assertEqual(all_cities.missing_cities(["A", "B", "C"], [{"city": "A"}]), ["B", "C"])
            with self.assertRaisesRegex(ValueError, "Expected 23"):
                all_cities.discover_cities(root)
            (root / "Images/B").rmdir()
            with self.assertRaisesRegex(ValueError, "Images"):
                all_cities.discover_cities(root, 3)

    def test_historical_city_labels_cannot_be_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            all_cities.missing_cities(["A"], [{"city": "Unknown"}])

    def test_selects_full_capacity_above_previous_1000_budget_without_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            city_dir = directory / "cities/New"
            cache = city_dir / "cache"
            cache.mkdir(parents=True)
            for name in ("mining_config.json", "summary.json"):
                (city_dir / name).write_text('{}')
            for name in ("source_scores.npz", "scores_complete.json", "descriptor_complete.json"):
                (cache / name).write_text('proof')
            config = {"checkpoint": str(directory / "teacher"), "real_data": str(directory),
                "backbone_repo": str(directory), "selection": {"seed": 42, "min_positive_similarity": .15,
                "min_quantile": .70, "max_quantile": .975, "max_sources_per_place": 2,
                "near_negative_radius_m": 25}}
            summary = {"selection_audit": {"cities": {"New": {"capacity_after_place_cap": 4321}}},
                       "descriptor_cache": str(cache)}
            rows = [{"source_path": f"/{i}", "source_sha256": f"{i:064x}", "city": "New"} for i in range(4321)]
            with patch.object(all_cities.mine_sources, "mine", return_value=summary) as mine, \
                    patch.object(all_cities.extend_plan, "_cached_selection", return_value=(rows, {})) as select:
                actual, saved = all_cities.mine_city(directory, "New", config)
                self.assertEqual(actual, rows)
                self.assertEqual(select.call_args.args[1], 4321)
                self.assertEqual(saved["selected_sources"], 4321)
                # A completed city is reused without extracting a model again.
                all_cities.mine_city(directory, "New", config)
                self.assertEqual(mine.call_count, 1)
            (city_dir / "all_sources.jsonl").write_text('changed')
            with self.assertRaisesRegex(ValueError, "changed"):
                all_cities.mine_city(directory, "New", config)

    def test_global_domains_survive_bounded_partition_and_old_jobs_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mining, parent, output = [root / n for n in ('mining', 'parent', 'output')]
            mining.mkdir()
            parent.mkdir()
            sources = []
            for i in range(1004):
                image = root / f'{i}.png'
                Image.new('RGB', (8, 6), (i % 256, i // 256, 17)).save(image)
                sources.append({'source_path': str(image), 'source_sha256': common.file_sha256(image),
                    'city': 'A' if i < 2 else 'B' if i % 2 else 'C', 'place_id': i // 2})
            common.write_jsonl(parent / 'sources.jsonl', sources[:2])
            _, old = plan.build_plan(SimpleNamespace(sources=parent / 'sources.jsonl', output_dir=parent,
                num_images=2, seed=42, domain_weights=plan.DEFAULT_DOMAIN_WEIGHTS, domain_stats=None))
            common.write_jsonl(parent / 'combined_plan.jsonl', old)
            for name in ('results.jsonl', 'generated.jsonl'):
                common.write_jsonl(parent / name, old)
            additional = mining / 'additional_sources.jsonl'
            common.write_jsonl(additional, sources[2:])
            for name in ('request_config.json', 'all_city_selection.json'):
                (mining / name).write_text('{}')
            selection = {'additional_sources': str(additional)}
            request = {'parent': str(parent), 'cities': ['A', 'B', 'C']}
            parent_config = {'combined_plan': str(parent / 'combined_plan.jsonl'), 'fingerprint': 'parent',
                'reference_execution': {}, 'runtime_profile': 'resident_bf16_48g'}
            with patch.object(all_cities, 'load_selection', return_value=(sources[2:], selection, request)), \
                    patch.object(all_city_campaign.scale_campaign, 'load_collection', return_value=(old, parent_config)):
                cfg = all_city_campaign.prepare(output, mining)
                jobs = common.read_jsonl(output / 'combined_plan.jsonl')
                self.assertEqual(jobs[:2], old)
                self.assertEqual([s['max_calls'] for s in cfg['stages']], [1000, 2])
                self.assertEqual(Counter(r['condition'] for r in jobs), cfg['domain_quotas'])
                self.assertEqual(len({r['source_sha256'] for r in jobs}), 1004)
                for stage in cfg['stages']:
                    _, staged_jobs = run.load_plan(Path(stage['run_dir']))
                    self.assertLessEqual(len(staged_jobs), 1000)

    def test_training_requires_approved_images_in_every_city_and_original_init(self):
        rows = [{'city': city, 'passed': True, 'eligible_for_training': True, 'weather_ok': True}
                for city in ('A', 'B', 'C')]
        self.assertEqual(finetune_all_cities.accepted_cities(rows, ['A','B','C']), rows)
        rows[-1]['passed'] = False
        with self.assertRaisesRegex(ValueError, 'C'):
            finetune_all_cities.accepted_cities(rows, ['A','B','C'])
        cfg = {'real_data': '/real', 'cities': [str(i) for i in range(23)],
            'accepted_manifest': '/accepted.jsonl', 'training_dir': '/training', 'epochs': 10,
            'backbone_repo': '/repo', 'initial_checkpoint': '/original/dino_salad.ckpt'}
        command = finetune_pretrained.training_command(cfg)
        cities = command[command.index('--cities') + 1:command.index('--synthetic-manifest')]
        self.assertEqual(cities, cfg['cities'])
        self.assertEqual(command[command.index('--init-checkpoint') + 1], cfg['initial_checkpoint'])
        self.assertNotIn('--backbone-weights', command)
        self.assertFalse(any('reliability' in arg for arg in command))

    def test_disk_preflight_prevents_any_generator_start_or_http_call(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'summary.json').write_text(json.dumps({'state': 'prepared'}))
            with patch.object(all_city_campaign, 'publish', return_value={
                    'available_free_bytes': 100, 'required_free_bytes_estimate': 5000}), \
                    patch.object(all_city_campaign.subprocess, 'run') as start, \
                    patch.object(all_city_campaign.requests, 'Session') as http:
                with self.assertRaisesRegex(RuntimeError, 'Insufficient disk'):
                    all_city_campaign.execute(directory, {})
                start.assert_not_called()
                http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
