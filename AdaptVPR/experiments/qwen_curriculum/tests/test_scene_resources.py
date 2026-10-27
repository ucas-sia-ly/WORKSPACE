import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace

from experiments.qwen_curriculum.run_scene_night_safe import call_budget, violation, service_memory_controls, task_cgroup, task_swap
from experiments.qwen_curriculum.scene_night_report import pair_slots


class SceneResourceTests(unittest.TestCase):
    def test_service_cgroup_limits_include_cache_and_disallow_service_swap(self):
        controls = service_memory_controls(12, 16)
        self.assertIn('MemoryHigh=12884901888', controls)
        self.assertIn('MemoryMax=17179869184', controls)
        self.assertIn('MemorySwapMax=0', controls)
        for high, maximum in ((17, 16), (0, 16), (12, float('inf'))):
            with self.assertRaises(ValueError):
                service_memory_controls(high, maximum)

    def test_report_does_not_place_transient_in_missing_appearance_slot(self):
        transient = dict(pair_id='same-source-style-seed', with_transient=True)
        pair = pair_slots([transient])[0]
        self.assertNotIn('appearance', pair)
        self.assertIs(pair['transient'], transient)
        appearance = dict(pair_id=transient['pair_id'], with_transient=False)
        self.assertEqual(set(pair_slots([transient, appearance])[0]), {'appearance', 'transient'})
        with self.assertRaises(ValueError):
            pair_slots([transient, transient])

    def test_reserved_failed_request_still_consumes_frozen_budget(self):
        self.assertEqual(call_budget(6, reserved=1, frozen_budget=6), 5)
        self.assertEqual(call_budget(2, reserved=1, frozen_budget=6), 2)
        with self.assertRaises(ValueError):
            call_budget(6, reserved=6, frozen_budget=6)
        with self.assertRaises(ValueError):
            call_budget(0, reserved=0, frozen_budget=6)

    def test_memory_gpu_and_host_reserve_independently_guarded(self):
        limits = SimpleNamespace(reserve_ram_gib=4, max_rss_gib=20, max_gpu_gib=44)
        value = dict(available_gib=8, owned_rss_gib=12, gpu_used_gib=40)
        self.assertIsNone(violation(value, limits))
        for field, amount in [('available_gib', 3.9), ('owned_rss_gib', 20.1),
                              ('gpu_used_gib', 44.1), ('new_swap_gib', 1.1)]:
            with self.subTest(field=field):
                self.assertIsNotNone(violation({**value, field: amount}, limits))

    def test_task_scope_ignores_other_process_swap_but_keeps_all_resource_limits(self):
        limits=SimpleNamespace(reserve_ram_gib=4,max_rss_gib=20,max_gpu_gib=44,
                               swap_scope='task',max_task_swap_gib=0)
        value=dict(available_gib=13,owned_rss_gib=4.6,gpu_used_gib=43.1,
                   new_swap_gib=8,task_swap_used_gib=0)
        self.assertIsNone(violation(value,limits))
        for field,amount in [('task_swap_used_gib',.01),('available_gib',3.9),
                             ('gpu_used_gib',44.1),('owned_rss_gib',20.1)]:
            with self.subTest(field=field):
                self.assertIsNotNone(violation({**value,field:amount},limits))

    def test_task_scope_does_not_silently_ignore_missing_accounting(self):
        limits=SimpleNamespace(reserve_ram_gib=4,max_rss_gib=20,max_gpu_gib=44,swap_scope='task')
        self.assertEqual(violation(dict(available_gib=13,owned_rss_gib=4,gpu_used_gib=40),limits),
                         'task swap accounting unavailable')

    def test_dedicated_cgroup_requires_finite_memory_and_swap_prohibition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'cgroups';group=root/'user.slice/prompt.service'
            group.mkdir(parents=True);proc=Path(tmp)/'proc_cgroup'
            proc.write_text('0::/user.slice/prompt.service\n')
            (group/'memory.max').write_text(str(16*1024**3))
            (group/'memory.swap.max').write_text('0')
            (group/'memory.swap.current').write_text('0')
            self.assertEqual(task_cgroup(proc,root),group)
            self.assertEqual(task_swap(group),0)
            (group/'memory.swap.max').write_text('max')
            with self.assertRaisesRegex(RuntimeError,'MemorySwapMax'):task_cgroup(proc,root)
            (group/'memory.swap.max').write_text('0');(group/'memory.max').write_text('max')
            with self.assertRaisesRegex(RuntimeError,'finite'):task_cgroup(proc,root)
            (group/'memory.swap.current').unlink()
            with self.assertRaisesRegex(RuntimeError,'accounting'):task_swap(group)

    def test_task_scope_rejects_shared_user_and_desktop_cgroups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'cgroups';root.mkdir();proc=Path(tmp)/'proc_cgroup'
            for relative in ('/user@1000.service','/app.slice/vscode.scope','/'):
                proc.write_text('0::'+relative+'\n')
                with self.assertRaisesRegex(RuntimeError,'dedicated'):task_cgroup(proc,root)


if __name__ == '__main__':
    unittest.main()
