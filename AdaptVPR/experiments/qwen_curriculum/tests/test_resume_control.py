from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import resume_control as control


def state(unit, active="inactive", pid=0):
    return {"unit": unit, "active_state": active, "sub_state": "dead", "main_pid": pid}


def command_result(code=0, stderr=""):
    return SimpleNamespace(returncode=code, stdout="", stderr=stderr)


class ResumeControlTests(unittest.TestCase):
    def test_night_first_controller_is_selected_for_the_new_place_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "outputs/weather_v3/state"
            output.mkdir(parents=True)
            (output / "plan_config.json").write_text('{"stage":"place_coverage_weather_v3"}')
            path = control.install_systemd_override(output, user_unit_dir=Path(temp) / "user", python="/env/bin/python")
            self.assertIn(str(Path(control.__file__).with_name("night_first_v3.py").resolve()), path.read_text())
            self.assertIn(str(output), path.read_text())

    def test_night_first_binding_keeps_explicit_state_directory_for_status_and_resume(self):
        import json
        with tempfile.TemporaryDirectory() as temp, patch.object(control.Path, "home", return_value=Path(temp)):
            output = Path(temp) / "outputs/weather_v3/state"
            output.mkdir(parents=True)
            (output / "plan_config.json").write_text(json.dumps({"stage": "place_coverage_weather_v3"}))
            (output / "summary.json").write_text(json.dumps({"state": "paused", "saved": 910}))
            override = control.install_systemd_override(output, python="/env/bin/python")
            text = override.read_text()
            self.assertIn(str(Path(control.__file__).with_name("night_first_v3.py").resolve()), text)
            self.assertIn('"--output-dir" "' + str(output) + '"', text)
            self.assertIn("StandardOutput=append:" + str(output / "coordinator.log"), text)
            self.assertEqual(control.configured_output_dir(), output)
            with patch.object(control, "unit_state", side_effect=lambda unit: state(unit)), \
                    patch.object(control, "systemctl") as ctl:
                report = control.status()
            self.assertEqual(report["output_dir"], str(output))
            self.assertEqual(report["summary"]["saved"], 910)
            ctl.assert_not_called()
            with patch.object(control, "unit_state", side_effect=[state(control.COORDINATOR_UNIT),
                    state(control.COORDINATOR_UNIT, "active", 91)]), \
                    patch.object(control, "systemctl", return_value=command_result()) as ctl:
                report = control.resume(output)
            self.assertEqual(report["output_dir"], str(output))
            ctl.assert_called_once_with("start", control.COORDINATOR_UNIT)

    def test_pause_stops_client_then_independent_qwen_and_checks_both(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(control, "systemctl", return_value=command_result()) as ctl, \
                patch.object(control, "unit_state", side_effect=lambda unit: state(unit)) as inspect:
            report = control.pause(Path(temp))
            self.assertEqual(ctl.call_args_list[0].args, ("stop", control.COORDINATOR_UNIT))
            self.assertEqual(ctl.call_args_list[1].args, ("stop", control.QWEN_UNIT))
            self.assertEqual([call.args[0] for call in inspect.call_args_list],
                             [control.COORDINATOR_UNIT, control.QWEN_UNIT])
            self.assertTrue(report["resources_released"])
            self.assertEqual(report["action"], "paused")

    def test_stop_failure_still_stops_qwen_but_does_not_claim_release(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(control, "systemctl", side_effect=[command_result(1, "access denied"), command_result()]) as ctl, \
                patch.object(control, "unit_state", side_effect=lambda unit: state(unit)):
            with self.assertRaisesRegex(RuntimeError, "access denied"):
                control.pause(Path(temp))
            self.assertEqual(ctl.call_args_list[-1].args, ("stop", control.QWEN_UNIT))

    def test_inactive_main_pid_must_be_zero_to_claim_release(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(control, "systemctl", return_value=command_result()), \
                patch.object(control, "unit_state", side_effect=lambda unit: state(unit, pid=91 if unit == control.QWEN_UNIT else 0)):
            with self.assertRaisesRegex(RuntimeError, "Still running"):
                control.pause(Path(temp))

    def test_resume_is_idempotent_for_already_running_task(self):
        with patch.object(control, "unit_state", return_value=state(control.COORDINATOR_UNIT, "active", 91)), \
                patch.object(control, "systemctl") as ctl:
            report = control.resume()
            self.assertEqual(report["action"], "already_running")
            ctl.assert_not_called()

    def test_resume_starts_only_after_coordinator_is_fully_stopped(self):
        with patch.object(control, "unit_state", side_effect=[state(control.COORDINATOR_UNIT), state(control.COORDINATOR_UNIT, "active", 91)]), \
                patch.object(control, "systemctl", return_value=command_result()) as ctl:
            report = control.resume()
            ctl.assert_called_once_with("start", control.COORDINATOR_UNIT)
            self.assertEqual(report["action"], "resumed")
        with patch.object(control, "unit_state", return_value=state(control.COORDINATOR_UNIT, "deactivating", 91)), \
                patch.object(control, "systemctl") as ctl:
            with self.assertRaisesRegex(RuntimeError, "finished stopping"):
                control.resume()
            ctl.assert_not_called()

    def test_status_is_read_only_and_loads_summary(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(control, "unit_state", side_effect=lambda unit: state(unit)), \
                patch.object(control, "systemctl") as ctl:
            directory = Path(temp)
            (directory / "summary.json").write_text('{"state": "stopped", "generated_images": 17}')
            report = control.status(directory)
            self.assertEqual(report["summary"]["generated_images"], 17)
            ctl.assert_not_called()

    def test_systemd_override_preserves_base_working_directory_and_env(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(control, "systemctl") as ctl:
            output = Path(temp) / "results with spaces"
            path = control.install_systemd_override(output, user_unit_dir=Path(temp) / "user", python="/env/bin/python")
            text = path.read_text()
            self.assertIn("campaign.py", text)
            self.assertIn('"--output-dir" "' + str(output.resolve()) + '"', text)
            self.assertIn("ExecStart=\nExecStart=", text)
            self.assertIn("ExecStopPost=/usr/bin/systemctl --user --no-block stop", text)
            self.assertIn("KillMode=control-group", text)
            self.assertIn("StandardOutput=append:" + str(output / "coordinator.log") + "\n", text)
            self.assertNotIn('StandardOutput=append:"', text)
            self.assertNotIn("WorkingDirectory=", text)
            self.assertNotIn("Environment=", text)
            self.assertNotIn("SIGSTOP", text)
            ctl.assert_not_called()

    def test_unit_state_rejects_unknown_or_incomplete_services(self):
        result = SimpleNamespace(returncode=0, stdout="LoadState=not-found\nActiveState=inactive\nMainPID=0\n", stderr="")
        with patch.object(control, "systemctl", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "not loaded"):
                control.unit_state(control.QWEN_UNIT)
        result.stdout = "LoadState=loaded\nActiveState=inactive\n"
        with patch.object(control, "systemctl", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                control.unit_state(control.QWEN_UNIT)

    def test_other_output_directory_cannot_silently_start_default_service(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(control.Path, "home", return_value=Path(temp)), \
                patch.object(control, "systemctl") as ctl:
            with self.assertRaisesRegex(RuntimeError, "configured for"):
                control.resume(Path(temp) / "other-output")
            ctl.assert_not_called()

    def test_default_status_and_resume_follow_migrated_service_binding(self):
        import json
        with tempfile.TemporaryDirectory() as temp, patch.object(control.Path, "home", return_value=Path(temp)):
            output = Path(temp) / "state"
            output.mkdir()
            (output / "migration_config.json").write_text('{}')
            override = control.install_systemd_override(output, python="/env/bin/python")
            self.assertIn("campaign.py", override.read_text())
            (output / "summary.json").write_text(json.dumps({"generated_images": 16892, "quality_version": 2}))
            with patch.object(control, "unit_state", side_effect=lambda unit: state(unit)):
                report = control.status()
            self.assertEqual(report["output_dir"], str(output))
            self.assertEqual(report["summary"]["quality_version"], 2)
            with patch.object(control, "unit_state", side_effect=[state(control.COORDINATOR_UNIT),
                    state(control.COORDINATOR_UNIT, "active", 91)]), \
                    patch.object(control, "systemctl", return_value=command_result()) as ctl:
                report = control.resume()
            self.assertEqual(report["output_dir"], str(output))
            ctl.assert_called_once_with("start", control.COORDINATOR_UNIT)

    def test_controller_selection_does_not_depend_on_legacy_markers(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "state"
            output.mkdir()
            (output / "migration_config.json").write_text('{}')
            (output / "organization_config.json").write_text('{}')
            path = control.install_systemd_override(output, user_unit_dir=Path(temp) / "user", python="/env/bin/python")
            self.assertIn("campaign.py", path.read_text())
            self.assertIn("KillMode=control-group", path.read_text())


if __name__ == "__main__":
    unittest.main()
