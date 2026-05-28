import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from install.setup_wizard import (
    WIZARD_STEPS,
    WizardState,
    build_litellm_command,
    build_mix_api_command,
    build_mix_worker_command,
    parse_slave_nodes,
    run_mix_api,
    write_nodes_file,
)


class SetupWizardStateTests(unittest.TestCase):
    def test_navigation_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            state = WizardState(settings_path)

            state.previous_step()
            self.assertEqual(state.current_step_index, 0)

            state.goto_step(10_000)
            self.assertEqual(state.current_step_index, len(WIZARD_STEPS) - 1)

            state.next_step()
            self.assertEqual(state.current_step_index, len(WIZARD_STEPS) - 1)

    def test_persist_and_resume_step_and_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            state = WizardState(settings_path)
            state.set_config("model_name", "qwen3_8b_q40")
            state.goto_step(2)
            state.persist()

            resumed = WizardState(settings_path)
            self.assertEqual(resumed.current_step_index, 2)
            self.assertEqual(resumed.settings["config"]["model_name"], "qwen3_8b_q40")


class SetupWizardCommandTests(unittest.TestCase):
    def test_build_litellm_command_uses_config_values(self):
        command = build_litellm_command(
            {
                "model_name": "llama3_2_3b_instruct_q40",
                "dllama_api_base": "http://127.0.0.1:9990/v1",
                "litellm_port": "4010",
                "api_key": "test-key",
            }
        )
        self.assertIn("llama3_2_3b_instruct_q40", command)
        self.assertIn("--port 4010", command)
        self.assertIn("--api_key test-key", command)

    def test_parse_slave_nodes_supports_ssh_targets_and_runtime_addresses(self):
        nodes = parse_slave_nodes(
            {
                "worker_port": "9999",
                "slave_nodes": "alice@10.0.0.2:10001,10.0.0.3",
            }
        )
        self.assertEqual(
            nodes,
            [
                {"ssh_target": "alice@10.0.0.2", "worker_address": "10.0.0.2:10001"},
                {"ssh_target": "10.0.0.3", "worker_address": "10.0.0.3:9999"},
            ],
        )

    def test_build_mix_worker_command_uses_remote_deploy_and_model_path(self):
        command = build_mix_worker_command(
            {
                "master_host": "10.0.0.1",
                "worker_port": "9999",
                "model_path": "models/root.m",
                "remote_model_path": "/srv/models/root.m",
                "remote_deploy_dir": "/srv/dllama",
            }
        )
        self.assertIn("cd /srv/dllama/mix/target &&", command)
        self.assertIn("python3 -m distributed_llama_python.worker --host 10.0.0.1 --port 9999", command)
        self.assertIn("--model /srv/models/root.m", command)

    def test_build_mix_api_command_uses_workers_from_slave_nodes(self):
        command = build_mix_api_command(
            {
                "dllama_api_base": "http://0.0.0.0:10090/v1",
                "model_path": "models/root.m",
                "tokenizer_path": "models/root.t",
                "worker_port": "9999",
                "slave_nodes": "alice@10.0.0.2:10001,10.0.0.3",
            },
            Path("/repo"),
        )
        self.assertEqual(command[:5], ["/repo/dllama-api", "--host", "0.0.0.0", "--port", "10090"])
        self.assertEqual(command[6], "/repo/models/root.m")
        self.assertEqual(command[8], "/repo/models/root.t")
        self.assertIn("--workers", command)
        self.assertEqual(command[-2:], ["10.0.0.2:10001", "10.0.0.3:9999"])

    def test_write_nodes_file_contains_ssh_targets_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            nodes_path = write_nodes_file(
                settings_path,
                {
                    "worker_port": "9999",
                    "slave_nodes": "alice@10.0.0.2:10001,10.0.0.3",
                },
            )
            self.assertEqual(nodes_path.read_text(encoding="utf-8"), "alice@10.0.0.2\n10.0.0.3\n")

    @patch("install.setup_wizard.subprocess.run")
    def test_run_mix_api_validates_missing_model_before_launch(self, mock_run):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            settings_path = repo_root / "settings.json"
            tokenizer_path = repo_root / "models" / "root.t"
            tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer_path.write_text("tokenizer", encoding="utf-8")

            state = WizardState(settings_path)
            state.set_config("model_path", "models/root.m")
            state.set_config("tokenizer_path", "models/root.t")

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertFalse(run_mix_api(state, repo_root))

            self.assertIn("Model file not found", stdout.getvalue())
            mock_run.assert_not_called()

    @patch("install.setup_wizard.subprocess.run")
    def test_run_mix_api_builds_binary_and_launches_when_inputs_exist(self, mock_run):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            settings_path = repo_root / "settings.json"
            model_path = repo_root / "models" / "root.m"
            tokenizer_path = repo_root / "models" / "root.t"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text("model", encoding="utf-8")
            tokenizer_path.write_text("tokenizer", encoding="utf-8")

            state = WizardState(settings_path)
            state.set_config("model_path", "models/root.m")
            state.set_config("tokenizer_path", "models/root.t")

            def run_side_effect(command, cwd=None, check=False):
                if command == ["make", "dllama-api"]:
                    (repo_root / "dllama-api").write_text("", encoding="utf-8")
                    return Mock(returncode=0)
                if command[0] == str(repo_root / "dllama-api"):
                    return Mock(returncode=0)
                raise AssertionError(f"Unexpected command: {command}")

            mock_run.side_effect = run_side_effect

            self.assertTrue(run_mix_api(state, repo_root))
            self.assertEqual(mock_run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
