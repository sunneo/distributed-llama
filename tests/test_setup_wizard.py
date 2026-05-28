import tempfile
import unittest
from pathlib import Path

from install.setup_wizard import WIZARD_STEPS, WizardState, build_litellm_command


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


if __name__ == "__main__":
    unittest.main()
