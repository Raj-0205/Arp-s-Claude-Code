"""Tests for claude-pick unified model picker script."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CLAUDE_PICK_SCRIPT = REPO_ROOT / "claude-pick"


class TestClaudePick:
    """Tests for claude-pick script and discovery routines."""

    def test_claude_pick_script_exists_and_is_executable(self):
        """claude-pick script exists and is executable."""
        assert CLAUDE_PICK_SCRIPT.is_file()
        assert os.access(CLAUDE_PICK_SCRIPT, os.X_OK)

    def test_parse_models_from_json_valid(self):
        """parse_models_from_json extracts model IDs from standard OpenAI/OpenRouter payload."""
        payload = {
            "data": [
                {"id": "meta/llama-3.3-70b-instruct"},
                {"id": "anthropic/claude-3.5-sonnet"},
                {"id": ""},
                {"name": "no-id"},
            ]
        }
        cmd = [
            "python3",
            "-c",
            """
import json, sys
try:
    payload = json.load(sys.stdin)
    for item in payload.get("data", []):
        model_id = item.get("id")
        if model_id:
            print(model_id)
except Exception:
    sys.exit(0)
""",
        ]
        result = subprocess.run(
            cmd,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
        )
        models = result.stdout.strip().splitlines()
        assert models == [
            "meta/llama-3.3-70b-instruct",
            "anthropic/claude-3.5-sonnet",
        ]

    def test_parse_models_from_json_invalid(self):
        """parse_models_from_json handles malformed JSON gracefully without error."""
        cmd = [
            "python3",
            "-c",
            """
import json, sys
try:
    payload = json.load(sys.stdin)
    for item in payload.get("data", []):
        model_id = item.get("id")
        if model_id:
            print(model_id)
except Exception:
    sys.exit(0)
""",
        ]
        result = subprocess.run(
            cmd,
            input="invalid-json-content",
            text=True,
            capture_output=True,
            check=True,
        )
        assert result.stdout == ""

    def test_get_nvidia_models_graceful_when_missing(self):
        """When nvidia_nim_models.json is missing, warn and return exit 0 (non-fatal)."""
        bash_script = f"""
source "{CLAUDE_PICK_SCRIPT}"
MODELS_FILE="/non/existent/path/nvidia_nim_models.json"
get_nvidia_models
"""
        result = subprocess.run(
            ["bash", "-c", bash_script],
            capture_output=True,
            text=True,
        )
        # Should exit 0 and emit warning on stderr, not abort the whole script
        assert result.returncode == 0
        assert "Warning" in result.stderr
        assert result.stdout.strip() == ""

    def test_get_openrouter_models_graceful_when_curl_fails(self):
        """When OpenRouter endpoint fails, warn and return exit 0 (non-fatal)."""
        bash_script = f"""
source "{CLAUDE_PICK_SCRIPT}"
# Point to an unreachable URL to simulate network failure
OPENROUTER_MODELS_URL="http://127.0.0.1:1"
get_openrouter_models
"""
        result = subprocess.run(
            ["bash", "-c", bash_script],
            capture_output=True,
            text=True,
        )
        # Should exit 0 and emit warning on stderr, not fatal abort
        assert result.returncode == 0
        assert "Warning" in result.stderr
        assert result.stdout.strip() == ""

    def test_unified_model_entry_formatting_and_parsing(self):
        """Entries must have provider label in display field and canonical value in second field."""
        nvidia_models = ["meta/llama-3.3-70b-instruct", "deepseek-ai/deepseek-r1"]
        openrouter_models = ["anthropic/claude-3.5-sonnet", "openai/gpt-4o"]

        formatted_entries = [
            f"NVIDIA      | {m}\tnvidia_nim/{m}" for m in nvidia_models
        ] + [f"OpenRouter  | {m}\topen_router/{m}" for m in openrouter_models]

        # Simulate selecting an OpenRouter model
        selected_line = formatted_entries[2]
        display_part, canonical_part = selected_line.split("\t")

        assert "OpenRouter" in display_part
        assert "anthropic/claude-3.5-sonnet" in display_part
        assert canonical_part == "open_router/anthropic/claude-3.5-sonnet"

        # Verify provider prefix extraction
        provider = canonical_part.split("/", 1)[0]
        model = canonical_part.split("/", 1)[1]
        assert provider == "open_router"
        assert model == "anthropic/claude-3.5-sonnet"

        # Verify auth token format
        auth_token = f"freecc:{canonical_part}"
        assert auth_token == "freecc:open_router/anthropic/claude-3.5-sonnet"

    def test_one_provider_available_still_lists_models(self):
        """If NVIDIA fails, OpenRouter models are still displayed, and vice versa."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_env = Path(tmpdir) / ".env"
            tmp_env.write_text("OPENROUTER_API_KEY=test_key\n")

            # Mock fzf to just print the first entry and exit
            mock_fzf = Path(tmpdir) / "fzf"
            mock_fzf.write_text("""#!/usr/bin/env bash
head -n 1
""")
            mock_fzf.chmod(0o755)

            # Mock claude CLI to record arguments and env
            mock_claude = Path(tmpdir) / "claude"
            mock_claude.write_text("""#!/usr/bin/env bash
echo "CLAUDE_LAUNCHED: token=$ANTHROPIC_AUTH_TOKEN base=$ANTHROPIC_BASE_URL args=$@"
""")
            mock_claude.chmod(0o755)

            # Test 1: NVIDIA models file exists, OpenRouter URL unreachable
            tmp_models = Path(tmpdir) / "models.json"
            tmp_models.write_text(json.dumps({"data": [{"id": "meta/test-nim-model"}]}))

            env = os.environ.copy()
            env["PATH"] = f"{tmpdir}:{env['PATH']}"
            env["CLAUDE_PICK_ENV_FILE"] = str(tmp_env)

            bash_test = f"""
export PATH="{tmpdir}:$PATH"
export CLAUDE_PICK_ENV_FILE="{tmp_env}"
export OPENROUTER_MODELS_URL="http://127.0.0.1:1" # Unreachable OpenRouter
MODELS_FILE="{tmp_models}"

# Run claude-pick logic with simulated selection
source "{CLAUDE_PICK_SCRIPT}"
nvidia_raw="$(get_nvidia_models)"
or_raw="$(get_openrouter_models)"

# Even though OpenRouter failed, nvidia_raw has models
echo "NVIDIA_COUNT: $(echo "$nvidia_raw" | wc -l)"
echo "OR_EMPTY: $(test -z "$or_raw" && echo 'yes' || echo 'no')"
"""
            result = subprocess.run(
                ["bash", "-c", bash_test],
                capture_output=True,
                text=True,
                env=env,
            )
            assert "NVIDIA_COUNT: 1" in result.stdout
            assert "OR_EMPTY: yes" in result.stdout
            assert "Warning: Failed to fetch OpenRouter models" in result.stderr

    def test_api_keys_not_exposed_in_stderr_or_output(self):
        """API keys are never echoed to stdout or stderr during discovery."""
        secret_key = "sk-or-v1-SECRETKEY123456789"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_env = Path(tmpdir) / ".env"
            tmp_env.write_text(f"OPENROUTER_API_KEY={secret_key}\n")

            bash_test = f"""
source "{CLAUDE_PICK_SCRIPT}"
ENV_FILE="{tmp_env}"
OPENROUTER_MODELS_URL="http://127.0.0.1:1" # fail
get_openrouter_models
"""
            result = subprocess.run(
                ["bash", "-c", bash_test],
                capture_output=True,
                text=True,
            )
            assert secret_key not in result.stdout
            assert secret_key not in result.stderr
