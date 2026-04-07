"""
Unit and integration tests for MiniMax LLM provider support in LanguageModelAPIHandler.

MiniMax OpenAI-compatible API:
  - Base URL: https://api.minimax.io/v1
  - Models: MiniMax-M2.7, MiniMax-M2.7-highspeed (204K context)
  - API key env: MINIMAX_API_KEY
  - Temperature: must be in (0.0, 1.0] — 0.0 is invalid
"""

import os
import sys
from threading import Event
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy optional dependencies that may not be installed in CI
# ---------------------------------------------------------------------------
def _make_stub(name):
    mod = ModuleType(name)
    mod.__spec__ = None
    return mod


for _mod in [
    "funasr", "torch", "torchaudio", "numpy",
    "transformers", "nltk",
    "websockets", "rich", "rich.console",
    "torch.hub",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Stub specific attributes needed by the pipeline at import time
_transformers = sys.modules["transformers"]
for _attr in ["AutoModelForCausalLM", "AutoTokenizer", "pipeline", "TextIteratorStreamer"]:
    setattr(_transformers, _attr, MagicMock())

_nltk = sys.modules["nltk"]
_nltk.data = MagicMock()
_nltk.download = MagicMock()

_torch = sys.modules["torch"]
_torch.hub = MagicMock()
_torch.manual_seed = MagicMock()
_torch.compile = MagicMock()
_torch.float16 = "float16"
_torch.bfloat16 = "bfloat16"
_torch.float32 = "float32"
_torch.nn = MagicMock()
_torch.load = MagicMock()
_torch.no_grad = MagicMock()

# numpy needs array-like behaviour for VAD; mock minimally
sys.modules["numpy"].ndarray = object  # type: ignore[attr-defined]
sys.modules["numpy"].array = MagicMock()  # type: ignore[attr-defined]

_rich = sys.modules["rich"]
_rich.console = MagicMock()
sys.modules["rich.console"].Console = MagicMock()

_funasr = sys.modules["funasr"]
_funasr.AutoModel = MagicMock()

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from s2s_server_pipeline import LanguageModelAPIHandler

MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_MODEL = "MiniMax-M2.7"
MINIMAX_MODEL_HS = "MiniMax-M2.7-highspeed"


def _make_handler(model_name=MINIMAX_MODEL, model_url=MINIMAX_BASE_URL, temperature=0.1, do_sample=True, env=None):
    """Helper to create a LanguageModelAPIHandler with optional env overrides."""
    stop_event = Event()
    cur_conn_end_event = Event()
    interruption_event = Event()

    with patch.dict(os.environ, env or {"LLM_API_KEY": "test-key"}):
        with patch("s2s_server_pipeline.OpenAI"):
            handler = LanguageModelAPIHandler(
                stop_event,
                cur_conn_end_event,
                queue_in=0,
                queue_out=0,
                interruption_event=interruption_event,
                model_name=model_name,
                model_url=model_url,
                temperature=temperature,
                do_sample=do_sample,
                generate_questions=False,
            )
    return handler


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestMiniMaxHandlerInit:

    def test_model_name_stored(self):
        h = _make_handler(model_name=MINIMAX_MODEL)
        assert h.model_name == MINIMAX_MODEL

    def test_highspeed_model_name_stored(self):
        h = _make_handler(model_name=MINIMAX_MODEL_HS)
        assert h.model_name == MINIMAX_MODEL_HS

    def test_model_url_stored(self):
        h = _make_handler()
        assert h.model_url == MINIMAX_BASE_URL

    def test_temperature_preserved_when_valid(self):
        h = _make_handler(temperature=0.7, do_sample=True)
        assert h.temperature == pytest.approx(0.7)

    def test_temperature_clamped_when_zero_and_do_sample_false(self):
        """MiniMax rejects temperature=0; handler must clamp to 0.01."""
        h = _make_handler(temperature=0.1, do_sample=False)
        assert h.temperature > 0, "Temperature must be > 0 for MiniMax"
        assert h.temperature == pytest.approx(0.01)

    def test_temperature_default_stays_positive(self):
        h = _make_handler(temperature=0.1, do_sample=True)
        assert h.temperature > 0

    def test_minimax_api_key_fallback(self):
        """When LLM_API_KEY is unset, MINIMAX_API_KEY should be used."""
        stop_event = Event()
        cur_conn_end_event = Event()
        interruption_event = Event()

        captured = {}

        def fake_openai(api_key=None, base_url=None):
            captured["api_key"] = api_key
            return MagicMock()

        env = {"MINIMAX_API_KEY": "minimax-secret-key"}
        # Ensure LLM_API_KEY is not set
        env_clean = {k: v for k, v in os.environ.items() if k != "LLM_API_KEY"}
        env_clean.update(env)

        with patch.dict(os.environ, env_clean, clear=True):
            with patch("s2s_server_pipeline.OpenAI", side_effect=fake_openai):
                LanguageModelAPIHandler(
                    stop_event,
                    cur_conn_end_event,
                    queue_in=0,
                    queue_out=0,
                    interruption_event=interruption_event,
                    model_name=MINIMAX_MODEL,
                    model_url=MINIMAX_BASE_URL,
                    generate_questions=False,
                )

        assert captured.get("api_key") == "minimax-secret-key"

    def test_llm_api_key_takes_precedence_over_minimax_key(self):
        """LLM_API_KEY should override MINIMAX_API_KEY when both are set."""
        captured = {}

        def fake_openai(api_key=None, base_url=None):
            captured["api_key"] = api_key
            return MagicMock()

        env = {"LLM_API_KEY": "explicit-key", "MINIMAX_API_KEY": "minimax-key"}
        with patch.dict(os.environ, env):
            with patch("s2s_server_pipeline.OpenAI", side_effect=fake_openai):
                stop_event = Event()
                cur_conn_end_event = Event()
                interruption_event = Event()
                LanguageModelAPIHandler(
                    stop_event,
                    cur_conn_end_event,
                    queue_in=0,
                    queue_out=0,
                    interruption_event=interruption_event,
                    model_name=MINIMAX_MODEL,
                    model_url=MINIMAX_BASE_URL,
                    generate_questions=False,
                )

        assert captured.get("api_key") == "explicit-key"

    def test_non_minimax_url_not_affected(self):
        """Non-MiniMax providers should not be affected by MiniMax logic."""
        h = _make_handler(
            model_name="deepseek-chat",
            model_url="https://api.deepseek.com",
            temperature=0.1,
            do_sample=False,
            env={"LLM_API_KEY": "deepseek-key"},
        )
        # do_sample=False → temperature=0 is fine for non-MiniMax providers
        assert h.temperature == 0

    def test_minimax_io_url_detected(self):
        """Both api.minimax.io URL variants should trigger MiniMax mode."""
        for url in [MINIMAX_BASE_URL, "https://api.minimax.io/v1"]:
            h = _make_handler(model_url=url, do_sample=False)
            assert h.temperature == pytest.approx(0.01), f"Expected clamp for URL: {url}"


# ---------------------------------------------------------------------------
# Integration test (requires MINIMAX_API_KEY in environment)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set — skipping live integration test",
)
class TestMiniMaxIntegration:

    def test_basic_inference(self):
        """Live call to MiniMax API — verifies end-to-end connectivity."""
        stop_event = Event()
        cur_conn_end_event = Event()
        interruption_event = Event()

        lm = LanguageModelAPIHandler(
            stop_event,
            cur_conn_end_event,
            queue_in=0,
            queue_out=0,
            interruption_event=interruption_event,
            model_name=MINIMAX_MODEL,
            model_url=MINIMAX_BASE_URL,
            generate_questions=False,
            temperature=0.1,
        )

        inputs = {
            "data": "Say hello in one sentence.",
            "user_input_count": 1,
            "uid": "test_uid",
            "audio_input": False,
        }

        generator = lm.process(inputs)
        outputs = "".join(t["answer_text"] for t in generator)
        assert isinstance(outputs, str) and len(outputs) > 0, "MiniMax returned empty response"

    def test_highspeed_model(self):
        """Live call using MiniMax-M2.7-highspeed model."""
        stop_event = Event()
        cur_conn_end_event = Event()
        interruption_event = Event()

        lm = LanguageModelAPIHandler(
            stop_event,
            cur_conn_end_event,
            queue_in=0,
            queue_out=0,
            interruption_event=interruption_event,
            model_name=MINIMAX_MODEL_HS,
            model_url=MINIMAX_BASE_URL,
            generate_questions=False,
            temperature=0.1,
        )

        inputs = {
            "data": "What is 2 + 2?",
            "user_input_count": 1,
            "uid": "test_uid_hs",
            "audio_input": False,
        }

        generator = lm.process(inputs)
        outputs = "".join(t["answer_text"] for t in generator)
        assert isinstance(outputs, str) and len(outputs) > 0


if __name__ == "__main__":
    # Quick smoke test for the integration test
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
