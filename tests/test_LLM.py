"""Unit tests for topologicpy.LLM.

These tests avoid real provider SDKs and network calls by injecting fake modules
into sys.modules before provider methods import them.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from topologicpy.LLM import LLM


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _fake_openai_response(text='{"ok": true}', prompt_tokens=3, completion_tokens=5):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _install_fake_openai(monkeypatch, behavior=None, text='{"ok": true}', model_ids=None):
    """Install a minimal fake openai module and return captured state."""
    state = {"clients": [], "requests": []}
    model_ids = model_ids or ["gpt-a", "gpt-b"]

    class FakeOpenAI:
        def __init__(self, **kwargs):
            state["clients"].append(kwargs)

            def create(**request):
                state["requests"].append(dict(request))
                if behavior is not None:
                    result = behavior(request, state)
                    if isinstance(result, BaseException):
                        raise result
                    if result is not None:
                        return result
                return _fake_openai_response(text=text)

            def list_models():
                return SimpleNamespace(data=[SimpleNamespace(id=m) for m in model_ids])

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
            self.models = SimpleNamespace(list=list_models)

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    return state


def _install_fake_requests(monkeypatch):
    """Install a minimal fake requests module and return captured state."""
    state = {"gets": [], "posts": []}

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    def get(url, timeout=None):
        state["gets"].append({"url": url, "timeout": timeout})
        return FakeResponse({"models": [{"name": "llama3.1"}, {"name": "mistral"}, {}]})

    def post(url, json=None, timeout=None):
        state["posts"].append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse({
            "message": {"content": "ollama response"},
            "prompt_eval_count": 2,
            "eval_count": 4,
        })

    fake_requests = types.ModuleType("requests")
    fake_requests.get = get
    fake_requests.post = post
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    return state


def _install_fake_anthropic(monkeypatch):
    state = {"clients": [], "requests": []}

    class FakeAnthropic:
        def __init__(self, api_key=None, timeout=None):
            state["clients"].append({"api_key": api_key, "timeout": timeout})

            def create(**request):
                state["requests"].append(dict(request))
                usage = SimpleNamespace(input_tokens=7, output_tokens=11)
                return SimpleNamespace(content=[SimpleNamespace(text="anthropic response")], usage=usage)

            self.messages = SimpleNamespace(create=create)

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    return state


def _install_fake_google(monkeypatch, fail_new_sdk=False):
    state = {"new_clients": [], "new_requests": [], "old_config": [], "old_requests": []}

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    types_module = types.ModuleType("google.genai.types")

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    class FakeNewClient:
        def __init__(self, api_key=None):
            state["new_clients"].append({"api_key": api_key})
            self.models = self

        def generate_content(self, model=None, contents=None, config=None):
            if fail_new_sdk:
                raise RuntimeError("new SDK unavailable")
            state["new_requests"].append({
                "model": model,
                "contents": contents,
                "config": getattr(config, "kwargs", {}),
            })
            return SimpleNamespace(text="google new response")

    def configure(api_key=None):
        state["old_config"].append({"api_key": api_key})

    class FakeGenerativeModel:
        def __init__(self, model):
            self.model = model

        def generate_content(self, prompt_text, generation_config=None):
            state["old_requests"].append({
                "model": self.model,
                "prompt_text": prompt_text,
                "generation_config": dict(generation_config or {}),
            })
            return SimpleNamespace(text="google old response")

    genai_module.Client = FakeNewClient
    genai_module.types = types_module
    types_module.GenerateContentConfig = GenerateContentConfig

    generativeai_module = types.ModuleType("google.generativeai")
    generativeai_module.configure = configure
    generativeai_module.GenerativeModel = FakeGenerativeModel

    google_module.genai = genai_module
    google_module.generativeai = generativeai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)
    monkeypatch.setitem(sys.modules, "google.generativeai", generativeai_module)
    return state


def test_by_parameters_provider_aliases_and_value_coercion():
    llm = LLM.ByParameters(
        provider="mystery_provider",
        model="  custom-model  ",
        baseURL=" http://localhost:9999/v1/ ",
        temperature="not-a-number",
        maxOutputTokens="0",
        timeout="bad",
        silent=True,
    )

    assert llm.provider == "openai-compatible"
    assert llm.model == "custom-model"
    assert llm.baseURL == "http://localhost:9999/v1"
    assert llm.temperature is None
    assert llm.maxOutputTokens is None
    assert llm.timeout == 60
    assert "apiKey=None" in repr(llm)

    assert LLM.ByParameters(provider="chatgpt", silent=True).provider == "openai"
    assert LLM.ByParameters(provider="lm-studio", silent=True).provider == "lmstudio"
    assert LLM.ProviderInfo("gemini", silent=True)["mode"] == "google"
    assert LLM.ProviderInfo("not-real", silent=True) == {}




def test_response_rejects_missing_llm_and_missing_model_without_importing_sdks():
    assert LLM.Response(None, "hello", silent=True)["error_type"] == "unknown_provider"

    llm = LLM.ByParameters(provider="huggingface", model=None, silent=True)
    response = LLM.Response(llm, "hello", silent=True)
    assert response["ok"] is False
    assert response["error_type"] == "model_error"
    assert "No model" in response["message"]


def test_openai_frontier_model_uses_max_completion_tokens_and_omits_temperature(monkeypatch):
    state = _install_fake_openai(monkeypatch, text='{"ok": true}')
    llm = LLM.ByParameters(
        provider="openai",
        model="gpt-5-mini",
        apiKey="sk-test",
        temperature=0.7,
        maxOutputTokens=123,
        timeout=9,
        systemPrompt="Be concise.",
        silent=True,
    )

    response = LLM.Response(llm, "Return JSON.", silent=True)

    assert response["ok"] is True
    assert response["json"] == {"ok": True}
    assert response["usage"] == {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}
    assert state["clients"] == [{"api_key": "sk-test"}]
    request = state["requests"][0]
    assert request["model"] == "gpt-5-mini"
    assert request["max_completion_tokens"] == 123
    assert "max_tokens" not in request
    assert "temperature" not in request
    assert request["timeout"] == 9
    assert request["messages"][0]["role"] == "system"


def test_openai_non_frontier_model_uses_temperature_and_max_tokens(monkeypatch):
    state = _install_fake_openai(monkeypatch, text="plain text")
    llm = LLM.ByParameters(provider="openai", model="gpt-4o-mini", apiKey="sk", temperature=0.4, maxOutputTokens=44, silent=True)

    response = LLM.Response(llm, "hello", silent=True)

    assert response["ok"] is True
    request = state["requests"][0]
    assert request["temperature"] == 0.4
    assert request["max_tokens"] == 44
    assert "max_completion_tokens" not in request
    assert response["json"] is None


def test_openai_compatible_retries_max_tokens_to_max_completion_tokens(monkeypatch):
    def behavior(request, state):
        if len(state["requests"]) == 1:
            assert "max_tokens" in request
            return ValueError("Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead.")
        assert "max_completion_tokens" in request
        assert "max_tokens" not in request
        return _fake_openai_response(text="retry ok")

    state = _install_fake_openai(monkeypatch, behavior=behavior)
    llm = LLM.ByParameters(provider="lmstudio", model="local-model", temperature=0.1, maxOutputTokens=55, silent=True)

    response = LLM.Response(llm, "hello", silent=True)

    assert response["ok"] is True
    assert response["text"] == "retry ok"
    assert len(state["requests"]) == 2
    assert state["requests"][0]["max_tokens"] == 55
    assert state["requests"][1]["max_completion_tokens"] == 55


def test_openai_compatible_retries_max_completion_tokens_to_max_tokens(monkeypatch):
    def behavior(request, state):
        if len(state["requests"]) == 1:
            assert "max_completion_tokens" in request
            return ValueError("Unsupported parameter: 'max_completion_tokens'. Use 'max_tokens' instead.")
        assert "max_tokens" in request
        assert "max_completion_tokens" not in request
        return _fake_openai_response(text="reverse retry ok")

    state = _install_fake_openai(monkeypatch, behavior=behavior)
    llm = LLM.ByParameters(provider="openai", model="gpt-5-mini", apiKey="sk", maxOutputTokens=22, silent=True)

    response = LLM.Response(llm, "hello", silent=True)

    assert response["ok"] is True
    assert response["text"] == "reverse retry ok"
    assert len(state["requests"]) == 2
    assert state["requests"][1]["max_tokens"] == 22


def test_openai_compatible_retries_by_removing_temperature(monkeypatch):
    def behavior(request, state):
        if len(state["requests"]) == 1:
            assert "temperature" in request
            return RuntimeError("unsupported parameter: temperature is not supported by this model")
        assert "temperature" not in request
        return _fake_openai_response(text="temperature retry ok")

    state = _install_fake_openai(monkeypatch, behavior=behavior)
    llm = LLM.ByParameters(provider="openai-compatible", model="remote-model", temperature=0.9, maxOutputTokens=10, silent=True)

    response = LLM.Response(llm, "hello", silent=True)

    assert response["ok"] is True
    assert response["text"] == "temperature retry ok"
    assert len(state["requests"]) == 2
    assert "temperature" not in state["requests"][1]


def test_prompt_json_and_test_public_wrappers_use_response(monkeypatch):
    _install_fake_openai(monkeypatch, text='Here is JSON: {"answer": 42, "items": [1, 2]} trailing')
    llm = LLM.ByParameters(provider="openai", model="gpt-4o-mini", apiKey="sk", silent=True)

    assert "Here is JSON" in LLM.Prompt(llm, "hello", silent=True)
    assert LLM.JSON(llm, "return json", schema={"type": "object"}, silent=True) == {"answer": 42, "items": [1, 2]}

    test_response = LLM.Test(llm, silent=True)
    assert test_response["ok"] is True
    assert test_response["details"]["ok"] is True


def test_models_for_openai_compatible_providers_and_client_defaults(monkeypatch):
    state = _install_fake_openai(monkeypatch, model_ids=["a", "b", None])

    lmstudio = LLM.ByParameters(provider="lmstudio", model="local", silent=True)
    assert LLM.Models(lmstudio, silent=True) == ["a", "b"]
    assert state["clients"][-1] == {"api_key": "lm-studio", "base_url": "http://localhost:1234/v1"}

    deepseek = LLM.ByParameters(provider="deepseek", model="deepseek-model", apiKey="deep", silent=True)
    assert LLM.Models(deepseek, silent=True) == ["a", "b"]
    assert state["clients"][-1] == {"api_key": "deep", "base_url": "https://api.deepseek.com"}

    local = LLM.ByParameters(provider="openai-compatible", model="x", apiKey="local", baseURL="http://server/v1", silent=True)
    assert LLM.Models(local, silent=True) == ["a", "b"]
    assert state["clients"][-1] == {"api_key": "local", "base_url": "http://server/v1"}


def test_ollama_response_models_and_invalid_numeric_options(monkeypatch):
    state = _install_fake_requests(monkeypatch)
    llm = LLM.ByParameters(
        provider="ollama",
        model="llama3.1",
        baseURL="http://localhost:11434/",
        temperature="bad",
        maxOutputTokens=0,
        timeout="bad",
        silent=True,
    )

    assert LLM.Models(llm, silent=True) == ["llama3.1", "mistral"]
    response = LLM.Response(llm, "hello", silent=True)

    assert response["ok"] is True
    assert response["text"] == "ollama response"
    assert response["usage"] == {"input_tokens": 2, "output_tokens": 4, "total_tokens": 6}
    assert state["gets"][0] == {"url": "http://localhost:11434/api/tags", "timeout": 60}
    post = state["posts"][0]
    assert post["url"] == "http://localhost:11434/api/chat"
    assert post["timeout"] == 60
    assert post["json"]["options"] == {}


def test_anthropic_response_omits_none_temperature_and_collects_system_messages(monkeypatch):
    state = _install_fake_anthropic(monkeypatch)
    llm = LLM.ByParameters(provider="anthropic", model="claude-test", apiKey="ant", temperature=None, maxOutputTokens=None, timeout=8, silent=True)

    response = LLM.Response(
        llm,
        [{"role": "developer", "content": "dev"}, {"role": "user", "content": "question"}],
        systemPrompt="sys",
        silent=True,
    )

    assert response["ok"] is True
    assert response["text"] == "anthropic response"
    assert response["usage"] == {"input_tokens": 7, "output_tokens": 11, "total_tokens": None}
    assert state["clients"] == [{"api_key": "ant", "timeout": 8}]
    request = state["requests"][0]
    assert request["model"] == "claude-test"
    assert request["max_tokens"] == 512
    assert "temperature" not in request
    assert request["system"] == "sys\n\ndev"
    assert request["messages"] == [{"role": "user", "content": "question"}]


def test_google_new_sdk_and_old_sdk_fallback(monkeypatch):
    state = _install_fake_google(monkeypatch, fail_new_sdk=False)
    llm = LLM.ByParameters(provider="google", model="gemini-test", apiKey="g", temperature=None, maxOutputTokens=77, silent=True)

    response = LLM.Response(llm, "hello", silent=True)

    assert response["ok"] is True
    assert response["text"] == "google new response"
    assert state["new_clients"] == [{"api_key": "g"}]
    assert state["new_requests"][0]["config"] == {"max_output_tokens": 77}
    assert "USER:\nhello" in state["new_requests"][0]["contents"]

    state = _install_fake_google(monkeypatch, fail_new_sdk=True)
    response = LLM.Response(llm, "fallback", silent=True)
    assert response["ok"] is True
    assert response["text"] == "google old response"
    assert state["old_config"] == [{"api_key": "g"}]
    assert state["old_requests"][0]["generation_config"] == {"max_output_tokens": 77}




