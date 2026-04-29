# Copyright (C) 2026
# This file is part of TopologicPy.
#
# TopologicPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class _LLMConfig:
    """
    Lightweight configuration object used by the LLM static methods.
    """
    provider: str = "openai"
    model: Optional[str] = None
    apiKey: Optional[str] = None
    baseURL: Optional[str] = None
    temperature: float = 0.2
    maxOutputTokens: Optional[int] = None
    timeout: int = 60
    systemPrompt: Optional[str] = None
    silent: bool = False

    def __repr__(self):
        return (
            "_LLMConfig("
            f"provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"apiKey={'***' if self.apiKey else None}, "
            f"baseURL={self.baseURL!r}, "
            f"temperature={self.temperature!r}, "
            f"maxOutputTokens={self.maxOutputTokens!r}, "
            f"timeout={self.timeout!r}, "
            f"systemPrompt={self.systemPrompt!r}, "
            f"silent={self.silent!r})"
        )


class LLM:
    """
    A provider-neutral interface for calling large language models.

    The class follows the TopologicPy static-method style. Create an LLM
    configuration object with LLM.ByParameters(...), then pass that object to
    the other static methods:

        llm = LLM.ByParameters(provider="ollama", model="llama3.1")
        text = LLM.Prompt(llm, "Say hello.")
        data = LLM.JSON(llm, "Return {'ok': true} as JSON.")

    Supported providers in this minimal implementation:
    - "openai"
    - "anthropic"
    - "google"
    - "deepseek"
    - "huggingface"
    - "ollama"
    - "lmstudio"
    - "openai-compatible"
    """

    _PROVIDERS = {
        "openai": {
            "requires_api_key": True,
            "default_base_url": None,
            "default_model": "gpt-5.4-mini",
            "mode": "openai",
        },
        "anthropic": {
            "requires_api_key": True,
            "default_base_url": None,
            "default_model": "claude-haiku-4-5",
            "mode": "anthropic",
        },
        "google": {
            "requires_api_key": True,
            "default_base_url": None,
            "default_model": "gemini-2.5-flash",
            "mode": "google",
        },
        "deepseek": {
            "requires_api_key": True,
            "default_base_url": "https://api.deepseek.com",
            "default_model": "deepseek-v4-flash",
            "mode": "openai",
        },
        "huggingface": {
            "requires_api_key": True,
            "default_base_url": "https://router.huggingface.co/v1",
            "default_model": None,
            "mode": "openai",
        },
        "ollama": {
            "requires_api_key": False,
            "default_base_url": "http://localhost:11434",
            "default_model": "llama3.1",
            "mode": "ollama",
        },
        "lmstudio": {
            "requires_api_key": False,
            "default_base_url": "http://localhost:1234/v1",
            "default_model": "local-model",
            "mode": "openai",
        },
        "openai-compatible": {
            "requires_api_key": False,
            "default_base_url": None,
            "default_model": None,
            "mode": "openai",
        },
    }

    # -------------------------------------------------------------------------
    # Constructors and metadata
    # -------------------------------------------------------------------------
    @staticmethod
    def ByParameters(
        provider: str = "openai",
        model: str = None,
        apiKey: str = None,
        baseURL: str = None,
        temperature: float = 0.2,
        maxOutputTokens: int = None,
        timeout: int = 60,
        systemPrompt: str = None,
        silent: bool = False
    ):
        """
        Creates and returns a lightweight LLM configuration object.

        Parameters
        ----------
        provider : str , optional
            The LLM provider. Supported values are "openai", "anthropic", "google",
            "deepseek", "huggingface", "ollama", "lmstudio", and
            "openai-compatible". Default is "openai".
        model : str , optional
            The model name. If None, a provider-specific default is used when
            available.
        apiKey : str , optional
            The provider API key. If None, provider-specific environment
            variables are checked where applicable.
        baseURL : str , optional
            The provider base URL. Required for some OpenAI-compatible local or
            remote providers. Defaults are used for Ollama, LM Studio,
            HuggingFace, and DeepSeek.
        temperature : float , optional
            The sampling temperature. Default is 0.2.
        maxOutputTokens : int , optional
            The maximum number of output tokens. Default is None.
        timeout : int , optional
            The timeout in seconds. Default is 60.
        systemPrompt : str , optional
            Optional default system prompt.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        _LLMConfig
            The LLM configuration object.
        """
        try:
            provider = LLM._normalize_provider(provider)
            info = LLM._PROVIDERS.get(provider, LLM._PROVIDERS["openai-compatible"])
            return _LLMConfig(
                provider=provider,
                model=model or info.get("default_model"),
                apiKey=apiKey,
                baseURL=baseURL or info.get("default_base_url"),
                temperature=temperature,
                maxOutputTokens=maxOutputTokens,
                timeout=timeout,
                systemPrompt=systemPrompt,
                silent=silent,
            )
        except Exception as e:
            if not silent:
                print(f"LLM.ByParameters - Error: {e}. Returning None.")
            return None

    @staticmethod
    def ProviderInfo(provider: str = None, silent: bool = False):
        """
        Returns metadata about supported providers.

        Parameters
        ----------
        provider : str , optional
            If specified, only metadata for the requested provider is returned.
            If None, metadata for all providers is returned.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            Provider metadata.
        """
        try:
            if provider is None:
                return json.loads(json.dumps(LLM._PROVIDERS))
            provider = LLM._normalize_provider(provider)
            return dict(LLM._PROVIDERS.get(provider, {}))
        except Exception as e:
            if not silent:
                print(f"LLM.ProviderInfo - Error: {e}. Returning None.")
            return None

    # -------------------------------------------------------------------------
    # Public inference methods
    # -------------------------------------------------------------------------
    @staticmethod
    def Prompt(
        llm,
        prompt: Union[str, List[Dict[str, Any]]],
        systemPrompt: str = None,
        temperature: float = None,
        maxOutputTokens: int = None,
        timeout: int = None,
        silent: bool = False,
    ) -> Optional[str]:
        """
        Sends a text prompt to the configured LLM and returns text.

        Parameters
        ----------
        llm : _LLMConfig
            The LLM object returned by LLM.ByParameters.
        prompt : str or list
            The input prompt. If a list is supplied, it is treated as a chat
            message list with dictionaries containing "role" and "content".
        systemPrompt : str , optional
            The system prompt. Overrides llm.systemPrompt if supplied.
        temperature : float , optional
            Overrides llm.temperature if supplied.
        maxOutputTokens : int , optional
            Overrides llm.maxOutputTokens if supplied.
        timeout : int , optional
            Overrides llm.timeout if supplied.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The generated text, or None on failure.
        """
        response = LLM.Response(
            llm=llm,
            prompt=prompt,
            systemPrompt=systemPrompt,
            temperature=temperature,
            maxOutputTokens=maxOutputTokens,
            timeout=timeout,
            silent=silent,
        )
        if response and response.get("ok"):
            return response.get("text")
        return None

    @staticmethod
    def JSON(
        llm,
        prompt: Union[str, List[Dict[str, Any]]],
        schema: dict = None,
        systemPrompt: str = None,
        temperature: float = None,
        maxOutputTokens: int = None,
        timeout: int = None,
        repair: bool = True,
        silent: bool = False,
    ):
        """
        Sends a prompt to the configured LLM and returns parsed JSON.

        Parameters
        ----------
        llm : _LLMConfig
            The LLM object returned by LLM.ByParameters.
        prompt : str or list
            The input prompt. If a list is supplied, it is treated as a chat
            message list with dictionaries containing "role" and "content".
        schema : dict , optional
            Optional JSON schema. In this minimal implementation, the schema is
            included in the prompt for all providers, and provider-native schema
            enforcement may be added later.
        systemPrompt : str , optional
            The system prompt. Overrides llm.systemPrompt if supplied.
        temperature : float , optional
            Overrides llm.temperature if supplied.
        maxOutputTokens : int , optional
            Overrides llm.maxOutputTokens if supplied.
        timeout : int , optional
            Overrides llm.timeout if supplied.
        repair : bool , optional
            If True, attempts to parse JSON from fenced or surrounding text. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict or list
            Parsed JSON, or None on failure.
        """
        json_system = systemPrompt or getattr(llm, "systemPrompt", None) or ""
        json_instruction = (
            "Return valid JSON only. Do not include markdown, comments, explanations, "
            "or text outside the JSON value."
        )
        if json_system:
            json_system = json_system.strip() + "\n\n" + json_instruction
        else:
            json_system = json_instruction

        if schema is not None:
            schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
            if isinstance(prompt, str):
                prompt = prompt + "\n\nRequired JSON schema:\n" + schema_text
            else:
                prompt = list(prompt) + [{"role": "user", "content": "Required JSON schema:\n" + schema_text}]

        response = LLM.Response(
            llm=llm,
            prompt=prompt,
            systemPrompt=json_system,
            temperature=temperature if temperature is not None else 0,
            maxOutputTokens=maxOutputTokens,
            timeout=timeout,
            silent=silent,
        )
        if not response or not response.get("ok"):
            return None

        text = response.get("text") or ""
        parsed = LLM._coerce_json(text, repair=repair)
        if parsed is None and not (silent or getattr(llm, "silent", False)):
            print("LLM.JSON - Error: Could not parse JSON from response.")
            print(text)
        return parsed

    @staticmethod
    def Response(
        llm,
        prompt: Union[str, List[Dict[str, Any]]],
        systemPrompt: str = None,
        temperature: float = None,
        maxOutputTokens: int = None,
        timeout: int = None,
        silent: bool = False,
    ) -> Dict[str, Any]:
        """
        Sends a prompt to the configured LLM and returns a normalized response dictionary.

        Parameters
        ----------
        llm : _LLMConfig
            The LLM object returned by LLM.ByParameters.
        prompt : str or list
            The input prompt. If a list is supplied, it is treated as a chat
            message list with dictionaries containing "role" and "content".
        systemPrompt : str , optional
            The system prompt. Overrides llm.systemPrompt if supplied.
        temperature : float , optional
            Overrides llm.temperature if supplied.
        maxOutputTokens : int , optional
            Overrides llm.maxOutputTokens if supplied.
        timeout : int , optional
            Overrides llm.timeout if supplied.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A normalized response dictionary with keys:
            ok, provider, model, text, json, raw, usage, error_type, message.
        """
        if llm is None:
            return LLM._error("unknown_provider", "The input llm object is None.", None, None, silent=silent)

        provider = LLM._normalize_provider(getattr(llm, "provider", "openai"))
        mode = LLM._PROVIDERS.get(provider, {}).get("mode", "openai")
        model = getattr(llm, "model", None)
        temp = getattr(llm, "temperature", 0.2) if temperature is None else temperature
        mot = getattr(llm, "maxOutputTokens", None) if maxOutputTokens is None else maxOutputTokens
        tout = getattr(llm, "timeout", 60) if timeout is None else timeout
        sp = systemPrompt if systemPrompt is not None else getattr(llm, "systemPrompt", None)
        effective_silent = bool(silent or getattr(llm, "silent", False))

        try:
            messages = LLM._messages(prompt, sp)

            if mode == "openai":
                return LLM._response_openai_compatible(llm, provider, model, messages, temp, mot, tout, effective_silent)

            if mode == "anthropic":
                return LLM._response_anthropic(llm, provider, model, messages, temp, mot, tout, effective_silent)

            if mode == "google":
                return LLM._response_google(llm, provider, model, messages, temp, mot, tout, effective_silent)

            if mode == "ollama":
                return LLM._response_ollama(llm, provider, model, messages, temp, mot, tout, effective_silent)

            return LLM._error("unknown_provider", f"Unknown provider: {provider}", provider, model, silent=effective_silent)

        except Exception as e:
            return LLM._exception_response(e, provider, model, effective_silent, "LLM.Response")

    @staticmethod
    def Test(llm, prompt: str = "Reply with OK only.", silent: bool = False) -> Dict[str, Any]:
        """
        Tests whether the configured LLM is reachable and can generate a response.

        Parameters
        ----------
        llm : _LLMConfig
            The LLM object returned by LLM.ByParameters.
        prompt : str , optional
            The test prompt. Default is "Reply with OK only."
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A normalized test dictionary.
        """
        if llm is None:
            return LLM._error("unknown_provider", "The input llm object is None.", None, None, silent=silent)

        provider = LLM._normalize_provider(getattr(llm, "provider", "openai"))
        model = getattr(llm, "model", None)
        effective_silent = bool(silent or getattr(llm, "silent", False))

        if provider == "ollama":
            base_url = (getattr(llm, "baseURL", None) or LLM._PROVIDERS["ollama"]["default_base_url"]).rstrip("/")
            try:
                import requests
                r = requests.get(base_url + "/api/tags", timeout=getattr(llm, "timeout", 60))
                r.raise_for_status()
            except Exception as e:
                return LLM._exception_response(e, provider, model, effective_silent, "LLM.Test")

        response = LLM.Response(llm, prompt, silent=effective_silent)
        if response.get("ok"):
            return {
                "ok": True,
                "provider": provider,
                "model": model,
                "message": response.get("text"),
                "details": response,
            }
        return {
            "ok": False,
            "provider": provider,
            "model": model,
            "message": response.get("message"),
            "details": response,
        }

    @staticmethod
    def Models(llm, silent: bool = False) -> Optional[List[str]]:
        """
        Lists available models where the configured provider supports model listing.

        Parameters
        ----------
        llm : _LLMConfig
            The LLM object returned by LLM.ByParameters.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The model names, or None on failure/unsupported provider.
        """
        if llm is None:
            if not silent:
                print("LLM.Models - Error: The input llm object is None. Returning None.")
            return None

        provider = LLM._normalize_provider(getattr(llm, "provider", "openai"))
        effective_silent = bool(silent or getattr(llm, "silent", False))

        try:
            if provider == "ollama":
                import requests
                base_url = (getattr(llm, "baseURL", None) or LLM._PROVIDERS["ollama"]["default_base_url"]).rstrip("/")
                r = requests.get(base_url + "/api/tags", timeout=getattr(llm, "timeout", 60))
                r.raise_for_status()
                data = r.json()
                return [m.get("name") for m in data.get("models", []) if m.get("name")]

            if provider in ("openai", "deepseek", "huggingface", "lmstudio", "openai-compatible"):
                from openai import OpenAI
                client = LLM._openai_client(llm, provider)
                models = client.models.list()
                data = getattr(models, "data", []) or []
                return [getattr(m, "id", None) for m in data if getattr(m, "id", None)]

            if not effective_silent:
                print(f"LLM.Models - Warning: Model listing is not implemented for provider '{provider}'. Returning None.")
            return None

        except Exception as e:
            if not effective_silent:
                print(f"LLM.Models - Error: {e}. Returning None.")
            return None

    # -------------------------------------------------------------------------
    # Provider-specific calls
    # -------------------------------------------------------------------------
    @staticmethod
    def _response_openai_compatible(llm, provider, model, messages, temperature, max_tokens, timeout, silent):
        try:
            from openai import OpenAI
            client = LLM._openai_client(llm, provider)
            request = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "timeout": timeout,
            }
            if max_tokens is not None:
                request["max_tokens"] = max_tokens

            resp = client.chat.completions.create(**request)
            text = ""
            try:
                text = resp.choices[0].message.content or ""
            except Exception:
                text = str(resp)

            usage = {}
            try:
                usage_obj = getattr(resp, "usage", None)
                if usage_obj is not None:
                    usage = {
                        "input_tokens": getattr(usage_obj, "prompt_tokens", None),
                        "output_tokens": getattr(usage_obj, "completion_tokens", None),
                        "total_tokens": getattr(usage_obj, "total_tokens", None),
                    }
            except Exception:
                usage = {}

            return LLM._ok(provider, model, text, resp, usage)

        except Exception as e:
            return LLM._exception_response(e, provider, model, silent, "LLM._response_openai_compatible")

    @staticmethod
    def _response_anthropic(llm, provider, model, messages, temperature, max_tokens, timeout, silent):
        try:
            from anthropic import Anthropic

            api_key = getattr(llm, "apiKey", None) or os.getenv("ANTHROPIC_API_KEY")
            client = Anthropic(api_key=api_key, timeout=timeout)

            system = ""
            anthropic_messages = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    system += ("\n\n" if system else "") + str(content)
                elif role in ("user", "assistant"):
                    anthropic_messages.append({"role": role, "content": str(content)})
                elif role == "developer":
                    system += ("\n\n" if system else "") + str(content)

            request = {
                "model": model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 512,
            }
            if system:
                request["system"] = system

            resp = client.messages.create(**request)

            parts = []
            for p in getattr(resp, "content", []) or []:
                if hasattr(p, "text"):
                    parts.append(p.text)
                elif isinstance(p, dict) and "text" in p:
                    parts.append(p["text"])
            text = "".join(parts)

            usage = {}
            try:
                usage_obj = getattr(resp, "usage", None)
                usage = {
                    "input_tokens": getattr(usage_obj, "input_tokens", None),
                    "output_tokens": getattr(usage_obj, "output_tokens", None),
                    "total_tokens": None,
                }
            except Exception:
                usage = {}

            return LLM._ok(provider, model, text, resp, usage)

        except Exception as e:
            return LLM._exception_response(e, provider, model, silent, "LLM._response_anthropic")

    @staticmethod
    def _response_google(llm, provider, model, messages, temperature, max_tokens, timeout, silent):
        # Prefer the newer google-genai SDK. Fall back to google-generativeai.
        api_key = getattr(llm, "apiKey", None) or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        prompt_text = LLM._messages_to_text(messages)

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            config_args = {"temperature": temperature}
            if max_tokens is not None:
                config_args["max_output_tokens"] = max_tokens

            resp = client.models.generate_content(
                model=model,
                contents=prompt_text,
                config=types.GenerateContentConfig(**config_args),
            )
            text = getattr(resp, "text", "") or ""
            return LLM._ok(provider, model, text, resp, {})

        except Exception as e_new:
            try:
                import google.generativeai as genai_old

                genai_old.configure(api_key=api_key)
                generation_config = {
                    "temperature": temperature,
                }
                if max_tokens is not None:
                    generation_config["max_output_tokens"] = max_tokens

                gmodel = genai_old.GenerativeModel(model)
                resp = gmodel.generate_content(prompt_text, generation_config=generation_config)
                text = getattr(resp, "text", "") or ""
                return LLM._ok(provider, model, text, resp, {})

            except Exception as e_old:
                if not silent:
                    print(f"LLM._response_google - Error using google-genai: {e_new}")
                return LLM._exception_response(e_old, provider, model, silent, "LLM._response_google")

    @staticmethod
    def _response_ollama(llm, provider, model, messages, temperature, max_tokens, timeout, silent):
        try:
            import requests
            base_url = (getattr(llm, "baseURL", None) or LLM._PROVIDERS["ollama"]["default_base_url"]).rstrip("/")
            url = base_url + "/api/chat"
            options = {"temperature": temperature}
            if max_tokens is not None:
                options["num_predict"] = max_tokens

            r = requests.post(
                url,
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": options,
                },
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            text = data.get("message", {}).get("content", "") or ""

            usage = {
                "input_tokens": data.get("prompt_eval_count"),
                "output_tokens": data.get("eval_count"),
                "total_tokens": (
                    (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
                    if data.get("prompt_eval_count") is not None or data.get("eval_count") is not None
                    else None
                ),
            }
            return LLM._ok(provider, model, text, data, usage)

        except Exception as e:
            return LLM._exception_response(e, provider, model, silent, "LLM._response_ollama")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _openai_client(llm, provider):
        from openai import OpenAI

        api_key = getattr(llm, "apiKey", None)
        base_url = getattr(llm, "baseURL", None)

        if provider == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
        elif provider == "deepseek":
            api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            base_url = base_url or LLM._PROVIDERS["deepseek"]["default_base_url"]
        elif provider == "huggingface":
            api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
            base_url = base_url or LLM._PROVIDERS["huggingface"]["default_base_url"]
        elif provider == "lmstudio":
            # LM Studio often does not require a real key, but the OpenAI SDK expects one.
            api_key = api_key or "lm-studio"
            base_url = base_url or LLM._PROVIDERS["lmstudio"]["default_base_url"]
        elif provider == "openai-compatible":
            api_key = api_key or os.getenv("OPENAI_API_KEY") or "local"
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    @staticmethod
    def _normalize_provider(provider):
        if provider is None:
            return "openai"
        p = str(provider).strip().lower().replace("_", "-")
        aliases = {
            "open ai": "openai",
            "open-ai": "openai",
            "chatgpt": "openai",
            "claude": "anthropic",
            "anthropic-claude": "anthropic",
            "gemini": "google",
            "google-gemini": "google",
            "hf": "huggingface",
            "hugging-face": "huggingface",
            "hugging face": "huggingface",
            "lm-studio": "lmstudio",
            "local": "openai-compatible",
            "openai-compatible-api": "openai-compatible",
            "openai-compatible-api-server": "openai-compatible",
        }
        return aliases.get(p, p)

    @staticmethod
    def _messages(prompt, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        if isinstance(prompt, list):
            for m in prompt:
                if isinstance(m, dict):
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    messages.append({"role": role, "content": str(content)})
                else:
                    messages.append({"role": "user", "content": str(m)})
        else:
            messages.append({"role": "user", "content": str(prompt)})
        return messages

    @staticmethod
    def _messages_to_text(messages):
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _coerce_json(text, repair=True):
        if text is None:
            return None
        if not isinstance(text, str):
            return text if isinstance(text, (dict, list)) else None

        s = text.strip()
        if not s:
            return None

        # Remove common markdown fences.
        fence = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", s, flags=re.DOTALL)
        if fence:
            s = fence.group(1).strip()

        try:
            return json.loads(s)
        except Exception:
            pass

        if not repair:
            return None

        # Try to extract the first object or array.
        candidates = []
        obj_start = s.find("{")
        obj_end = s.rfind("}")
        if obj_start >= 0 and obj_end > obj_start:
            candidates.append(s[obj_start:obj_end + 1])

        arr_start = s.find("[")
        arr_end = s.rfind("]")
        if arr_start >= 0 and arr_end > arr_start:
            candidates.append(s[arr_start:arr_end + 1])

        for c in candidates:
            try:
                return json.loads(c)
            except Exception:
                continue

        return None

    @staticmethod
    def _ok(provider, model, text, raw, usage=None):
        return {
            "ok": True,
            "provider": provider,
            "model": model,
            "text": text,
            "json": LLM._coerce_json(text, repair=True),
            "raw": raw,
            "usage": usage or {},
            "error_type": None,
            "message": None,
        }

    @staticmethod
    def _error(error_type, message, provider, model, raw=None, silent=False):
        if not silent:
            print(f"LLM - Error: {message}")
        return {
            "ok": False,
            "provider": provider,
            "model": model,
            "text": None,
            "json": None,
            "raw": raw,
            "usage": {},
            "error_type": error_type,
            "message": message,
        }

    @staticmethod
    def _exception_response(exception, provider, model, silent, context):
        error_type = LLM._classify_exception(exception)
        message = str(exception)
        if not silent:
            print(f"{context} - {error_type}: {repr(exception)}")
        return {
            "ok": False,
            "provider": provider,
            "model": model,
            "text": None,
            "json": None,
            "raw": exception,
            "usage": {},
            "error_type": error_type,
            "message": message,
        }

    @staticmethod
    def _classify_exception(exception):
        name = exception.__class__.__name__.lower()
        msg = str(exception).lower()

        if "insufficient_quota" in msg or "quota" in msg:
            return "quota_error"
        if "rate" in name or "rate limit" in msg or "429" in msg:
            return "rate_limit_error"
        if "authentication" in name or "unauthorized" in msg or "api key" in msg or "401" in msg:
            return "authentication_error"
        if "not found" in msg or "model" in msg and "not" in msg and "found" in msg or "404" in msg:
            return "model_not_found"
        if "timeout" in name or "timed out" in msg or "timeout" in msg:
            return "timeout"
        if (
            "connection" in name
            or "connection" in msg
            or "connection refused" in msg
            or "max retries exceeded" in msg
            or "winerror 10061" in msg
        ):
            return "connection_error"
        return "unknown_error"
