import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .providers import (
    BaseLLMProvider,
    GenerationConfig,
    OpenAIProvider,
    OllamaProvider,
)


class LLMGenerator:

    def __init__(self, config_path: str | None = None):
        self.providers: dict[str, BaseLLMProvider] = {}
        self.config: dict[str, Any] = {}
        self.gen_config = GenerationConfig()
        self.models: list[str] = []
        self._model_providers: dict[str, str] = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        gen = self.config.get("generation", {})
        self.gen_config = GenerationConfig(
            temperature=gen.get("temperature"),
            max_tokens=gen.get("max_tokens"),
            top_p=gen.get("top_p"),
        )
        models_cfg = self.config.get("models", [])
        self.models = [m["name"] for m in models_cfg]
        self._model_providers = {m["name"]: m["provider"] for m in models_cfg}

    def setup_providers(self) -> None:
        """Register providers needed by the models in config."""
        cls_map = {
            "openai": OpenAIProvider,
            "ollama": OllamaProvider,
        }

        api_cfg = self.config.get("api", {})
        needed_models = {m["provider"] for m in self.config.get("models", [])}

        for name in needed_models:
            if name in self.providers:
                continue
            cls = cls_map.get(name)
            if cls is None:
                raise ValueError(f"Unknown provider: {name}")
            if name == "ollama":
                url = api_cfg.get("ollama", {}).get(
                    "base_url", "http://localhost:11434"
                )
                self.providers[name] = cls(base_url=url)
            else:
                env_key = api_cfg.get(name, {}).get("env_key")
                api_key = os.getenv(env_key) if env_key else None
                self.providers[name] = cls(api_key=api_key)

    def get_model_provider(self, model_name: str) -> str:
        try:
            return self._model_providers[model_name]
        except KeyError:
            raise ValueError(f"model '{model_name}' not in config")

    def _generate_with_retry(
        self, prompt: str, provider: str, model: str,
        retries: int = 3, delay: float = 2.0,
    ) -> str:
        if provider not in self.providers:
            raise ValueError(f"provider '{provider}' not registered")

        last_err = None
        for attempt in range(1, retries + 1):
            try:
                return self.providers[provider].generate(
                    prompt, model, self.gen_config,
                )
            except Exception as e:
                last_err = e
                if attempt < retries:
                    wait = delay * (2 ** (attempt - 1))
                    print(f"  Retry {attempt}/{retries} after {wait:.0f}s: {e}")
                    time.sleep(wait)

        raise RuntimeError(f"gave up after {retries} attempts: {last_err}")

    def generate(self, prompt: str, provider_name: str, model: str) -> str:
        return self._generate_with_retry(prompt, provider_name, model)

    @staticmethod
    def _save_json(path: Path, data) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def generate_corpus(
        self,
        prompts: list[dict[str, Any]],
        models: list[str] | None = None,
        output_dir: str = "data/raw",
        num_runs: int = 3,
        api_delay: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Run generation across all prompt x model combos, saving incrementally.

        Supports resume — existing entries in corpus.json are skipped.
        """
        models = models or self.models
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        corpus_file = out / "corpus.json"

        if corpus_file.exists():
            with open(corpus_file, "r") as f:
                results = json.load(f)
            done = {
                (r["prompt_id"], r["model"], r["run"])
                for r in results if "error" not in r
            }
            print(f"Resuming: {len(done)} existing entries loaded.")
        else:
            results, done = [], set()

        total = len(prompts) * len(models) * num_runs
        completed, new_count = 0, 0
        cfg = self.gen_config

        for p in prompts:
            for model in models:
                prov = self.get_model_provider(model)

                for run in range(1, num_runs + 1):
                    completed += 1
                    key = (p["id"], model, run)

                    if key in done:
                        print(f"[{completed}/{total}] SKIP {p['id']} | {model} | run {run}")
                        continue

                    print(f"[{completed}/{total}] {p['id']} | {model} | run {run}/{num_runs}")

                    try:
                        text = self._generate_with_retry(p["text"], prov, model)

                        if not text or not text.strip():
                            print("  WARNING: empty response, skipping")
                            results.append({
                                "prompt_id": p["id"], "model": model,
                                "run": run, "error": "empty response",
                                "timestamp": datetime.now().isoformat(),
                            })
                            continue

                        results.append({
                            "prompt_id": p["id"],
                            "prompt_text": p["text"],
                            "genre": p.get("genre", "unknown"),
                            "model": model, "provider": prov, "run": run,
                            "generated_text": text,
                            "timestamp": datetime.now().isoformat(),
                            "config": {
                                "temperature": cfg.temperature,
                                "max_tokens": cfg.max_tokens,
                                "top_p": cfg.top_p,
                            },
                        })
                        done.add(key)
                        new_count += 1

                        self._save_json(corpus_file, results)
                        if prov != "ollama":  # rate-limit for hosted APIs only
                            time.sleep(api_delay)

                    except Exception as e:
                        print(f"  ERROR: {e}")
                        results.append({
                            "prompt_id": p["id"], "model": model,
                            "run": run, "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        })

        self._save_json(corpus_file, results)
        ok = sum(1 for r in results if "error" not in r)
        print(f"\nDone. {ok} successful ({new_count} new), {len(done) - new_count} resumed.")
        return results

    def generate_tone_corpus(
        self,
        prompts: list[dict[str, Any]],
        tone_modifiers: dict[str, str],
        models: list[str] | None = None,
        output_path: str = "data/raw/corpus_robustness.json",
        num_runs: int = 2,
        api_delay: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Same as generate_corpus but prepends tone instructions to each prompt."""
        models = models or self.models
        corpus_file = Path(output_path)
        corpus_file.parent.mkdir(parents=True, exist_ok=True)

        if corpus_file.exists():
            with open(corpus_file, "r") as f:
                results = json.load(f)
            done = {
                (r["prompt_id"], r["tone"], r["model"], r["run"])
                for r in results if "error" not in r
            }
            print(f"Resuming: {len(done)} existing entries loaded.")
        else:
            results, done = [], set()

        total = len(prompts) * len(tone_modifiers) * len(models) * num_runs
        completed, new_count = 0, 0
        cfg = self.gen_config

        for p in prompts:
            for tone, prefix in tone_modifiers.items():
                full_prompt = prefix.strip() + "\n\n" + p["text"]

                for model in models:
                    prov = self.get_model_provider(model)

                    for run in range(1, num_runs + 1):
                        completed += 1
                        key = (p["id"], tone, model, run)

                        if key in done:
                            print(f"[{completed}/{total}] SKIP {p['id']} | {tone} | {model} | run {run}")
                            continue

                        print(f"[{completed}/{total}] {p['id']} | {tone} | {model} | run {run}/{num_runs}")

                        try:
                            text = self._generate_with_retry(full_prompt, prov, model)

                            if not text or not text.strip():
                                print("  WARNING: empty response, skipping")
                                results.append({
                                    "prompt_id": p["id"], "tone": tone,
                                    "model": model, "run": run,
                                    "error": "empty response",
                                    "timestamp": datetime.now().isoformat(),
                                })
                                continue

                            results.append({
                                "prompt_id": p["id"],
                                "prompt_text": p["text"],
                                "full_prompt": full_prompt,
                                "tone": tone,
                                "genre": p.get("genre", "unknown"),
                                "model": model, "provider": prov, "run": run,
                                "generated_text": text,
                                "timestamp": datetime.now().isoformat(),
                                "config": {
                                    "temperature": cfg.temperature,
                                    "max_tokens": cfg.max_tokens,
                                    "top_p": cfg.top_p,
                                },
                            })
                            done.add(key)
                            new_count += 1

                            self._save_json(corpus_file, results)
                            if prov != "ollama":  # rate-limit for hosted APIs only
                                time.sleep(api_delay)

                        except Exception as e:
                            print(f"  ERROR: {e}")
                            results.append({
                                "prompt_id": p["id"], "tone": tone,
                                "model": model, "run": run,
                                "error": str(e),
                                "timestamp": datetime.now().isoformat(),
                            })

        self._save_json(corpus_file, results)
        ok = sum(1 for r in results if "error" not in r)
        print(f"\nDone. {ok} successful ({new_count} new), {len(done) - new_count} resumed.")
        return results

    def load_prompts(self, prompt_file: str) -> list[dict[str, Any]]:
        with open(prompt_file, "r") as f:
            data = yaml.safe_load(f)

        if isinstance(data, dict):
            constraints = data.get("output_constraints", "")
            prompts = data.get("prompts", data)
        else:
            constraints = ""
            prompts = data

        if constraints:
            for p in prompts:
                p["text"] = p["text"].strip() + " " + constraints.strip()

        return prompts
