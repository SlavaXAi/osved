"""Read-only connectivity/account probe. Never prints API keys or token-bearing URLs."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

from app.config import Settings


def probe(settings: Settings) -> dict:
    result = {}
    if settings.openrouter_api_key:
        headers = {"Authorization": "Bearer " + settings.openrouter_api_key.get_secret_value()}
        for name, path in (("openrouter_key", "key"), ("openrouter_credits", "credits")):
            try:
                with urlopen(
                    Request("https://openrouter.ai/api/v1/" + path, headers=headers), timeout=25
                ) as response:
                    data = json.load(response).get("data", {})
                result[name] = {
                    k: data.get(k)
                    for k in (
                        "limit",
                        "limit_remaining",
                        "usage",
                        "usage_daily",
                        "usage_monthly",
                        "total_credits",
                        "total_usage",
                    )
                    if k in data
                }
            except Exception as exc:
                result[name] = {
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "code", None),
                }
        try:
            with urlopen("https://openrouter.ai/api/v1/models", timeout=25) as response:
                models = json.load(response)["data"]
            by_id = {x["id"]: x for x in models}
            desired = [
                settings.openrouter_search_model,
                settings.openrouter_card_model,
                settings.openrouter_model,
            ]
            result["configured_models"] = {
                key: {"available": key in by_id, "pricing": by_id.get(key, {}).get("pricing")}
                for key in desired
            }
            result["search_alternatives"] = [
                x["id"]
                for x in models
                if x["id"].startswith(
                    ("openai/gpt-5", "perplexity/sonar", "google/gemini-2.5-flash")
                )
            ][:18]
        except Exception as exc:
            result["models_error"] = type(exc).__name__
    if settings.telegram_bot_token:
        for method in ("getMe", "getWebhookInfo"):
            try:
                with urlopen(
                    "https://api.telegram.org/bot"
                    + settings.telegram_bot_token.get_secret_value()
                    + "/"
                    + method,
                    timeout=25,
                ) as response:
                    data = json.load(response)
                result[method] = {
                    "ok": data.get("ok"),
                    "username": data.get("result", {}).get("username"),
                    "webhook_configured": bool(data.get("result", {}).get("url")),
                }
            except Exception as exc:
                result[method] = {
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "code", None),
                }
    return result


if __name__ == "__main__":
    print(json.dumps(probe(Settings()), ensure_ascii=False, indent=2))
