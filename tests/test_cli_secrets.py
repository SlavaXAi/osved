import stat
from pathlib import Path

from app.cli import update_env_value, write_private_env


def test_private_env_is_mode_0600_and_not_echoed_elsewhere(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    values = {
        "TELEGRAM_BOT_TOKEN": "test-bot-token",
        "TELEGRAM_API_ID": "123",
        "TELEGRAM_API_HASH": "test-api-hash",
        "ALLOWED_USER_ID": "",
        "OPENROUTER_API_KEY": "",
        "OPENROUTER_MODEL": "",
        "MONID_API_KEY": "",
    }

    write_private_env(env_path, values)

    assert "test-bot-token" in env_path.read_text(encoding="utf-8")
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


def test_update_owner_preserves_other_secrets(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=private\nALLOWED_USER_ID=\n",
        encoding="utf-8",
    )

    update_env_value(env_path, "ALLOWED_USER_ID", "12345")

    content = env_path.read_text(encoding="utf-8")
    assert "TELEGRAM_BOT_TOKEN=private" in content
    assert "ALLOWED_USER_ID=12345" in content
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
