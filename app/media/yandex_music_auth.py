"""One-time Yandex device sign-in. Never print or send access tokens to Telegram."""

import argparse
import json
import os
import time
from pathlib import Path


def main():
    from yandex_music import Client
    from yandex_music.utils.request import Request

    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()

    def on_code(code):
        print(
            json.dumps(
                {
                    "verification_url": code.verification_url,
                    "user_code": code.user_code,
                    "expires_in": code.expires_in,
                }
            ),
            flush=True,
        )

    result = Client(request=Request(timeout=25)).device_auth(
        on_code=on_code,
        timeout=600,
        device_name="OSVED music",
    )
    args.token_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.token_file.with_suffix(".tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(
            {
                "access_token": result.access_token,
                "refresh_token": result.refresh_token,
                "expires_at": time.time() + result.expires_in,
            },
            stream,
        )
    temporary.replace(args.token_file)
    args.token_file.chmod(0o600)
    print(json.dumps({"connected": True}), flush=True)


if __name__ == "__main__":
    main()
