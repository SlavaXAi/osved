#!/bin/sh
set -eu

# The CLI stores credentials in the runtime user's home. The secret stays in
# the container environment and is never copied into an image layer.
if [ -n "${MONID_API_KEY:-}" ]; then
    monid keys add --key "$MONID_API_KEY" --label runtime >/dev/null 2>&1 \
        || monid keys activate --label runtime >/dev/null 2>&1
fi

exec "$@"
