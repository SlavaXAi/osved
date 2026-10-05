ARG MONID_RUNTIME_IMAGE=node:22-bookworm-slim
FROM ${MONID_RUNTIME_IMAGE} AS monid-runtime

USER root
RUN node /usr/local/lib/node_modules/npm/bin/npm-cli.js install -g @monid-ai/cli@0.1.7

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Embedded Cyrillic fonts for the on-demand, plain-text video PDFs.
RUN apt-get update && apt-get install -y --no-install-recommends fonts-liberation2 \
    && rm -rf /var/lib/apt/lists/*

# The research adapters intentionally use the Monid CLI as a single provider
# gateway. It requires Node >=20, so copy the audited runtime from a build stage
# instead of accepting Debian's older nodejs package.
COPY --from=monid-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=monid-runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/@monid-ai/cli/dist/index.js /usr/local/bin/monid \
    && monid --version

COPY pyproject.toml uv.lock README.md ./
COPY app ./app
RUN pip install uv==0.9.11 \
    && uv sync --frozen --no-dev --no-editable

COPY config ./config
COPY docker-entrypoint.sh /usr/local/bin/newsfeed-entrypoint

RUN useradd --create-home --uid 10001 newsfeed \
    && mkdir -p /app/data /app/exports \
    && chown -R newsfeed:newsfeed /app \
    && chmod 0755 /usr/local/bin/newsfeed-entrypoint

USER newsfeed

ENTRYPOINT ["newsfeed-entrypoint"]
CMD ["osved", "serve"]
