# puppetllm: fake server compatible with the Anthropic Messages API / Bedrock / OpenAI (for human-in-the-loop debugging).
# Details: README.md
FROM python:3.12-slim AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The proxy itself. Tests are bundled too, so `--profile test` can run them.
COPY puppetllm/ /app/puppetllm/

EXPOSE 8765
CMD ["python3", "-m", "uvicorn", "puppetllm.fake_server:app", \
     "--host", "0.0.0.0", "--port", "8765"]

# Optional Agent SDK relay. Keep its bundled CLI out of the default server image.
FROM base AS agent-sdk
COPY pyproject.toml README.md LICENSE /app/
# The extra pins the SDK: the transport patches depend on its internals.
RUN pip install --no-cache-dir '.[agent-sdk]' \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin puppetllm \
    && mkdir -p /var/lib/puppetllm/sdk \
    && chown -R puppetllm:puppetllm /var/lib/puppetllm
ENV PUPPETLLM_SDK_STATE_DIR=/var/lib/puppetllm/sdk
USER puppetllm
CMD ["python3", "-m", "puppetllm", "relay", "--kind", "agent-sdk"]

# Preserve `docker build .` / the default compose service as the server-only image.
FROM base AS server
