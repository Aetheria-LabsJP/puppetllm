# puppetllm: Anthropic Messages API / Bedrock 互換 fake server (human-in-the-loop debug 用)。
# 詳細: README.md
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# proxy 本体。test も同梱して `--profile test` で実行可能。
COPY puppetllm/ /app/puppetllm/

EXPOSE 8765
CMD ["python3", "-m", "uvicorn", "puppetllm.fake_server:app", \
     "--host", "0.0.0.0", "--port", "8765"]
