# puppetllm: fake server compatible with the Anthropic Messages API / Bedrock / OpenAI (for human-in-the-loop debugging).
# Details: README.md
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The proxy itself. Tests are bundled too, so `--profile test` can run them.
COPY puppetllm/ /app/puppetllm/

EXPOSE 8765
CMD ["python3", "-m", "uvicorn", "puppetllm.fake_server:app", \
     "--host", "0.0.0.0", "--port", "8765"]
