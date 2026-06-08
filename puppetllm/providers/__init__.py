"""puppetllm のプロバイダ・アダプタ群。

canonical core (fake_server.py) はプロバイダ非依存の正規化 snapshot + /_control/* を持ち、
各プロバイダはここで「ワイヤ形式の入出力」だけを担当する:

- anthropic: `/v1/messages` (SSE / 単一 JSON)  — 現状は fake_server.py 内に実装
- bedrock  : `/model/{id}/invoke[-with-response-stream]` (AWS eventstream / JSON)

新プロバイダ (例: Vertex) を足す場合はこのパッケージに `<name>.py` を追加し、
`build_router()` で FastAPI router を返して fake_server から include する。
content blocks / 制御 API は共通なので、各アダプタは request decode と response encode のみ書く。
"""
