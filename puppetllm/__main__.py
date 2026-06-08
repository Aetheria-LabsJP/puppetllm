"""`python -m puppetllm` のエントリポイント。

fake_server.py を直接 `python -m puppetllm.fake_server` で起動すると、fake_server が
`__main__` として読み込まれた上で providers.bedrock 経由で `puppetllm.fake_server` として
再 import され、循環 import (build_router 未定義) になる。ここを薄い __main__ にして
fake_server を **常に `puppetllm.fake_server` としてのみ** import させることで回避する。
"""

import sys

from .fake_server import main

if __name__ == "__main__":
    sys.exit(main())
