"""Entry point for `python -m puppetllm`.

If fake_server.py is launched directly via `python -m puppetllm.fake_server`, it is
loaded as `__main__` and then re-imported as `puppetllm.fake_server` through
providers.bedrock, causing a circular import (build_router undefined). This thin
__main__ avoids that by ensuring fake_server is **always imported only as
`puppetllm.fake_server`**.
"""

import sys

from .fake_server import main

if __name__ == "__main__":
    sys.exit(main())
