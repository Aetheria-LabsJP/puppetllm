"""The `puppetllm` console command / `python -m puppetllm`.

    puppetllm serve [--port 8765] [--pending-timeout 30] [--default-response TEXT]
                    [--on-unmatched pending|default|error] [--seed N]
                    [--config FILE] [--rules FILE]
    puppetllm relay --kind openai ...        (the relay responder, see puppetllm.relay)
    puppetllm wait  [--url http://127.0.0.1:8765] [--timeout 30]
    puppetllm --version

`serve` is the default: `python -m puppetllm --port 9000` still works.

fake_server.py must be imported only as `puppetllm.fake_server` (running it directly as
`__main__` re-imports it through the providers and breaks), which is why this module
exists.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__

_SUBCOMMANDS = ("serve", "relay", "wait")


def build_parser() -> argparse.ArgumentParser:
    from .fake_server import add_serve_arguments

    parser = argparse.ArgumentParser(
        prog="puppetllm",
        description="Fake Anthropic / Bedrock / OpenAI API server for debugging and testing")
    parser.add_argument("--version", action="version", version=f"puppetllm {__version__}")
    sub = parser.add_subparsers(dest="command")
    add_serve_arguments(sub.add_parser("serve", help="run the fake server (default)"))
    sub.add_parser("relay", help="relay responder: forward pendings to a real API",
                   add_help=False)
    wait = sub.add_parser("wait", help="block until a server answers /_control/health")
    wait.add_argument("--url", default="http://127.0.0.1:8765")
    wait.add_argument("--timeout", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "relay":
        from .relay import main as relay_main
        return relay_main(argv[1:])
    if not argv or argv[0] not in _SUBCOMMANDS and argv[0] not in ("-h", "--help", "--version"):
        argv = ["serve", *argv]
    args = build_parser().parse_args(argv)
    if args.command == "wait":
        from .testing import Puppet, PuppetError
        try:
            Puppet(args.url).wait_ready(args.timeout)
        except PuppetError as e:
            print(f"[puppetllm] {e}", file=sys.stderr)
            return 1
        return 0
    from .fake_server import serve
    return serve(args)


if __name__ == "__main__":
    sys.exit(main())
