#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: ./start-agent-sdk.sh [--model MODEL] [--effort EFFORT]

Build and start the API server, Agent SDK relay, and status watchdog.
No host Python virtual environment is required.

  --model MODEL  Override the upstream model (for example claude-sonnet-4-6).
  --effort LEVEL Override reasoning effort: auto, low, medium, high, xhigh, max.
  -h, --help     Show this help.

The default model is claude-opus-5-5. PUPPETLLM_AGENT_SDK_MODEL in the
environment or .env can also set a default; --model takes precedence.
The default effort is high. PUPPETLLM_AGENT_SDK_EFFORT in the environment
or .env can override it; --effort takes precedence and accepts mixed case.
Set the real upstream ANTHROPIC_API_KEY in the environment or .env first.
USAGE
}

set_effort() {
    effort="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    case "$effort" in
        auto|low|medium|high|xhigh|max)
            export PUPPETLLM_AGENT_SDK_EFFORT="$effort"
            ;;
        *)
            printf '%s\n' 'Error: --effort must be auto, low, medium, high, xhigh, or max.' >&2
            exit 2
            ;;
    esac
}

while (($#)); do
    case "$1" in
        --model)
            if (($# < 2)) || [[ -z "$2" || "$2" == -* ]]; then
                printf '%s\n' 'Error: --model requires a model name.' >&2
                exit 2
            fi
            export PUPPETLLM_AGENT_SDK_MODEL="$2"
            shift 2
            ;;
        --model=*)
            model="${1#--model=}"
            if [[ -z "$model" || "$model" == -* ]]; then
                printf '%s\n' 'Error: --model requires a model name.' >&2
                exit 2
            fi
            export PUPPETLLM_AGENT_SDK_MODEL="$model"
            shift
            ;;
        --effort)
            if (($# < 2)) || [[ -z "$2" || "$2" == -* ]]; then
                printf '%s\n' 'Error: --effort requires an effort level.' >&2
                exit 2
            fi
            set_effort "$2"
            shift 2
            ;;
        --effort=*)
            set_effort "${1#--effort=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Error: unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
docker compose --profile agent-sdk build proxy agent-sdk-relay
# Keep an existing server's in-memory history when updating only the adapter.
docker compose up -d --wait --no-recreate proxy
exec docker compose --profile agent-sdk up -d --no-build --no-deps agent-sdk-relay agent-sdk-watchdog
