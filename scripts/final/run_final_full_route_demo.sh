#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec bash final_codes_docker/run_full_route_demo.sh "$@"

