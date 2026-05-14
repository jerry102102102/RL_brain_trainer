#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec bash final_codes_docker/run_local_test_demo.sh "$@"

