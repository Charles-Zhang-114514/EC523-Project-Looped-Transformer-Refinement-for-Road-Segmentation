#!/bin/bash
# Source this tracked file from an SCC Bash shell to activate the project.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' 'Use: source scripts/activate_scc.sh' >&2
    exit 1
fi

_ec523_activate_scc() {
    local ec523_repo_dir
    ec523_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return 1

    # Optional personal path overrides; shared environment choices stay here.
    if [[ -f "$ec523_repo_dir/env.sh" ]]; then
        source "$ec523_repo_dir/env.sh" || return 1
    fi

    module load miniconda/25.3.1 || return 1
    module load academic-ml/fall-2026 || return 1
    conda activate fall-2026-pyt || return 1

    export PYTHONNOUSERSITE=1
    export EC523_PROJECT_DIR="$ec523_repo_dir"
    export PYTHONPATH="$EC523_PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
    export EC523_DATA_DIR="${EC523_DATA_DIR:-/projectnb/ec523kb/projects/RoadSeg/dataset}"
    export EC523_STARTER_DIR="${EC523_STARTER_DIR:-$(dirname -- "$ec523_repo_dir")/project_a/cli}"
    export TMPDIR="$EC523_PROJECT_DIR/tmp"
    export TMP="$TMPDIR"
    export TEMP="$TMPDIR"
    export XDG_CACHE_HOME="$EC523_PROJECT_DIR/cache/xdg"
    export TORCH_HOME="$EC523_PROJECT_DIR/cache/torch"
    export HF_HOME="$EC523_PROJECT_DIR/cache/huggingface"
    export PYTHONPYCACHEPREFIX="$EC523_PROJECT_DIR/cache/pycache"
    mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" \
        "$HF_HOME" "$PYTHONPYCACHEPREFIX" || return 1
}

if _ec523_activate_scc; then
    unset -f _ec523_activate_scc
else
    unset -f _ec523_activate_scc
    return 1
fi
