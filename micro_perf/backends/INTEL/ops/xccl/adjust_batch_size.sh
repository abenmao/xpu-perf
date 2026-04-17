#!/bin/bash
# Script to adjust batch_size upper limit in xccl_ops JSON test files
# for Intel B60 GPU performance tuning.
#
# Usage:
#   ./adjust_batch_size.sh b60      # Apply B60 limits
#   ./adjust_batch_size.sh restore  # Restore original limits (2097152)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOAD_DIR="${SCRIPT_DIR}/../../../../workloads/xccl_ops"

ORIGINAL_LIMIT=2097152

# B60-specific limits
declare -A B60_LIMITS=(
    ["all_reduce.json"]=524288
    ["all_gather.json"]=1048576
    ["reduce_scatter.json"]=1048576
    ["all_to_all.json"]=1048576
)

FILES=("all_reduce.json" "all_gather.json" "reduce_scatter.json" "all_to_all.json")

COMPACT_JSON_HELPER='
import json

def compact_json(obj, indent=0):
    sp = "    " * indent
    sp1 = "    " * (indent + 1)
    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            items.append(f"{sp1}{json.dumps(k)}: {compact_json(v, indent + 1).lstrip()}")
        return "{\n" + ",\n".join(items) + f"\n{sp}}}"
    elif isinstance(obj, list):
        if all(not isinstance(e, (dict, list)) for e in obj):
            return json.dumps(obj)
        else:
            items = []
            for e in obj:
                items.append(f"{sp1}{compact_json(e, indent + 1).lstrip()}")
            return "[\n" + ",\n".join(items) + f"\n{sp}]"
    else:
        return json.dumps(obj)
'

set_batch_size_limit() {
    local file="$1"
    local limit="$2"
    local filepath="${WORKLOAD_DIR}/${file}"

    if [ ! -f "$filepath" ]; then
        echo "[ERROR] File not found: $filepath"
        return 1
    fi

    python3 -c "
import json, sys
${COMPACT_JSON_HELPER}

filepath = sys.argv[1]
limit = int(sys.argv[2])

with open(filepath, 'r') as f:
    data = json.load(f)

for case in data['cases']:
    case['batch_size'] = [v for v in case['batch_size'] if v <= limit]

with open(filepath, 'w') as f:
    f.write(compact_json(data) + '\n')

print(f'  {sys.argv[3]}: batch_size upper limit -> {limit}')
" "$filepath" "$limit" "$file"
}

restore_batch_size() {
    local file="$1"
    local filepath="${WORKLOAD_DIR}/${file}"

    if [ ! -f "$filepath" ]; then
        echo "[ERROR] File not found: $filepath"
        return 1
    fi

    python3 -c "
import json, sys
${COMPACT_JSON_HELPER}

filepath = sys.argv[1]
original_limit = int(sys.argv[2])

# Full original batch_size sequence: powers of 2 from 1 to original_limit
full_list = []
v = 1
while v <= original_limit:
    full_list.append(v)
    v *= 2

with open(filepath, 'r') as f:
    data = json.load(f)

for case in data['cases']:
    case['batch_size'] = full_list

with open(filepath, 'w') as f:
    f.write(compact_json(data) + '\n')

print(f'  {sys.argv[3]}: batch_size upper limit -> {original_limit} (restored)')
" "$filepath" "$ORIGINAL_LIMIT" "$file"
}

usage() {
    echo "Usage: $0 {b60|restore}"
    echo ""
    echo "  b60      Apply Intel B60 GPU batch_size limits:"
    echo "             all_reduce.json:      524288"
    echo "             all_gather.json:      1048576"
    echo "             reduce_scatter.json:  1048576"
    echo "             all_to_all.json:      1048576"
    echo ""
    echo "  restore  Restore original batch_size upper limit (2097152) for all files"
}

case "$1" in
    b60)
        echo "Applying Intel B60 GPU batch_size limits..."
        for file in "${FILES[@]}"; do
            set_batch_size_limit "$file" "${B60_LIMITS[$file]}"
        done
        echo "Done."
        ;;
    restore)
        echo "Restoring original batch_size limits..."
        for file in "${FILES[@]}"; do
            restore_batch_size "$file"
        done
        echo "Done."
        ;;
    *)
        usage
        exit 1
        ;;
esac
