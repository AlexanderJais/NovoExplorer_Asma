#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== NovoExplorer Setup ==="

# 1. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

source .venv/bin/activate

# 2. Ensure build toolchains that some dependencies need when no wheel
#    matches the current platform (most notably gseapy on Intel macOS, which
#    publishes no x86_64 wheels and falls back to a Rust-based sdist build).
ensure_rust_toolchain() {
    if command -v cargo >/dev/null 2>&1; then
        return 0
    fi
    if [ -f "$HOME/.cargo/env" ]; then
        # shellcheck source=/dev/null
        . "$HOME/.cargo/env"
        command -v cargo >/dev/null 2>&1 && return 0
    fi
    echo "Rust toolchain (cargo) not found; installing via rustup..."
    if ! command -v curl >/dev/null 2>&1; then
        echo "  Warning: curl is required to install Rust automatically. Please install Rust manually from https://rustup.rs" >&2
        return 1
    fi
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
    # shellcheck source=/dev/null
    . "$HOME/.cargo/env"
}

case "$(uname -s)" in
    Darwin)
        # gseapy does not ship a macOS x86_64 wheel on PyPI for any 1.x
        # release, and even on arm64 older Python versions may miss wheels,
        # so a Rust toolchain is the reliable fallback for pip's sdist build.
        if ! ensure_rust_toolchain; then
            echo "  Continuing without Rust; gseapy install may fail." >&2
        fi
        ;;
esac

# 3. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install --prefer-binary -r requirements.txt

# 4. Create resource directories
mkdir -p resources/gene_sets

# 5. Download gene set files if not present
echo "Checking gene set files..."

GENE_SETS_DIR="resources/gene_sets"

if [ ! -f "$GENE_SETS_DIR/MSigDB_Hallmark_2020.gmt" ]; then
    echo "Downloading MSigDB Hallmark gene sets..."
    python3 -c "
import gseapy
try:
    lib = gseapy.get_library('MSigDB_Hallmark_2020', organism='Mouse')
    with open('$GENE_SETS_DIR/MSigDB_Hallmark_2020.gmt', 'w') as f:
        for term, genes in lib.items():
            f.write(term + '\tna\t' + '\t'.join(genes) + '\n')
    print('  Downloaded MSigDB_Hallmark_2020')
except Exception as e:
    print(f'  Warning: Could not download MSigDB_Hallmark_2020: {e}')
"
fi

if [ ! -f "$GENE_SETS_DIR/GO_Biological_Process_2023.gmt" ]; then
    echo "Downloading GO Biological Process gene sets..."
    python3 -c "
import gseapy
try:
    lib = gseapy.get_library('GO_Biological_Process_2023', organism='Mouse')
    with open('$GENE_SETS_DIR/GO_Biological_Process_2023.gmt', 'w') as f:
        for term, genes in lib.items():
            f.write(term + '\tna\t' + '\t'.join(genes) + '\n')
    print('  Downloaded GO_Biological_Process_2023')
except Exception as e:
    print(f'  Warning: Could not download GO_Biological_Process_2023: {e}')
"
fi

if [ ! -f "$GENE_SETS_DIR/KEGG_2021_Mouse.gmt" ]; then
    echo "Downloading KEGG gene sets..."
    python3 -c "
import gseapy
try:
    lib = gseapy.get_library('KEGG_2021_Mouse', organism='Mouse')
    with open('$GENE_SETS_DIR/KEGG_2021_Mouse.gmt', 'w') as f:
        for term, genes in lib.items():
            f.write(term + '\tna\t' + '\t'.join(genes) + '\n')
    print('  Downloaded KEGG_2021_Mouse')
except Exception as e:
    print(f'  Warning: Could not download KEGG_2021_Mouse: {e}')
"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "       source .venv/bin/activate"
echo ""
echo "  2. Launch the app:"
echo "       streamlit run novogene_explorer.py -- /path/to/your/novogene/results"
echo ""
echo "     Or without arguments to pick the folder in the browser:"
echo "       streamlit run novogene_explorer.py"
