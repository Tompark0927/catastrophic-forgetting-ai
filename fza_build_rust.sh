#!/usr/bin/env bash
# fza_build_rust.sh — One-click build script for the FZA Rust libp2p core (v23.0)
# 
# Usage:
#   chmod +x fza_build_rust.sh
#   ./fza_build_rust.sh
#
# This script:
#   1. Checks for Rust toolchain.
#   2. Installs `maturin` into the active venv if missing.
#   3. Compiles the Rust code in `fza_core_rust/` into a Python .so extension.
#   4. Installs the module into the current Python venv.
#
# After a successful build, FZA will automatically use the faster Rust backend.

set -e

echo "==============================================" 
echo "🔨 FZA Phase XVII — Rust libp2p Core Builder"
echo "=============================================="

# Step 1: Check Rust
if ! command -v rustup &> /dev/null; then
    echo "❌ Rust is not installed!"
    echo "   Install it with:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "   Then re-run this script."
    exit 1
fi

echo "✅ Rust found: $(rustc --version)"
echo "✅ Cargo found: $(cargo --version)"

# Step 2: Add necessary Rust targets
rustup target add aarch64-apple-darwin 2>/dev/null || true
rustup target add x86_64-apple-darwin  2>/dev/null || true

# Step 3: Install maturin
echo ""
echo "📦 Installing maturin..."
pip install --upgrade maturin

# Step 4: Compile + install the Rust extension module
echo ""
echo "🦀 Compiling fza_core_rust → fza_mesh_rs.so..."
cd fza_core_rust

# `maturin develop --release` builds and installs the .so into the active venv.
# The Python process can then `import fza_mesh_rs` directly.
maturin develop --release

cd ..

echo ""
echo "✅ Build COMPLETE! The native Rust mesh node is now available."
echo "   Import it with: from fza_mesh_rs import FzaMeshNodeRs"
echo ""
echo "   Test it with: python -c \""
echo "   from fza_mesh_rs import FzaMeshNodeRs"
echo "   node = FzaMeshNodeRs('fza-local-test', '/ip4/0.0.0.0/tcp/9001')"
echo "   node.start()"
echo "   print(node.get_stats())"
echo "   \""
