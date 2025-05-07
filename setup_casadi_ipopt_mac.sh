#!/bin/bash

# Exit on error
set -e

echo "Setting up CasADi IPOPT linkage..."

# Find the GCC version installed via Homebrew
GCC_DIR=$(brew --prefix gcc)
LIBGFORTRAN=$(find "$GCC_DIR"/lib/gcc/* -name libgfortran.5.dylib | head -n 1)

if [ -z "$LIBGFORTRAN" ]; then
    echo "libgfortran.5.dylib not found. Make sure you've run: brew install gcc"
    exit 1
fi

LIBGFORTRAN_DIR=$(dirname "$LIBGFORTRAN")
echo "Found libgfortran at: $LIBGFORTRAN"

# Export DYLD_LIBRARY_PATH for current shell
export DYLD_LIBRARY_PATH="$LIBGFORTRAN_DIR:$DYLD_LIBRARY_PATH"
echo "Exported DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"

# Optionally persist to shell profile
SHELL_RC="$HOME/.zshrc"
if [[ $SHELL == *bash ]]; then
    SHELL_RC="$HOME/.bash_profile"
fi

if ! grep -q "$LIBGFORTRAN_DIR" "$SHELL_RC"; then
    echo "Adding DYLD_LIBRARY_PATH to $SHELL_RC"
    echo "export DYLD_LIBRARY_PATH=\"$LIBGFORTRAN_DIR:\$DYLD_LIBRARY_PATH\"" >> "$SHELL_RC"
else
    echo "DYLD_LIBRARY_PATH already present in $SHELL_RC"
fi

# Patch CasADi’s libipopt.3.dylib to hardcode the path (optional)
# CASADI_PATH=$(python3 -c "import casadi; import os; print(os.path.dirname(casadi.__file__))")
# IPOPT_LIB="$CASADI_PATH/libipopt.3.dylib"

# if [ -f "$IPOPT_LIB" ]; then
#     echo "Patching libipopt.3.dylib..."
#     install_name_tool -change "@rpath/libgfortran.5.dylib" "$LIBGFORTRAN" "$IPOPT_LIB"
#     echo "Patched libipopt.3.dylib"
# else
#     echo "Could not find libipopt.3.dylib in CasADi path: $CASADI_PATH"
#     exit 1
# fi

echo "Setup complete. Restart your terminal or source your shell config to apply changes."
