#!/usr/bin/env python3
"""
Patch Triton to support CUDA 13.0

Run this script once after installing dependencies:
    python patch_triton.py

The script will automatically find and patch your Triton installation.
A backup file (.bak) is created before modifying.
"""

import sys
from pathlib import Path

def find_triton_compiler():
    """Locate the Triton compiler.py file"""
    try:
        import triton
        triton_path = Path(triton.__file__).parent
        compiler_path = triton_path / "backends" / "nvidia" / "compiler.py"

        if compiler_path.exists():
            return compiler_path
        else:
            print(f"❌ Triton compiler not found at: {compiler_path}")
            return None
    except ImportError:
        print("❌ Triton is not installed. Please install dependencies first:")
        print("   uv sync")
        return None

def check_already_patched(content):
    """Check if the file is already patched"""
    return 'major == 13' in content or 'major >= 13' in content

def apply_patch(compiler_path):
    """Apply the CUDA 13.0 patch"""
    print(f"📝 Reading: {compiler_path}")

    with open(compiler_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if check_already_patched(content):
        print("✅ Already patched! No changes needed.")
        return True

    # Create backup
    backup_path = compiler_path.with_suffix('.py.bak')
    if not backup_path.exists():
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"💾 Backup created: {backup_path}")

    # Define the patch
    old_code = '''    if major == 12:
        if minor < 6:
            return 80 + minor
        else:
            return 80 + minor - 1
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor
    raise RuntimeError("Triton only support CUDA 10.0 or higher, but got CUDA version: " + cuda_version)'''

    new_code = '''    if major == 13:
        # CUDA 13.x support
        return 86 + minor
    if major == 12:
        if minor < 6:
            return 80 + minor
        else:
            return 80 + minor - 1
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor
    raise RuntimeError("Triton only support CUDA 10.0 or higher, but got CUDA version: " + cuda_version)'''

    # Apply patch
    if old_code not in content:
        print("❌ Could not find expected code pattern. Triton version may be different.")
        return False

    patched_content = content.replace(old_code, new_code)

    # Write patched file
    with open(compiler_path, 'w') as f:
        f.write(patched_content)

    print(f"✅ Patched: {compiler_path}")
    return True

def verify_patch():
    """Verify the patch works"""
    try:
        from triton.backends.nvidia.compiler import ptx_get_version
        version = ptx_get_version("13.0")
        print(f"✅ Verification successful: CUDA 13.0 -> PTX {version}")
        return True
    except RuntimeError as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Triton CUDA 13.0 Patch")
    print("=" * 60)
    print()

    # Find Triton
    compiler_path = find_triton_compiler()
    if not compiler_path:
        sys.exit(1)

    # Apply patch
    if not apply_patch(compiler_path):
        sys.exit(1)

    # Verify
    print()
    if not verify_patch():
        sys.exit(1)

    print()
    print("=" * 60)
    print("✅ Patch completed successfully!")
    print("=" * 60)
    print()
    print("You can now run training:")
    print("  bash external/verl/run_1node.sh")
    print()
    print("To restore original Triton:")
    print(f"  cp {compiler_path}.bak {compiler_path}")

if __name__ == "__main__":
    main()
