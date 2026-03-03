#!/usr/bin/env python3
"""
Patch SGLang model_config.py to handle offline mode properly.

Run this script once after installing dependencies:
    python patch_sglang.py

This fixes an issue where HF_HUB_OFFLINE=1 causes sglang to crash when checking
for hf_quant_config.json. The patch changes from using HfApi.file_exists() to
using hf_hub_download() with proper exception handling for offline mode.

A backup file (.bak) is created before modifying.
"""

import sys
from pathlib import Path


def find_sglang_model_config():
    """Locate the sglang model_config.py file"""
    try:
        import sglang.srt.configs.model_config
        config_path = Path(sglang.srt.configs.model_config.__file__)

        if config_path.exists():
            return config_path
        else:
            print(f"❌ SGLang model_config.py not found at: {config_path}")
            return None
    except ImportError:
        print("❌ SGLang is not installed. Please install dependencies first:")
        print("   uv sync")
        return None


def check_already_patched(content):
    """Check if the file is already patched"""
    return 'OfflineModeIsEnabled' in content and 'hf_hub_download' in content


def apply_patch(config_path):
    """Apply the offline mode compatibility patch"""
    print(f"📝 Reading: {config_path}")

    with open(config_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if check_already_patched(content):
        print("✅ Already patched! No changes needed.")
        return True

    # Create backup
    backup_path = config_path.with_suffix('.py.bak')
    if not backup_path.exists():
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"💾 Backup created: {backup_path}")

    # Define the patch
    old_code = '''            if not is_local:
                from huggingface_hub import HfApi

                hf_api = HfApi()
                if hf_api.file_exists(self.model_path, "hf_quant_config.json"):
                    quant_cfg = modelopt_quant_config'''

    new_code = '''            if not is_local:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.errors import (
                    EntryNotFoundError,
                    OfflineModeIsEnabled,
                )

                try:
                    hf_hub_download(self.model_path, "hf_quant_config.json")
                except (OfflineModeIsEnabled, EntryNotFoundError):
                    pass
                else:
                    quant_cfg = modelopt_quant_config'''

    # Apply patch
    if old_code not in content:
        print("❌ Could not find expected code pattern. SGLang version may be different.")
        print("   Looking for pattern around line 370-375 with HfApi().file_exists()")
        return False

    patched_content = content.replace(old_code, new_code)

    # Write patched file
    with open(config_path, 'w') as f:
        f.write(patched_content)

    print(f"✅ Patched: {config_path}")
    return True


def verify_patch():
    """Verify the patch was applied correctly"""
    try:
        import sglang.srt.configs.model_config
        from pathlib import Path

        config_path = Path(sglang.srt.configs.model_config.__file__)
        with open(config_path, 'r') as f:
            content = f.read()

        if 'OfflineModeIsEnabled' in content and 'hf_hub_download' in content:
            print(f"✅ Verification successful: Offline mode handling is now compatible with HF_HUB_OFFLINE=1")
            return True
        else:
            print(f"❌ Verification failed: Patch may not have been applied correctly")
            return False
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    print("=" * 60)
    print("SGLang Offline Mode Compatibility Patch")
    print("=" * 60)
    print()

    # Find SGLang config
    config_path = find_sglang_model_config()
    if not config_path:
        sys.exit(1)

    # Apply patch
    if not apply_patch(config_path):
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
    print("SGLang can now work properly with HF_HUB_OFFLINE=1")
    print()
    print("To restore original SGLang:")
    print(f"  cp {config_path}.bak {config_path}")


if __name__ == "__main__":
    main()
