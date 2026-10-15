"""Install recipe audit hooks without importing the full training stack."""

import os

if os.environ.get('W4_GRPO_AUDIT') == '1':
    try:
        from ds41_audit import install

        install()
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(86)
