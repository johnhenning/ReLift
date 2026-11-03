#!/usr/bin/env python
"""Cache the pretrained backbone loaded and discarded by historical AllTracker.

Run outside benchmark timing. Current inference skips this initialization and
does not need this checkpoint; frozen reference implementations still load it.
"""

from __future__ import annotations

from torchvision.models import ConvNeXt_Tiny_Weights


def main() -> None:
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    weights.get_state_dict(progress=True, check_hash=True)
    print(f"Historical baseline backbone cached: {weights.url}")


if __name__ == "__main__":
    main()
