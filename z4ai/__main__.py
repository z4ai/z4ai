# SPDX-License-Identifier: Apache-2.0

"""Enable ``python -m z4ai`` to invoke the command-line interface."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
