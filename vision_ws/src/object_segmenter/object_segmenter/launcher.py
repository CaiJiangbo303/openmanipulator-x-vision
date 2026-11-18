#!/usr/bin/env python3

import os
from pathlib import Path


def main():
    python_exe = (
        Path.home()
        / 'openmanipulator-x-vision'
        / '.venv-sam2'
        / 'bin'
        / 'python'
    )

    if not python_exe.exists():
        raise RuntimeError(
            f'SAM2 Python not found: {python_exe}'
        )

    os.execv(
        str(python_exe),
        [
            str(python_exe),
            '-m',
            'object_segmenter.segmenter_node',
        ]
    )


if __name__ == '__main__':
    main()
