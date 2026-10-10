#!/usr/bin/env python3
"""Consumer entry point. All installer messages and documentation are English."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))
from distribution import main

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped. Run setup again to resume downloads and preparation.')
        raise SystemExit(130)
    except Exception as error:
        print('Installation/startup failed: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
