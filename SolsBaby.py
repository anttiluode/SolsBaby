"""Convenient desktop launcher: python SolsBaby.py."""
import sys

from solsbaby.__main__ import main

if __name__ == '__main__':
    raise SystemExit(main(['gui', *sys.argv[1:]]))
