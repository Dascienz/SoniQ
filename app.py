#!/usr/bin/env python3
import os
import sys
from emulator.env import Environment

def main():
    env = Environment(rom='SonictheHedgehog.bin')
    env.run_trainer()

if __name__ == '__main__':
    main()
    