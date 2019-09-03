#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from model.runners import Trainer
from keyboard.events import Keyboard
from screen.state import compare, Screenshot, ScreenshotProcessor
from screen.state import DIGITS, GAMEOVER

class Environment:

    def __init__(self, rom):
        if sys.platform not in ['darwin']:
            raise SystemError('sys.platform "{0}" not currently supported!'.format(sys.platform))
        self.rom = rom
        self.sdl_dir = '/Users/dascienz/DascienzProjects/sonic-reinforcement-learning/dgen-sdl-1.33'
        self.init_modes = ('train', 'test', 'play')
        self.keyboard = Keyboard()
        self.dy = None
    
    def save_state(self, slot):
        self.keyboard.key_press(str(slot))
        self.keyboard.key_press('o')

    def load_state(self, slot):
        self.keyboard.key_press(str(slot))
        self.keyboard.key_press('p')
    
    def start_game(self, slot, speed, reset=False):
        if not reset:
            subprocess.Popen(['./dgen', '-s', str(slot), '-H', str(speed * 60), self.rom], cwd=self.sdl_dir)
        else:
            subprocess.Popen(['./dgen', self.rom], cwd=self.sdl_dir)
            time.sleep(9.25)
            self.keyboard.key_down('.') 
            time.sleep(0.25)
            self.keyboard.key_up('.')
            time.sleep(0.25)
            self.save_state(0)
    
    def stop_game(self):
        try:
            self.keyboard.key_press('esc')
            raise KeyboardInterrupt('Game has been stopped.')
        except KeyboardInterrupt:
            pass
    
    def screen_init(self):
        print('Initializing screen area...')
        self.start_game(slot=0, speed=1, reset=False)
        time.sleep(1.0)
        self.dy = -100
        score = Screenshot(self.dy).score # get score
        while (score != 0):
            score = Screenshot(self.dy).score
            self.dy += 1  
        self.dy -= 1
        print('Done!')
        print()

    def train_model(self):
        self.screen_init()
        self.load_state(slot=0)
        # trainer = Trainer(dy=dy)
        # DO STUFF


    def test_model(self):
        pass
    
    def run_player(self):
        self.start_game(slot=0, speed=1.0, reset=True)

    def run_trainer(self, print_eval=False): 
        self.train_model()
        if print_eval:
            loop = True
            print('self.dy = %s' % (self.dy))
            while loop:
                s = Screenshot(self.dy)
                print("Score: %s\tTime: %s\tRings: %s" % (s.score, s.time, s.ring))
    
    def run_tester(self):
        self.test_model()