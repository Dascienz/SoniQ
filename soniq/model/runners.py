#!/usr/bin/env python3
import os
import cv2
import csv
import time
import pickle
import random
import numpy as np
import tensorflow as tf
from collections import deque
# project modules
from model.base import Model
from keyboard.events import Keyboard
from screen.state import compare, Screenshot, ScreenshotProcessor

wdir = os.path.dirname(__file__)
parent_dir = os.path.dirname(wdir)
data_dir = os.path.join(parent_dir, 'data')
pickle_dir = os.path.join(data_dir, 'pickle')
checkpoint_dir = os.path.join(wdir, '.model_checkpoints')

class Trainer:

    def __init__(self, model, environment, dy):
        self.dy = dy
        self.model = model
        self.keyboard = Keyboard()
        self.environment = environment
        self.checkpoints = checkpoint_dir
        self.processor = ScreenshotProcessor(resize_x=80, resize_y=80)
    
    def act(self, c):
        if c == 0:
            self.keyboard.key_up('a')
            self.keyboard.key_down('d')
            return
        if c == 1:
            self.keyboard.key_up('d')
            self.keyboard.key_down('a')
            return
        if c == 2:
            self.keyboard.key_up('s')
            self.keyboard.key_down('m')
            return
        if c == 3:
            self.keyboard.key_up('m')
            self.keyboard.key_down('s')
            return
    
    def get_state(self, frames):
        if frames:
            frames.pop(0)
            screen = Screenshot(self.dy)
            state, score = screen.state, screen.score
            state = self.processor.transform(state)
            frames.append(state)
            state = np.stack(frames, axis=2)
            return (state, score)
        else:
            for _ in range(4):
                state = Screenshot(self.dy).state
                state = self.processor.transform(state)
                frames.append(state)
            score = Screenshot(self.dy).score
            state = np.stack(frames, axis=2)
            return (state, score)

    def run(self):
        actions = 4
        gamma = 0.95
        epsilon = 1.0
        observe = 1000
        memory = 10000
        batch_size = 10
        episodes = 7500
        model = Model(batch_size=batch_size, 
                      actions=actions, 
                      input_shape=[None, 80, 80, 4], 
                      learning_rate=1e-2)
        with tf.Session as sess:
            check = False
            sess.run(model.initialize)
            if check:
                new_saver = tf.train.saver()
                checkpoint = tf.train.get_checkpoint_state(self.checkpoints)
                if checkpoint and checkpoint.model_checkpoint_path:
                    new_saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                saver = tf.train.saver()
            #Q-reinforcement algorithm for training the network.
            replay = deque()
            scores, losses = [], []
            time.sleep(0.25)

            print()
            print('*----- GAME START -----*')

            for e in range(episodes):
                if e%100 == 0:
                    saver.save(sess, os.path.join(checkpoint_dir, 'model_{}.ckpt'.format(e)))

                count = 0
                terminal = 0 
                done = False 
                frames = []
                self.environment.load_state(slot=0)

                print()
                print('Episode: {}, Random %: {}'.format(str(e+1), str(100*epsilon)))

                # We are in initial state S
                state, score = self.get_state(frames)
                while score < 0:
                    state, score = self.get_state(frames)

                while not done: 
                    # Run our Q function on S to get Q values for all possible actions.
                    if (np.random.random() <= epsilon): # choose random action
                        action = np.zeros((actions))
                        action[np.random.randint(0, 4)] = 1
                        action = np.argmax(action)
                    else: # choose best action from Q(s,a) values
                        action, Qvals = sess.run([predict, readout], feed_dict={x: [state]}) # Q-values
                    self.act(action) #Take action.

                    #Observe new state and score.
                    new_state, new_score = self.get_states(frames)
                    while new_score < 0:
                        new_state, new_score = self.get_states(frames) # new state, new score

                    # check gameover screen
                    if (Screenshot(self.dy).is_gameover() <= 1065):
                        reward, count, terminal = -1, 0, 1
                        replay.append((state, action, reward, new_state, terminal))
                        done = True

                    diff = new_score - score

                    if diff == 0:
                        count += 1
                        #reward, terminal = 0, 0
                        #replay.append((state, action, reward, new_state, terminal))
                        if count > 300:
                            done = True
                    elif diff > 0:
                        reward, count, terminal = 1, 0, 0
                        replay.append((state, action, reward, new_state, terminal))
                    elif diff < 0:
                        reward, count, terminal = 0, 0, 0
                        replay.append((state, action, reward, new_state, terminal))
                    # state and score reset
                    state, score = new_state, new_score
                # add to score list
                scores.append(score)
                
                try:
                    print("Qval Sample: {}".format(Qvals))
                except:
                    pass

                #Pop first transition from replay memory.
                if (len(replay) > memory):
                    replay.popleft()
                #Replay memory loop.
                if (len(replay) >= observe):
                    batch = random.sample(replay, batch_size)
                    state_batch = [i[0] for i in batch] #states
                    action_batch = [i[1] for i in batch] #actions
                    reward_batch = [i[2] for i in batch] #rewards
                    new_state_batch = [i[3] for i in batch] #new states
                    terminal_batch = [i[4] for i in batch] #terminal 0,1
                    y_batch = [] #target Q-values
                    predictions = sess.run(readout, feed_dict={x: new_state_batch}) #predictions

                    for idx in range(len(batch)):
                        if terminal_batch[idx] == 1: #is terminal state
                            y_batch.append(reward_batch[idx]) #reward only
                        else: #is not terminal state
                            y_batch.append(reward_batch[idx] + gamma*np.max(predictions[idx])) #discounted future reward

                    y_batch = y_batch / np.max(y_batch) #Avoid growing Q-values
                    
                    sess.run(train_step, feed_dict={x: state_batch, a: action_batch, y: y_batch}) #update model
                    
                    l = sess.run(loss, feed_dict={x: state_batch, a: action_batch, y: y_batch}) #record loss
                    
                    print("Loss: ", l)
                    losses.append(l)
                #End of replay memory loop.

                #Epsilon step down to decrease random actions.
                if (EPSILON > 0.1) and (len(replay) > OBSERVE):
                    EPSILON -= (1/EPISODES)

            #End of episodes.
            sm.closeGame()

            now = time.time()

            #Write training parameters to file.
            with open("Training/parameters_%s.txt" % (now), "w") as outfile:
                outfile.write("episodes\t%s" % EPISODES)
                outfile.write("\ngamma\t%s" % GAMMA)
                outfile.write("\nbatch\t%s" % BATCH)
                outfile.write("\nmemory\t%s" % MEMORY)
                outfile.write("\nstddev\t%s" % STDDEV)
                outfile.write("\nlearning_rate\t%s" % LEARNING_RATE)

            #Write scores to file.
            with open("Training/scores_%s.csv" % (now), "w") as outfile:
                writer = csv.writer(outfile)
                for score in scores:
                    writer.writerow([score])

            #Write training loss to file.
            with open("Training/losses_%s.csv" % (now), "w") as outfile:
                writer = csv.writer(outfile)
                for loss in losses:
                    writer.writerow([loss])
        return

def testNetwork():
    with tf.Session() as sess:
        ACTIONS = 4
        STDDEV = 0.1
        
        #Build the convolutional neural network.
        x = tf.placeholder(dtype=tf.float32, shape = [None, 80, 80, 1]) # input observation

        W_conv1 = tf.Variable(tf.truncated_normal(shape=[8,8,1,32], stddev=STDDEV))
        b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1,strides=[1,4,4,1],padding='SAME')+b_conv1)
        #h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        W_conv2 = tf.Variable(tf.truncated_normal(shape=[4,4,32,64], stddev=STDDEV))
        b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1,W_conv2,strides=[1,2,2,1],padding='SAME')+b_conv2)

        W_conv3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,64], stddev=STDDEV))
        b_conv3 = tf.Variable(tf.constant(0.01,shape=[64]))
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2,W_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)
        h_conv3_flat = tf.reshape(h_conv3,[-1,1600])

        W_fc1 = tf.Variable(tf.truncated_normal(shape=[1600,512], stddev=STDDEV))
        b_fc1 = tf.Variable(tf.constant(0.01, shape=[512]))
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1)+b_fc1)

        W_fc2 = tf.Variable(tf.truncated_normal(shape=[512,ACTIONS], stddev=STDDEV))
        b_fc2 = tf.Variable(tf.constant(0.01, shape=[ACTIONS]))

        prediction = (tf.matmul(h_fc1, W_fc2) + b_fc2)

        sess.run(tf.global_variables_initializer())

        episode = 4900
        saved_graph = tf.train.import_meta_graph("SavedNetworks/model_{}.ckpt.meta".format(episode))
        saved_graph.restore(sess, "SavedNetworks/model_{}.ckpt".format(episode))
        
        initiateGame()

        time.sleep(1.5)

        play = True
        while play:
            state, score = getState(dy) # initial state, initial score
            qvals = sess.run(prediction, feed_dict={x: [state]})[0]
            action = np.argmax(qvals)
            buttonPress(action)
            if(gameOverCheck(observationGrab(), dy) <= 1065):
                play = False
                sm.closeGame()
