import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Concatenate, Reshape
from multiprocessing.pool import ThreadPool
from threading import Lock
import numpy as np
import math
from os import path
import time
import tqdm
from itertools import product
from logic2048 import *
from cfgs.config import cfg

class Simple_RLAgent(Model):
    def __init__(self, train):
        '''A class which build a simple RL agent to play the 2048 game.
        :param train: bool value. True if want to train new model
        '''
        super(Simple_RLAgent, self).__init__()
        convLayers = []    #self.convLayers[i][j] is the ith layer, jth conv
        assert len(cfg.MODEL.CONV_NUM_FILTER)==cfg.MODEL.NUM_CONV_LAYERS
        assert cfg.MODEL.NUM_CONV_LAYERS == 2 # since reducing the activation map further doesn't make sense
        filter_size = cfg.MODEL.CONV_NUM_FILTER
        for i in range(cfg.MODEL.NUM_CONV_LAYERS):
            conv1 = Conv2D(filter_size[i], kernel_size=(1,2), activation='relu')
            conv2 = Conv2D(filter_size[i], kernel_size=(2,1), activation='relu')
            convLayers.append([conv1, conv2])
        
        expanded_dim =  2*4*filter_size[1]*2 + 3*3*filter_size[1]*2 + 4*3*filter_size[0]*2
        hiddenLayer = Dense(units=cfg.MODEL.FULL_CONNECTED_LAYER_SIZE, activation='relu')
        outLayer = Dense(units=cfg.MODEL.OUTPUT_SIZE)

        self.convLayers = convLayers
        self.hiddenLayer = hiddenLayer
        self.outLayer = outLayer
        # self.init_model()
        self.training = train

    def init_model(self):
        '''Construct the tensorflow model
        '''
        convLayers = []    #self.convLayers[i][j] is the ith layer, jth conv
        assert len(cfg.MODEL.CONV_NUM_FILTER)==cfg.MODEL.NUM_CONV_LAYERS
        assert cfg.MODEL.NUM_CONV_LAYERS == 2 # since reducing the activation map further doesn't make sense
        filter_size = cfg.MODEL.CONV_NUM_FILTER
        for i in range(cfg.MODEL.NUM_CONV_LAYERS):
            conv1 = Conv2D(filter_size[i], kernel_size=(1,2), activation='relu')
            conv2 = Conv2D(filter_size[i], kernel_size=(2,1), activation='relu')
            convLayers.append([conv1, conv2])
        
        expanded_dim =  2*4*filter_size[1]*2 + 3*3*filter_size[1]*2 + 4*3*filter_size[0]*2
        hiddenLayer = Dense(units=cfg.MODEL.FULL_CONNECTED_LAYER_SIZE, activation='relu')
        outLayer = Dense(units=cfg.MODEL.OUTPUT_SIZE)

        self.convLayers = convLayers
        self.hiddenLayer = hiddenLayer
        self.outLayer = outLayer
        return

    def call(self, x):
        '''Run the constructed model on input x
        '''
        map1 = self.convLayers[0][0](x)
        map2 = self.convLayers[0][1](x)
        map3 = self.convLayers[1][0](map1)
        map4 = self.convLayers[1][1](map1)
        map5 = self.convLayers[1][0](map2)
        map6 = self.convLayers[1][1](map2)
        reshape_layer = Reshape((-1,))

        concat_map = Concatenate(axis=-1)([reshape_layer(map1), reshape_layer(map2),\
                    reshape_layer(map3), reshape_layer(map4), reshape_layer(map5), reshape_layer(map6)])
        hidden_map = self.hiddenLayer(concat_map)
        logits = self.outLayer(hidden_map)

        return logits

    def power_grid(self, grid, max_power=16):
        '''Convert the grid of 2048 game into a power of 2 grid
        :param grid: (GRID_LENxGRID_LEN) numpy matrix of game grid
        :param max_power: Maximum power of 2 possible on grid. Default 16
        :return power_grid: (1x4x4x16) matrix where element (1,i,j,k)=1.0 if grid[i,j]=2^k. 0 otherwise
        '''
        power_grid = np.zeros((cfg.GAME_CONFIG.GRID_LEN, cfg.GAME_CONFIG.GRID_LEN, max_power), dtype=np.float32)
        temp_grid = grid.copy()
        temp_grid[temp_grid==0] = 1
        log_grid = np.log2(temp_grid).astype(np.int)
        idx = np.arange(cfg.GAME_CONFIG.GRID_LEN)
        for i,j in product(idx, idx):
            power_grid[i, j, log_grid[i,j]] = 1.0
        return power_grid

def play_single_game(f_in):
    model, epsilon, controls = f_in
    grid = new_game(cfg.GAME_CONFIG.GRID_LEN)
    finish = False
    ep_score, n_iter = 0, 0
    replay_labels, replay_memory = [], []

    while not finish:
        # run model and select move with highest Q value
        prev_grid = grid.copy()
        state = model.power_grid(prev_grid)
        lock.acquire()
        control_scores = model(np.expand_dims(state, axis=0))
        lock.release()
        control_buttons = np.flip(np.argsort(control_scores),axis=1)
        # copy the Q-values as labels, for use in Bellman update
        labels = control_scores[0].numpy().copy()
        prev_max = np.max(prev_grid)
        # follow greedy or explore?
        if np.random.rand()<epsilon:    # explore / random move
            legal_moves = []
            for i in range(4):
                temp_grid = prev_grid.copy()
                temp_grid, changed, _ = controls[i](temp_grid)
                if changed: legal_moves.append(i)
                else: continue
            if len(legal_moves)==0:
                finish = True
                continue

            # apply random move
            move = np.random.choice(legal_moves)
            temp_grid = prev_grid.copy()
            temp_grid, _, score = controls[move](temp_grid) 
        else:                           # make greedy move with max expected reward
            for move in control_buttons[0]:
                prev_state = prev_grid.copy()
                temp_grid, changed, score = controls[move](prev_state)
                if not changed: #illegal move
                    labels[move] = 0
                    continue
                else:
                    break

        n_merges = countEmptyCells(temp_grid) - countEmptyCells(prev_grid)
        finish = game_over(temp_grid)
        if not finish: temp_grid = insert_new(temp_grid)
        grid = temp_grid.copy()
        ep_score += score
        next_max = np.max(temp_grid)

        #update reward
        if next_max == prev_max: labels[move] = 0
        else: labels[move] = (math.log(next_max,2))*0.1
        labels[move] += n_merges        # having more empty squares is better

        # get the next state max Q value
        temp_grid = model.power_grid(temp_grid)
        lock.acquire()
        temp_scores = model(np.expand_dims(temp_grid, axis=0))
        lock.release()
        max_qval = np.max(temp_scores)
        labels[move] += cfg.MODEL.TRAIN.GAMMA*max_qval

        # insert episode into replay memory
        prev_state = model.power_grid(prev_grid)
        replay_labels.append(labels)
        replay_memory.append(prev_state)
        
        n_iter += 1

    return replay_memory, replay_labels, n_iter, ep_score, next_max

def generate_replay_dataset(model):
    n_iter = 0
    NUM_EPS_PER_ITER = 10
    max_score, max_tile= -1, -1
    replay_memory = []
    replay_labels = []
    controls = {0:up, 1:left, 2:right, 3:down}
    epsilon = cfg.MODEL.TRAIN.EPSILON
    pool = ThreadPool(cfg.MODEL.TRAIN.NUM_WORKERS)

    global pbar, train_loss
    ep_finished = 0
    while ep_finished < cfg.MODEL.TRAIN.NUM_EPISODES:
        rtn_arr = pool.map(play_single_game, [(model, epsilon, controls) for _ in range(NUM_EPS_PER_ITER*cfg.MODEL.TRAIN.NUM_WORKERS)])
        ep_replay_mem, ep_replay_labels, ep_iters, ep_scores, ep_maxs = zip(*rtn_arr)
        n_iter += sum(ep_iters)
        ep_finished += NUM_EPS_PER_ITER*cfg.MODEL.TRAIN.NUM_WORKERS
        replay_memory.extend(ep_replay_mem)
        replay_labels.extend(ep_replay_labels)
        # update epsilon value
        if epsilon>0.1 and n_iter%2500==0:
            epsilon /= 1.005

        if len(replay_memory)>=cfg.MODEL.TRAIN.MEM_CAPACITY:    # update model using replay memory
            yield replay_memory, replay_labels
            replay_memory, replay_labels = [], []
        
        ep_scores = np.array(ep_scores)
        ep_max_id = np.argmax(ep_scores)
        if ep_scores[ep_max_id]>max_score:
            max_score = ep_scores[ep_max_id]
            max_tile = ep_maxs[ep_max_id]
        pbar.update(n=NUM_EPS_PER_ITER*cfg.MODEL.TRAIN.NUM_WORKERS)
        pbar.set_postfix({'Best_score': max_score, 'Best_tile': max_tile, 'loss':float(train_loss.result())})
        # if (ep+1)%1000 == 0:
        #     print('Best score: {}, Best episode num: {}'.format(max_score, max_score_ep))

if __name__=='__main__':
    from cfgs.config import cfg_from_yaml_file
    cfg_from_yaml_file('cfgs/SimpleRL.yaml', cfg)
    model = Simple_RLAgent(train=True)
    print('[INFO    ] Model initialized')
    model(tf.ones((1,4,4,16)))
    lock = Lock()
    try:
        latest = tf.train.latest_checkpoint(cfg.MODEL.CHECKPOINT_DIR)
        model.load_weights(latest)
        print('[INFO    ] Using saved checkpoint: ', latest)
    except:
        pass
    print('Number of Trainable variables: ', len(model.trainable_variables))
    ckpt_path = cfg.MODEL.CHECKPOINT_DIR+'cp-{:03d}.ckpt'
    ckpt_dir = path.dirname(ckpt_path)
    loss_obj = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.MODEL.TRAIN.LEARNING_RATE_START)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_mse')
    batch_size = cfg.MODEL.TRAIN.BATCH_SIZE
    epoch = 1
    print('[INFO    ] Started data generation')
    pbar = tqdm.tqdm(total=cfg.MODEL.TRAIN.NUM_EPISODES, desc='train', dynamic_ncols=True)
    for replay_memory, replay_labels in generate_replay_dataset(model):
        replay_memory = np.array(replay_memory, dtype=np.float32)
        replay_labels = np.array(replay_labels, dtype=np.float32)
        train_ds = tf.data.Dataset.from_tensor_slices((replay_memory, replay_labels)).shuffle(len(replay_labels)).batch(cfg.MODEL.TRAIN.BATCH_SIZE)
        
        for grid_ins, labels in train_ds:
            with tf.GradientTape() as tape:
                preds = model(grid_ins)
                loss = loss_obj(labels, preds)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, preds)

        if epoch%25==0: # save model weights
            model.save_weights(ckpt_path.format(epoch))
        epoch += 1
    


