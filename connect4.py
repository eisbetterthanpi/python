import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# python "F:\connect4.py"

class connect4():
    def __init__(self, rows=6, cols=7,l=4):
        self.rows=rows
        self.cols=cols
        self.l=l
        # self.board=[[0]*self.cols for x in range(self.rows)]
        # self.board=[
        # [0, -1, 1, 1, 1, -1, 0],
        # [0, 0, 1, 0, 1, 0, 0],
        # [0, 1, 1, 1, 0, 0, 0],
        # [1, 0, -1, 0, 1, 0, 0],
        # [0, 0, 0, 0, 0, -1, 0],
        # [0, 0, 0, 0, 0, 0, 0],
        # ]
        # self.board=np.array(self.board)
        self.board=np.array([[0]*self.cols for x in range(self.rows)])

    def getboard(self,board=None): #flat array
        if board is None:
            board=self.board
        # t=self.board[0] print([t.extend(x) for x in self.board[1:]])
        # gah=[]
        # # [lambda x:gah+=x for x in board]
        # # [gah.append(x) for x in self.board]
        # for x in self.board:
        #     gah+=x
        # return np.reshape(board,(1,self.cols*self.rows))
        return board.flatten() #.ravel()

    def win(self,board=None):
        if board is None:
            board=self.board
        # print("dfb",board,board[0])
        c=[1*self.l,-1*self.l]
        r=["_","|","/","\\"]
        # print(board)
        for y in range(self.rows):
            for x in range(self.cols):
                # print(x,self.l)
                # print('sfg',board[y][x:x+self.l])
                # print('sfg',board[y])
                # print("yx",y,x)
                try:
                    # print('q1')
                    if sum(board[y][x:x+self.l]) in c: # horizontal
                        # print("_",y,x)
                        return "_"
                except IndexError:
                    # continue
                    pass
                try:
                    if sum([j[x] for j in board[y:y+self.l]]) in c: # vertical
                        return "|"
                except IndexError:
                    pass
                try:
                    if sum([board[y+i][x+i] for i in range(self.l)]) in c: # forward slash
                        return "/"
                except IndexError:
                    pass
                try:
                    if sum([board[y+i][x-i] for i in range(self.l)]) in c: # back slash
                        return "\\"
                except IndexError:
                    pass

                # try:
                #     test=[board[y][x:x+self.l], #_
                #     [j[x] for j in board[y:y+self.l]], #|
                #     [board[y+i][x+i] for i in range(self.l)], #/
                #     [board[y+i][x-i] for i in range(self.l)]] #\
                #     print(test)
                #     for n,t in enumerate(test):
                #         if sum(t) in c:
                #             print(r[n])
                #             return r[n]
                # except IndexError:
                #     # print('cont')
                #     # continue
                #     pass

                # for n,t in enumerate(test):
                # for n in range(4):
                #     t=[board[y][x:x+self.l], #_
                #     [j[x] for j in board[y:y+self.l]], #|
                #     [board[y+i][x+i] for i in range(self.l)], #/
                #     [board[y+i][x-i] for i in range(self.l)]][n]
                #     try:
                #         # test=[board[y][x:x+self.l], #_
                #         # [j[x] for j in board[y:y+self.l]], #|
                #         # [board[y+i][x+i] for i in range(self.l)], #/
                #         # [board[y+i][x-i] for i in range(self.l)]] #\
                #         # print(test)
                #         # print(y,x,test[n],sum(test[n]) , c,sum(test[n]) in c)
                #         # if sum(test[n]) in c:
                #         print(t)
                #         if sum(t) in c:
                #             print(r[n])
                #             return r[n]
                #     except IndexError:
                #         # print('cont')
                #         # continue
                #         pass
        return

    def getValidActions(self,s=None):
        if s is None:
            s=self.board
        x = np.array(s[-1])
        # print("act",x,np.where(x == 0)[0])
        # [1 if x==0 else 0 for x in board[-1]]
        return np.where(x == 0)[0]

    def slot(self,play,board=None,p=None):
        if board is None:
            board=self.board
        # sum(self.getboard(board))
        # print('rdhter',board,sum(self.getboard(board)))
        if p is None:
            p=[1,-1][sum(self.getboard(board))]
        play=int(play)
        if play not in self.getValidActions(board):
            return "err"

        top=[x[play] for x in board]
        # print(self.board[:,2],"op") # numpy
        # print(list(zip(*self.board))) # ok
        if not 0 in top:
            return "err2"
        elif board is None:
            self.board[top.index(0)][play]=p
        # else:
        board[top.index(0)][play]=p
        return board

    def show(self,board=None):
        if board is None:
            board=self.board
        # [print(x) for x in reversed(self.board)]
        print(np.flip(board,0))
        return

# s=game.board # board state s
# v0s p0s
# [v0(st), p0(st)]=f(s) #nn f  # v0s board state score bet -1 and 1    p0(s) policy [ , , , , , ] probability vector over all possible actions
# pit = estimate of policy from state s
# v probability of the current player winning in position
# # zt 1 or -1 final outcome win/lose
# => loss function

# action a state s
# Q(s,a) # (action value) expected reward
# N(s,a) # (visit count) num of times this action taken accross simulations
# P0(s) # initial estimate of policy from nn
# => U(s,a) # upper confidence bound on the Q-values
# cpuct # degree of exploration

import logging
import math
EPS = 1e-8
log = logging.getLogger(__name__)

# mcts.py
def search(state, game, nnet):
    # if game.end(s): return -game.reward(s)
    reward=1
    print("win state",state,type(state))
    if game.win(state): return -reward
    # Q=nnet.predict(s)
    # Q=nnet.predict([s])
    Q={}
    N={}
    # state_id = '/'.join([str(x) for x in s])
    # state_id = ''.join([str(x) for x in game.getboard(s)])
    state_id = ''.join([str(x) for x in state])
    # print('scheck1',s)
    # if state_id in Q:
    #     Q[state_id][action] = nnet.predict([s])[-1]
    # else:
    #     Q[state_id] = {action:nnet.predict([s])[-1]}

    # print("qqqqqqq",Q,len(Q),len(Q[0]))
    visited=np.array([[0]*42])
    # visited=np.array([[0]*self.cols for x in range(self.rows)])
    s=game.getboard(state)
    if s not in visited:
        # visited.add(s)
        # s=np.array(s)
        # print("sssss",s,len(s))
        visited=np.append(visited,s,axis=0)
        # P[s], v = nnet.predict(s)
        # prediction = nnet.predict(s)
        # prediction = nnet.predict([s])
        prediction = nnet.predict(state)
        # print('if',prediction)
        print("notin",s.shape,prediction[0].shape)
        # P[s], v = prediction[0] ,prediction[-1]
        pos, v = prediction[0] ,prediction[-1]

        # P = nnet.predict(s)
            # history = model.fit([rownum, colnum], output, batch_size=32, epochs=1)
            # results = model.evaluate([rownum, colnum], output, batch_size=128)
            # predictions = model.predict([rownum, colnum])
        return -v
    max_u, best_a = -float("inf"), -1
    c_puct=0.5 #degree of exploration
    # P[s], v = nnet.predict(s) # initial estimate of policy from nn for state s
    # print("pres",s)
    # print('prepre',s,s.shape)
    print('press',state,state.shape)
    # prediction = nnet.predict(s)
    # prediction = nnet.predict([s])
    prediction = nnet.predict(np.array([state]))
    # prediction = nnet.predict([state])
    # prediction = nnet.predict([[state]])
    # prediction = nnet.predict([state,state])
    # print('out',prediction)
    # print('out1',prediction[0])
    # print('out2',prediction[-1])
    # P[s], v = prediction[:-1] ,prediction[-1]
    # predictions = model.predict(x_test[:3])
    # tp=tf.argmax(predictions, 1)

    for a in game.getValidActions(state):
        print('ba',a,s,state)
        print('poos',game.slot(a,state))
        # print('poos',nnet.predict([game.slot(a,state)]))
        # sd=game.getboard(game.slot(a,state))
        sd=np.array([game.slot(a,state)])
        # sd=np.array(game.slot(a,state))
        print("sded",sd,sd.shape)
        print('poost',nnet.predict(sd))
        # pos = nnet.predict([game.slot(a,state)])#[-1][0][0]
        # pos = nnet.predict(np.array([sd]))[-1][0][0]
        pos = nnet.predict(sd)[-1][0][0]
        print('pos',pos)
        if state_id in Q:
            # Q[state_id][a] = nnet.predict([s])[-1][0]
            # Q[state_id][a] = nnet.predict(state)[-1][0]
            Q[state_id][a] = nnet.predict(sd)[-1][0][0]
        else:
            # Q[state_id] = {a:nnet.predict([s])[-1][0]}
            # Q[state_id] = {a:nnet.predict(state)[-1][0]}
            Q[state_id] = {a:nnet.predict(sd)[-1][0][0]}

        if state_id in N:
            if a in N[state_id]:
                # N[state_id][a] =N[state_id][a]
                N[state_id][a] +=1
            else: N[state_id] = {a:1}
        else: N[state_id] = {a:1}
        # print('qqqqq',Q,state_id,a)
        # u = Q[s][a] + c_puct*P[s][a]*sqrt(sum(N[s]))/(1+N[s][a])
        # u = Q[s][a] + c_puct*P[s][a]*(sum(N[s]))/(1+N[s][a])**(1/2)
        # u = Q[s][a] + c_puct*pos[a]*(sum(N[s]))/(1+N[s][a])**(1/2)
        # u = Q[state_id][a] + c_puct*pos[a]*(sum(N[s]))/(1+N[s][a])**(1/2)
        u = Q[state_id][a] + c_puct*pos*(sum(N[state_id]))/(1+N[state_id][a])**(1/2)
        # u = Q[state_id][a] + c_puct*pos
        # v=(sum(N[state_id]))
        # w=(1+N[state_id][a])**(1/2)

        if u>max_u:
            max_u = u
            best_a = a
    a = best_a
    # sp = game.nextState(s, a)
    sp = game.slot(a,state)
    print("sps",a,s,sp)
    v = search(sp, game, nnet)
    # Q[s][a] = (N[s][a]*Q[s][a] + v)/(N[s][a]+1)
    # N[s][a] += 1
    Q[state_id][a] = (N[state_id][a]*Q[state_id][a] + v)/(N[state_id][a]+1)
    N[state_id][a] += 1
    return -v

# politer self play
# https://gist.github.com/suragnair/fa6e1935d3b6cf650ac039bf04bc9b13#file-politer-selfplay-py
def policyIterSP(game):
    nnet = initNNet()                                       # initialise random neural network
    examples = []
    for i in range(numIters):
        for e in range(numEps):
            examples += executeEpisode(game, nnet)          # collect examples from this game
        new_nnet = trainNNet(examples)
        frac_win = pit(new_nnet, nnet)                      # compare new net with previous net
        if frac_win > threshold:
            nnet = new_nnet                                 # replace with new net
    return nnet

def executeEpisode(game, nnet):
    import random
    examples = []
    # s = game.startState()
    state=game.board
    s = game.getboard()
    s=np.array(s)
    print("sboard",s,len(s),s.shape)
    # mcts = MCTS()                                           # initialise search tree
    numMCTSSims=10
    while True:
        for _ in range(numMCTSSims):
            # mcts.search(s, game, nnet)
            # search(s, game, nnet)
            search(state, game, nnet)
        # prediction = nnet.predict(s) # shape 1?
        # prediction = nnet.predict(state)
        prediction = nnet.predict(np.array([state]))
        # prediction = nnet.predict([s]) # with input shape 42?
        # pis, v = prediction[0] ,prediction[-1]
        pis, v = prediction[0][0] ,prediction[-1][0]
        # examples.append([s, mcts.pi(s), None])              # rewards can not be determined yet
        # a = random.choice(len(mcts.pi(s)), p=mcts.pi(s))    # sample action from improved policy
        examples.append([s, pis, None])
        # a = random.choice(len(pis), p=pis)
        a = random.choice(pis)
        print("rana",a,pis)
        # s = game.nextState(state,a)
        s = game.slot(a,state)
        # if game.gameEnded(state):
        if game.win(state):
            # examples = assignRewards(examples, game.gameReward(s))
            examples = assignRewards(examples, 1)
            # game.win(state): return 1
            return examples


# # https://rubikscode.net/2019/07/08/deep-q-learning-with-python-and-tensorflow-2-0/
# class Agent:
#     def __init__(self, enviroment, optimizer):
#         self._state_size = enviroment.observation_space.n
#         self._action_size = enviroment.action_space.n
#         self._optimizer = optimizer
#         self.expirience_replay = deque(maxlen=2000)
#         self.gamma = 0.6 #discount
#         self.epsilon = 0.1 #exploration rate
#
#         self.q_network = self._build_compile_model()
#         self.target_network = self._build_compile_model()
#         self.alighn_target_model()
#
#     def store(self, state, action, reward, next_state, terminated):
#         self.expirience_replay.append((state, action, reward, next_state, terminated))
#
#     def _build_compile_model(self):
#         model = Sequential()
#         model.add(Embedding(self._state_size, 10, input_length=1))
#         model.add(Reshape((10,)))
#         model.add(Dense(50, activation='relu'))
#         model.add(Dense(50, activation='relu'))
#         model.add(Dense(self._action_size, activation='linear'))
#
#         model.compile(loss='mse', optimizer=self._optimizer)
#         return model
#
#     def alighn_target_model(self):
#         self.target_network.set_weights(self.q_network.get_weights())
#
#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return enviroment.action_space.sample()
#
#         q_values = self.q_network.predict(state)
#         return np.argmax(q_values[0])
#
#     def retrain(self, batch_size):
#         minibatch = random.sample(self.expirience_replay, batch_size)
#
#         for state, action, reward, next_state, terminated in minibatch:
#
#             target = self.q_network.predict(state)
#
#             if terminated:
#                 target[0][action] = reward
#             else:
#                 t = self.target_network.predict(next_state)
#                 target[0][action] = reward + self.gamma * np.amax(t)
#
#             self.q_network.fit(state, target, epochs=1, verbose=0)


# data, model, compile, fit, evaluate
def initNNet():
    rows,cols=6,7
    inputs = tf.keras.Input(shape=(6,7)) #figure out
    # inputs = tf.keras.Input(shape=(42,)) #[] line works python array? or reshape to wierd
    # inputs = tf.keras.Input(shape=(1,))
    # inputs = tf.keras.Input(shape=(7,))

    x = layers.LSTM(124, activation = 'sigmoid', name='layer1', dropout = 0.4)(inputs)
    # x = layers.Conv2D(4, 4)(inputs)
    # x = layers.Conv2D(1,(1,64))(inputs)
    # x = layers.Conv2D(1,(64,))(inputs) #nope, kernel size should be tup size 2
    # x = layers.Conv2D(None,(1,64))(inputs)
    # x = layers.Conv2D(1,64)(inputs)

    # x = layers.Dense((None,67), activation="relu", name="dense_1")(inputs)
    # x = layers.Dense((None,67), activation="relu", name="dense_1")(x)
    # x = layers.Dense(67, activation="relu", name="dense_1")(inputs)
    # x = layers.Dense((8,9), activation="relu", name="dense_1")(x)
    # x = layers.Dense((6,7), activation="relu", name="dense_2")(x)
    x = layers.Dense(67, activation="relu", name="dense_2")(x)
    # outputs = layers.Dense((0,7), name="predictions")(x)
    # outputs = layers.Dense(7, activation="softmax", name="predictions")(x)
    # outputs = layers.Dense(8, activation="softmax", name="predictions")(x)
    # model = keras.Model(inputs=inputs, outputs=outputs)

    # yrow = keras.Input(shape=(row,), name="inputs")
    # xcol = keras.Input(shape=(1col,), name="targets")
    p = layers.Dense(7, activation="softmax", name="predict")(x)
    v = layers.Dense(1, activation="softmax", name="value")(x)
    # model = keras.Model(inputs=[yrow,xcol], outputs=[p,v])
    model = keras.Model(inputs=inputs, outputs=[p,v])

    # P[s], v = model.predict()
    # model = keras.Model(inputs=[inputs, targets], outputs=predictions)
    # model = keras.Model(inputs=[image_input, timeseries_input], outputs=[score_output, class_output])
    model.summary()
    return model


# inputs = keras.Input(shape=(3,), name="inputs")
# targets = keras.Input(shape=(10,), name="targets")
# logits = keras.layers.Dense(10)(inputs)
# predictions = LogisticEndpoint(name="predictions")(logits, targets)
#
#
# image_input = keras.Input(shape=(32, 32, 3), name="img_input")
# x1 = layers.Conv2D(3, 3)(image_input)
# x1 = layers.GlobalMaxPooling2D()(x1)
#
# timeseries_input = keras.Input(shape=(None, 10), name="ts_input")
# x2 = layers.Conv1D(3, 3)(timeseries_input)
# x2 = layers.GlobalMaxPooling1D()(x2)
#
# x = layers.concatenate([x1, x2])
# score_output = layers.Dense(1, name="score_output")(x)
# class_output = layers.Dense(5, name="class_output")(x)

#
# def othellomodel():
#     # input = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y
#     # inputs = tf.keras.Input(shape=(rows,cols), name="digits")
#     inputs = tf.keras.Input(rows,cols)
#
#     x_image = tf.reshape(inputs,(rows,cols, 1))                # batch_size  x board_x x board_y x 1
#     # h_conv1 = tf.keras.layers.Activation('relu')(tf.keras.layers.BatchNormalization(axis=3)(tf.keras.layers.Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
#     h_conv1 = tf.keras.layers.Activation('relu')(tf.keras.layers.BatchNormalization(axis=3)(tf.keras.layers.Conv2D(512, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
#     h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
#     h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
#     h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
#     h_conv4_flat = Flatten()(h_conv4)
#     s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
#     s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
#     pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
#     v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1
#
#     model = keras.Model(inputs=self.input_boards, outputs=[pi, v])
#     # model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
#     model.summary()
#     return model



def NNett(): #dummy
    model = initNNet()
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3), # Optimizer
    # optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # metrics=[CategoricalTruePositives()],
    metrics=[keras.metrics.SparseCategoricalAccuracy()], # List of metrics to monitor
    )

    # # different losses to different outputs
    # model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
    #     # loss=[None, keras.losses.CategoricalCrossentropy()],
    #     loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()],
    #     # metrics=[[keras.metrics.MeanAbsolutePercentageError(),keras.metrics.MeanAbsoluteError(),],
    #         # [keras.metrics.CategoricalAccuracy()],],)
    #     # loss_weights
    #     )

    # import os #sys
    # dirname = os.path.dirname() #(__file__)
    # filename = os.path.join(dirname, 'relative/path/to/file/you/want')
    # "F:/callbacksave/"
    # callbacks = [keras.callbacks.ModelCheckpoint(
    #         filepath="F:/callbacksave" + "/ckpt-loss={loss:.2f}",
    #         # filepath="mymodel_{epoch}", # Path where to save the model
    #         # save_freq=100, # saves a SavedModeland training loss every 100 batches
    #         save_best_only=True,  # Only save a model iff `val_loss` has improved.
    #         monitor="val_loss", # The saved model name will include the current epoch.
    #         verbose=1,)]

    rownum, colnum=16,17
    # history = model.fit([rownum, colnum], played, batch_size=32, epochs=1)
    # results = model.evaluate([rownum, colnum], played, batch_size=128)
    history = model.fit([rownum, colnum], output, batch_size=32, epochs=1)
    results = model.evaluate([rownum, colnum], output, batch_size=128)
    predictions = model.predict([rownum, colnum])
    return [results, predictions]


# getboard(self,board=None):
# win(self,board=None):
# getValidActions(self,s=None):
# slot(self,play,board=None,p=None):
# show(self,board=None):



game=connect4()
nnet=initNNet()
executeEpisode(game, nnet)

def mest():
    game=connect4()
    nnet=initNNet()
    # executeEpisode(game, nnet)

    state=[[0, -1, -1, -1, 1, -1, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0],
    [-1, 0, -1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0]] #(6,7)    (None, 7)
    # board=[0, 1, 1, 1, 1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, -1, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] #(42,)
    # board=[board]
    state=np.array(state)
    state=np.array([state])
    print(state)
    # print(sum(sum(state)))
    # s=game.getboard(state)
    # s=[np.zeros(42,)]
    # s=np.reshape(state,(1,42))
    # s=[[0]*42]
    # print(board.shape)
    # state=np.array([state,state])
    # print(sb)
    # print(s,s.shape)
    # prediction = nnet.predict([state])
    prediction = nnet.predict(state)
    print(prediction)
    # visited=np.array([])
    # visited=np.array([[0]*42])
    # print(visited)
    # visited=np.append(visited,board,axis=0)
    # print(visited)
    # visited=np.append(visited,board,axis=0)
    # print(visited)

    # print('win',game.win(board),game.win())
    # game.slot(1,state,p=None)
    # sb=game.getboard(state)
    # print(sb)
    game.slot(1,sb)
# mest()

def test():
    game=connect4(6,7,4)

    # b='''[[ 0  0  1 -1 -1  1  0]
    #  [ 0  0 -1  1  1  0  0]
    #  [ 0  0 -1  1  0  0  0]
    #  [ 0  0  1 -1  0  0  0]
    #  [ 0  0  0  0  0  0  0]
    #  [ 0  0  0  0  0  0  0]]'''
    # b=b.replace(' ',',')
    # b=b.replace(',,',',')
    # b=b.split()
    # ''.join(b)

    s= [[1, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 1, -1],
    [0, 0, 1, 0, 1, 0, -1],
    [0, 0, 0, 1, 0, 0, -1],
    [0, 0, 0, 0, 0, 1, -1],
    [0, 0, 0, -1, -1, -1, -1]]
    print(sum(sum(s)))
    s=np.array(s)
    # game.show(s)
    # print(game.win(s))
    # print('gb')
    # # print(game.win())
    # # slot(self,play,board=None,p=None)
    # # print(game.getboard(s))
    # game.show()
    # game.show(s)

    print("tttttttttttttt")
    # sb=game.getboard(s)
    # print(s,sb)
    game.slot(1,s,p=None)
    # game.show()
    # game.show(s)
    # game.slot(4)
    # game.show()
    game.show(s)
# test()

#
def twoplayer():
    game=connect4(6,7,4)
    pl=1
    while True:
        print("44444444444")
        game.show()
        tap= input(pl)
        print("her",tap,pl)
        game.slot(tap,pl)
        print(game.win())
        if game.win():
            print("win",pl)
            break
        pl*=-1
# twoplayer()

# def match(p1,p2):
#     game=connect4(6,7,4)
#     pl=1
#     while True:
#         # print("44444444444")
#         # game.show()
#         tap= 1
#         if tap not in self.board[-1]:
#             tap = 1
#         game.slot(tap,pl)
#         if game.win():
#             print("win",pl)
#             return pl
#
#         pl*=-1

# python "F:\connect4.py"
