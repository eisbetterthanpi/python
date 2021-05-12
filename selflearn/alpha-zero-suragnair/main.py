import logging
from Coach import Coach
from connect4.Connect4Game import Connect4Game as Game
from connect4.tensorflow.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

# https://colab.research.google.com/drive/1tehvpwPlx0w8ysI_sbvMMtsrxb_9ZIu1

args = dotdict({
    'numIters': 10, #1000
    'numEps': 10, #100             # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15, #15       #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000, #200000   # Number of game examples to train the neural networks.
    'numMCTSSims': 25, #25         # Number of games moves for MCTS to simulate.
    'arenaCompare': 1, #40        # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': 'F:/selflearn/alpha-zero-suragnair/temp/', #windows
    # 'checkpoint': './drive/MyDrive/selflearn/temp/', #'./temp/', #colab, og
    'load_model': True, #False
    'load_folder_file': ('F:/selflearn/alpha-zero-suragnair/temp/','best.pth.tar'), #windows
    # 'load_folder_file': ('./drive/MyDrive/selflearn/temp/','best.pth.tar'), #colab # 'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'), #og
    'numItersForTrainExamplesHistory': 20,
})
# args = {
#     'numIters': 1000, #1000
#     'numEps': 100, #100             # Number of complete self-play games to simulate during a new iteration.
#     'tempThreshold': 15, #15       #
#     'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
#     'maxlenOfQueue': 200000, #200000   # Number of game examples to train the neural networks.
#     'numMCTSSims': 25, #25         # Number of games moves for MCTS to simulate.
#     'arenaCompare': 10, #40        # Number of games to play during arena play to determine if new net will be accepted.
#     'cpuct': 1,
#
#     'checkpoint': 'F:/selflearn/alpha-zero-suragnair/temp/', #windows
#     # 'checkpoint': './drive/MyDrive/selflearn/temp/', #'./temp/', #colab, og
#     'load_model': True, #False
#     'load_folder_file': ('F:/selflearn/alpha-zero-suragnair/temp/','best.pth.tar'), #windows
#     # 'load_folder_file': ('./drive/MyDrive/selflearn/temp/','best.pth.tar'), #colab # 'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'), #og
#     'numItersForTrainExamplesHistory': 20,
# }

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(6)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
