# coding:utf-8
import argparse
import gym
import numpy as np
import pickle
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda
from models.PredNet import PredNet

from chainer import serializers
import time

from movie_utils import make_movie
import time



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--log-file', '-l', default='reward.log', type=str, help='reward log file name')
parser.add_argument('--epoch', '-e', default='1', type=int, help='learning epoch')
args = parser.parse_args()

class PredNet3Layer(chainer.Chain):

    def __init__(self , width , height , channels, batchSize):
        super(PredNet3Layer , self).__init__()
        self.add_link("l1",PredNet(width=width,height=height,channels=channels,batchSize=batchSize))
        self.add_link("l2",PredNet(width=width,height=height,channels=channels,batchSize=batchSize))
        self.add_link("l3",PredNet(width=width,height=height,channels=channels,batchSize=batchSize))


    def __call__(self, x):

        h = self.l1(x)
        h = self.l2(x)
        h = self.l3(x)

        return h

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()



def learn(movie_type = "nothing"):
    """
    movie.pickleを用いて，PredNetの学習を行う
    Returns:

    """

    data = pickle.load(open("./movies/" + movie_type  + "_movie.pkl","r"))
    print "movie " , movie_type , " loaded"
    xp = cuda.cupy if args.gpu >= 0 else np
    model = L.Classifier( PredNet3Layer(width=160, height=128, channels=[3,48,96,192], batchSize=1 ), lossfun=F.mean_squared_error)
    model.compute_accuracy = False


    # load model
    #model = pickle.load(open("./model.pkl"))

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        xp = cuda.cupy
        model.to_gpu()
        print('Running on a GPU')
    else:
        print('Running on a CPU')

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_movie = data[:-1]
    train_teacher = data[1:]

    print "data size(s)"
    print train_movie.shape , train_teacher.shape

    acc_loss = 0
    for epoch in range(args.epoch):
        print "epoch (" , epoch , ") start "
        # movie number loop
        model.predictor.reset_state()
        x = F.expand_dims(chainer.Variable(xp.array(train_movie[0])) ,axis = 0)


        # for sequence
        loss = 0
        for i in range(len(train_movie)):
            t = F.expand_dims(chainer.Variable(xp.array(train_teacher[i])) , axis = 0 )
            loss += model(x, t)
            x = model.y.data

        #learning technique for LSTM/RNN
        model.zerograds()
        loss.backward()
        loss.unchain_backward()
        print("model updated")
        print(" loss : " ,  loss.data)
        acc_loss += loss
        loss = 0
        optimizer.update()

    serializers.save_npz("3Layer_" + str(epoch) + "_" + "model.npz" , model)
    if args.gpu >= 0: model.to_cpu()
    print "model saving"
    pickle.dump(model,open("3Layer_" + str(epoch) + "_" + "model.pkl" , "wb") , -1)



def observe(file_name = "movie"):
    """
    背景学習のための，観測スクリプト
    前進し続け，動画をpickle形式で保存する

    Returns: (movie.pkl)
    """

    # 前進のみ行う
    action = 0
    env = gym.make('Lis-v2')
    observation = env.reset()
    observation, reward, end_episode, _ = env.step(action)

    seq_len = 30 + 1
    data = np.zeros( (seq_len , 3 ,128 , 160) , dtype=np.float32)

    for seq in range(seq_len):
        # 現在の環境では，5ステップ前後で1周
        observation, reward, end_episode, _ = env.step(action)
        image = np.array(observation["image"][0].resize((160,128))).transpose(2, 0, 1)[::-1].astype(np.float32) / 255
        data[seq] = image


    print "data generated " , data.shape
    pickle.dump(data,open(file_name + ".pkl","wb"),-1)

    env.close()

def make_error_movie(model="./model.pkl", movie_len = 100):
    """
    既存のモデルを読み込み，ある環境でのエラーを動画化する

    Args:
        model: pickle file

    Returns:

    """
    # 前進のみ行う
    action = 0
    env = gym.make('Lis-v2')
    observation = env.reset()
    observation, reward, end_episode, _ = env.step(action)

    seq_len = 30 + 1
    data = np.zeros( (seq_len , 3 ,128 , 160) , dtype=np.float32)

    for seq in range(seq_len):
        # 現在の環境では，5ステップ前後で1周
        observation, reward, end_episode, _ = env.step(action)
        image = np.array(observation["image"][0].resize((160,128))).transpose(2, 0, 1)[::-1].astype(np.float32) / 255
        data[seq] = image


    print "data generated " , data.shape
    pickle.dump(data,open("movie.pkl","wb"),-1)

    env.close()




if __name__ == "__main__":

    start = time.time()
    learn()
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

