# coding:utf-8
import argparse
import gym
import numpy as np
import pickle
import chainer
from chainer import functions as F
from Lis_pred1 import PredNet1Layer
from Lis_pred2 import PredNet2Layer
from Lis_pred3 import PredNet3Layer

from chainer import links as L
from chainer import serializers
from chainer import cuda
from models.PredNet import PredNet

import time

from movie_utils import make_movie
import time

def make_error_movie(model_file="1Layer_999", movie_len = 60,input_movie="normal",out_name="output_movie.mp4"):
    """
    既存のモデルを読み込み，ある環境でのエラーを動画化する

    Args:
        model: pickle file

    Returns:

    """
    size = (160,128)

    model = L.Classifier( PredNet1Layer(width=160, height=128, channels=[3,48,96,192], batchSize=1 ), lossfun=F.mean_squared_error)
    #model = L.Classifier( PredNet2Layer(width=160, height=128, channels=[3,48,96,192], batchSize=1 ), lossfun=F.mean_squared_error)
    #model = L.Classifier( PredNet3Layer(width=160, height=128, channels=[3,48,96,192], batchSize=1 ), lossfun=F.mean_squared_error)
    model.compute_accuracy = False
    serializers.load_npz("./models/" +  model_file +"_model.npz",model)

    data = pickle.load(open("./movies/" +input_movie+"_movie.pkl","r"))

    movie = np.zeros((movie_len,3 ,size[1],size[0]),dtype=np.float32)

    for i, seq in enumerate(data):
        print i
        # 現在の環境では，5ステップ前後で1周
        seq = F.expand_dims(seq , axis=0)
        _ = model(seq, seq)
        image = model.y.data
        movie[i] = image

    make_movie(movie,file_name=out_name,fps=30)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model',  type=str ,default="2Layer_nothing_9")
    parser.add_argument('--len',  type=int , default=60)
    parser.add_argument('--input_movie',  type=str , default="normal")
    parser.add_argument('--out_name',  type=str , default="output_movie.mp4")
    args = parser.parse_args()

    make_error_movie(args.model ,args.len , args.input_movie,args.out_name)

