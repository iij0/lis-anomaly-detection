# -*- coding:utf-8 -*-
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import pickle
import os.path
from chainer import cuda
from chainer import variable

from tools.movie_utils import get_movie_filename
from tools.movie_utils import get_movies
from tools.movie_utils import make_movie


class EltFilter(chainer.Link):
    def __init__(self, width, height, channels, batchSize=1, wscale=1, bias=0, nobias=False,
                 initialW=None, initial_bias=None):
        W_shape = (batchSize, channels, height, width)
        super(EltFilter, self).__init__(W=W_shape)

        if initialW is not None:
            self.W.data[...] = initialW
        else:
            std = wscale * xp.sqrt(1. / (width * height * channels))
            self.W.data[...] = np.random.normal(0, std, W_shape)

        if nobias:
            self.b = None
        else:
            self.add_param('b', W_shape)
            if initial_bias is None:
                initial_bias = bias
            self.b.data[...] = initial_bias

    def __call__(self, x):
        y = x * self.W
        if self.b is not None:
            y = y + self.b
        return y


class ConvLSTM(chainer.Chain):
    def __init__(self, width, height, in_channels, out_channels, batchSize=1):
        self.state_size = (batchSize, out_channels, height, width)
        self.in_channels = in_channels
        super(ConvLSTM, self).__init__(
            h_i=L.Convolution2D(out_channels, out_channels, 3, pad=1),
            c_i=EltFilter(width, height, out_channels, nobias=True, batchSize=batchSize),

            h_f=L.Convolution2D(out_channels, out_channels, 3, pad=1),
            c_f=EltFilter(width, height, out_channels, nobias=True, batchSize=batchSize),

            h_c=L.Convolution2D(out_channels, out_channels, 3, pad=1),

            h_o=L.Convolution2D(out_channels, out_channels, 3, pad=1),
            c_o=EltFilter(width, height, out_channels, nobias=True, batchSize=batchSize),
        )

        for nth in range(len(self.in_channels)):
            self.add_link('x_i' + str(nth), L.Convolution2D(self.in_channels[nth], out_channels, 3, pad=1, nobias=True))
            self.add_link('x_f' + str(nth), L.Convolution2D(self.in_channels[nth], out_channels, 3, pad=1, nobias=True))
            self.add_link('x_c' + str(nth), L.Convolution2D(self.in_channels[nth], out_channels, 3, pad=1, nobias=True))
            self.add_link('x_o' + str(nth), L.Convolution2D(self.in_channels[nth], out_channels, 3, pad=1, nobias=True))

        self.reset_state()

    def to_cpu(self):
        super(ConvLSTM, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(ConvLSTM, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def reset_state(self):
        self.c = self.h = None

    def __call__(self, x):
        if self.h is None:
            self.h = variable.Variable(xp.zeros(self.state_size, dtype=x[0].data.dtype), volatile='auto')
        if self.c is None:
            self.c = variable.Variable(xp.zeros(self.state_size, dtype=x[0].data.dtype), volatile='auto')

        ii = self.x_i0(x[0])
        for nth in range(1, len(self.in_channels)):
            ii += getattr(self, 'x_i' + str(nth))(x[nth])
        ii += self.h_i(self.h)

        ii += self.c_i(self.c)
        ii = F.sigmoid(ii)

        ff = self.x_f0(x[0])
        for nth in range(1, len(self.in_channels)):
            ff += getattr(self, 'x_f' + str(nth))(x[nth])
        ff += self.h_f(self.h)
        ff += self.c_f(self.c)
        ff = F.sigmoid(ff)

        cc = self.x_c0(x[0])
        for nth in range(1, len(self.in_channels)):
            cc += getattr(self, 'x_c' + str(nth))(x[nth])
        cc += self.h_c(self.h)
        cc = F.tanh(cc)
        cc *= ii
        cc += (ff * self.c)

        oo = self.x_o0(x[0])
        for nth in range(1, len(self.in_channels)):
            oo += getattr(self, 'x_o' + str(nth))(x[nth])
        oo += self.h_o(self.h)
        oo += self.c_o(self.c)
        oo = F.sigmoid(oo)
        y = oo * F.tanh(cc)

        self.c = cc
        self.h = y
        return y


class PredNet(chainer.Chain):
    def __init__(self, width, height, channels, args, r_channels=None, batchSize=1):
        super(PredNet, self).__init__()
        if r_channels is None:
            r_channels = channels
        assert args.gpu is not None, "args.gpu is needed ( in PredNet ) "

        self.layers = len(channels)
        self.sizes = [None] * self.layers
        w, h = width, height
        for nth in range(self.layers):
            self.sizes[nth] = (batchSize, channels[nth], h, w)
            w = int(w / 2)
            h = int(h / 2)

        for nth in range(self.layers):
            if nth != 0:
                self.add_link('ConvA' + str(nth), L.Convolution2D(channels[nth - 1] * 2, channels[nth], 3, pad=1))
            self.add_link('ConvP' + str(nth), L.Convolution2D(r_channels[nth], channels[nth], 3, pad=1))

            if nth == self.layers - 1:
                self.add_link('ConvLSTM' + str(nth), ConvLSTM(self.sizes[nth][3], self.sizes[nth][2],
                                                              (self.sizes[nth][1] * 2,), r_channels[nth],
                                                              batchSize=batchSize))
            else:
                self.add_link('ConvLSTM' + str(nth), ConvLSTM(self.sizes[nth][3], self.sizes[nth][2],
                                                              (self.sizes[nth][1] * 2, r_channels[nth + 1]),
                                                              r_channels[nth], batchSize=batchSize))
        self.reset_state()

    def to_cpu(self):
        super(PredNet, self).to_cpu()
        for nth in range(self.layers):
            if getattr(self, 'P' + str(nth)) is not None:
                getattr(self, 'P' + str(nth)).to_cpu()

    def to_gpu(self, device=None):
        super(PredNet, self).to_gpu(device)
        for nth in range(self.layers):
            if getattr(self, 'P' + str(nth)) is not None:
                getattr(self, 'P' + str(nth)).to_gpu(device)

    def reset_state(self):
        for nth in range(self.layers):
            setattr(self, 'P' + str(nth), None)
            getattr(self, 'ConvLSTM' + str(nth)).reset_state()

    def __call__(self, x):
        assert xp is not None, " xp is not defined ( in PredNet )"
        for nth in range(self.layers):
            if getattr(self, 'P' + str(nth)) is None:
                setattr(self, 'P' + str(nth), variable.Variable(
                    xp.zeros(self.sizes[nth], dtype=x.data.dtype), volatile='auto'))

        E = [None] * self.layers
        for nth in range(self.layers):
            if nth == 0:
                E[nth] = F.concat(
                    (F.relu(x - getattr(self, 'P' + str(nth))), F.relu(getattr(self, 'P' + str(nth)) - x)))
            else:
                A = F.max_pooling_2d(F.relu(getattr(self, 'ConvA' + str(nth))(E[nth - 1])), 2, stride=2)
                E[nth] = F.concat(
                    (F.relu(A - getattr(self, 'P' + str(nth))), F.relu(getattr(self, 'P' + str(nth)) - A)))

        R = [None] * self.layers
        for nth in reversed(range(self.layers)):
            if nth == self.layers - 1:
                R[nth] = getattr(self, 'ConvLSTM' + str(nth))((E[nth],))
            else:
                upR = F.unpooling_2d(R[nth + 1], 2, stride=2, cover_all=False)
                R[nth] = getattr(self, 'ConvLSTM' + str(nth))((E[nth], upR))

            if nth == 0:
                setattr(self, 'P' + str(nth), F.clipped_relu(getattr(self, 'ConvP' + str(nth))(R[nth]), 1.0))
            else:
                setattr(self, 'P' + str(nth), F.relu(getattr(self, 'ConvP' + str(nth))(R[nth])))

        return self.P0


def prepare_prednet_dataset(_data):
    """

    :param data: numpy array ( np.float32 ) shape ( len , movie_id , 3 , width , height )
    :return: TupleDataset( data , data )  :
    auto encoder shape  ( movie_id , len , 3, width , height ) , (movie_id , len , 3, width , height )
    """

    assert len(_data.shape) == 5, " data axis error (in prepare prednet dataset (len , movie_id , 3 , width , height) )"
    # length , movie_number , width , height = _data.shape[0] , _data.shape[1] , _data.shape[2] , _data.shape[3]
    data = _data.transpose((1, 0, 2, 3, 4)).astype(np.float32)[:-1]
    teacher = _data.transpose((1, 0, 2, 3, 4)).astype(np.float32)[1:]

    return data, teacher



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=10000, type=int, help='epochs (default 50)')
    parser.add_argument('--iter', default=1, type=int, help='split point for train/test')
    parser.add_argument('--channels', '-c', default="3 48 96 192", type=str, help='channels')
    parser.add_argument('--width', '-w', default=160, type=int, help='')
    parser.add_argument('--height', default=128, type=int, help='')
    parser.add_argument('--batchsize', default=1, type=int, help='batchsize of train and test')
    parser.add_argument('--movie_len', default=30, type=int, help='movie len (all)')
    parser.add_argument('--movie_name', default="generated.mp4", type=str, help='movie name')
    parser.add_argument('--split_at', default=40, type=int, help='split point for train/test')
    parser.add_argument('--bprop', default=10, type=int, help='Back propagation length (frames)')
    parser.add_argument('--reportfreq', default=1, type=int, help='loss print frequency ( epoch )')
    parser.add_argument('--loadpickle', default=False, type=bool, help='if True , load pickle ( if exists ) ')

    args = parser.parse_args()
    print(args)
    movie_len = args.movie_len + 1  # want to predict(frame)
    file_names = get_movie_filename("./data")
    # print("used movies : " + " ".join(file_names))
    print("movie loading...")

    # if load pickle mode , load pickle
    if args.loadpickle and os.path.exists("./data/data.pickle"):
        movies = pickle.load(open("./data/data.pickle", "rb"))
    else:
        movies = get_movies(file_names, frame_count=movie_len, size=(args.width, args.height))
        pickle.dump(movies, open("./data/data.pickle", "wb"), -1)
        print("data.pickle is generated")

    # parse channels
    args.channels = [int(channel) for channel in args.channels.split()]

    xp = cuda.cupy if args.gpu >= 0 else np
    model = L.Classifier(
        PredNet(width=args.width, height=args.height, channels=args.channels, batchSize=args.batchsize, args=args),
        lossfun=F.mean_squared_error)
    model.compute_accuracy = False

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
        print('Running on a GPU')
    else:
        print('Running on a CPU')

    # optimizer = chainer.optimizers.Adam()
    optimizer = chainer.optimizers.SMORMS3()
    optimizer.setup(model)

    train_movie, train_teacher = prepare_prednet_dataset(movies)

    for epoch in range(args.epoch):
        acc_loss = 0
        # movie number loop
        for movie, teacher in zip(train_movie,train_teacher):
            # iteration per epoch
            for iterate in range(args.iter):
                model.predictor.reset_state()
                # for sequence
                x = F.expand_dims(xp.asarray(movie[0]), axis=0)
                for i in range(len(movie)):
                    t = F.expand_dims(xp.asarray(teacher[i]), axis=0)
                    loss = model(x, t)
                    x = model.y.data
                    #x = F.expand_dims(xp.asarray(x), axis=0)
                    # learning technique for LSTM/RNN
                    if (i + 1) % args.bprop == 0:
                        model.zerograds()
                        loss.backward()
                        loss.unchain_backward()
                        acc_loss += loss
                        loss = 0
                        optimizer.update()

        # 定期的に結果を観察する
        if epoch % args.reportfreq == 0:
            print("loss epoch %d : " % epoch, acc_loss.data)
            # inference
            model.predictor.reset_state()
            generated_image = np.zeros((args.movie_len, 3, args.height, args.width))
            x = train_movie[0][0]

            # inference
            if args.gpu >= 0: model.to_cpu()
            for frame in range(args.movie_len):
                x = F.expand_dims(x, axis=0)
                _ = model(x, x)
                x = model.y.data[0]
                generated_image[frame] = model.y.data[0].copy()
            if args.gpu >= 0: model.to_gpu()

            make_movie(generated_image, file_name=str(epoch) + "_" + args.movie_name, fps=24)
