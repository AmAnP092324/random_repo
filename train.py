import argparse
import numpy as np
import main as m

dict_ = {'airplane':1, 'automobile':2, 'bird':3, 'cat':4, 'deer':5,
         'dog':6, 'frog':7, 'horse':8, 'ship':9, 'truck':10}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.5, help="Momentum")
    parser.add_argument("--num_hidden", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--sizes", type=str, default="100,100,100", help="Sizes of hidden layers")
    parser.add_argument("--activation", type=str, default="sigmoid", help="Activation function")
    parser.add_argument("--loss", type=str, default="sq", help="Loss function")
    parser.add_argument("--opt", type=str, default="adam", help="Optimizer")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--anneal", type=bool, default=True, help="Whether to anneal")
    parser.add_argument("--save_dir", type=str, default="pa1/", help="Save directory")
    parser.add_argument("--expt_dir", type=str, default="pa1/exp1/", help="Experiment directory")
    parser.add_argument("--train", type=str, default="", help="Training data file")
    parser.add_argument("--test", type=str, default="test.csv", help="Testing data file")

    args = parser.parse_args()

    sizes = list(map(int, args.sizes.split(",")))
    
    train_set_x = np.load(f"{args.train}/train.npy", allow_pickle=True)
    train_set_y_orig = np.load(f"{args.train}/train_labels.npy", allow_pickle=True)
    train_set_y_orig = np.reshape(train_set_y_orig, (50000, 1))
    train_y = []
    for e in train_set_y_orig: 
        pos = dict_[e[0]]
        list_ = [0 for i in range(10)]
        list_[pos-1] = 1
        train_y.append(list_) 
    train_set_y = np.array(train_y).reshape((50000, 10))

    test_set_x = np.load(f"{args.test}/test.npy", allow_pickle=True)

    Net = m.NeuralNetwork(momentum=args.momentum, num_hidden=args.num_hidden, sizes=sizes, activation=args.activation, lr=args.lr, loss=args.loss, opt=args.opt)
    Net.train(train_set_x, train_set_y, epochs=2, batch_size=args.batch_size)
    Net.validate(train_set_x, train_set_y)
