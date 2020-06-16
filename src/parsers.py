import argparse


def get_parser(description="ABPSC MNIST"):
    parser = argparse.ArgumentParser(description=description)

    # Model Arguments
    parser.add_argument('-hidden_size', type=int, default=400, help='linear hidden layer dimension, default=400')
    parser.add_argument('-latent_size', type=int, default=8, help='number of latent dimension, default=8')
    parser.add_argument('-kernel_size', type=int, default=64, help='Kernel size of transposed convolution layer, default=64')
    parser.add_argument('-channel_size', type=int, default=1, help='Dimension of images, default=1')
    parser.add_argument('-dataset', type=str, default='mnist', help='currently only support mnist, default=mnist')
    parser.add_argument('-image_size', type=int, default=28, help='Image height and length, default=28')

    # Training Arguments
    parser.add_argument('-device', type=str, default='cuda:0', help='Whether use gpu for training, default=cuda:0')
    parser.add_argument('-batch_size', type=int, default=32, help='training batch size, default=32')
    parser.add_argument('-learning_rate', type=float, default=0.001, help='learning rate for generator, default=0.001')
    parser.add_argument('-num_epochs', type=int, default=20, help='number of training epochs, default=20')
    parser.add_argument('-seed', type=int, default=28, help='random seed, default 28')
    parser.add_argument('-noise_variance', type=float, default=0.3, help='Gaussian Noise Variance, default=0.3')
    parser.add_argument('-langevin_stepsize', type=float, default=0.05, help='Step size of langevin sampling. default=0.05')
    parser.add_argument('-langevin_steps', type=int, default=10, help='Number of langevin sampling steps, default=10')
    parser.add_argument('-slab_variance', type=float, default=0.1, help='Variance of gaussian mixture model, default=0.1')
    parser.add_argument('-alpha', type=float, default=0.01, help='Spike variable, default=0.01')

    return parser
