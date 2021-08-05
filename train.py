from data_utils import train_generator
from model import General_GAN
#
# def get_args():
#

if __name__ == '__main__':
    train_generator = train_generator('./data')
    anime_gan = General_GAN()
    anime_gan.fit(log_file='train_20210805.log', train_generator=train_generator)
