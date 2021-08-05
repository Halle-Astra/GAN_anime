from data_utils import train_generator
from model import General_GAN
from data_utils import dataset_resize
from data_utils import calculate_steps
# def get_args():
#

if __name__ == '__main__':
    dataset_resize('./data')
    steps = calculate_steps('./data')
    train_generator = train_generator('./data')
    anime_gan = General_GAN()
    anime_gan.fit(log_file='train_20210805.log', train_generator=train_generator, steps_tqdm=steps)
