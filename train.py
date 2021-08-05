from data_utils import train_generator
from model import General_GAN
from data_utils import dataset_resize
from data_utils import calculate_steps
# def get_args():
#

if __name__ == '__main__':
    batch_size = 2
    # dataset_resize('./data')
    steps = calculate_steps('./data', batch_size=batch_size)
    train_generator = train_generator('./data', batch_size=batch_size)
    anime_gan = General_GAN(batch_size=batch_size)
    # def train():
    #     dist.init_parallel_env()
    anime_gan.fit(log_file='train_20210805.log', train_generator=train_generator, steps_tqdm=steps, input_constant=False)

    # dist.spawn(train, nprocs=2)

