from data_utils import train_generator
from model import General_GAN
from data_utils import dataset_resize
from data_utils import calculate_steps
import time
from data_utils import clear_output
# def get_args():
#

if __name__ == '__main__':
    batch_size = 16
    output_shape = (128, 128)
    # dataset_resize('./data')  # , resize_dst=(128, 128))
    clear_output()
    steps = calculate_steps('./data', batch_size=batch_size)
    train_generator = train_generator('./data', batch_size=batch_size, resize_dst=output_shape)
    anime_gan = General_GAN(batch_size=batch_size, output_shape=output_shape)
    # def train():
    #     dist.init_parallel_env()
    datetime_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    anime_gan.fit(log_file='train_{}.log'.format(datetime_str), train_generator=train_generator,
                  steps_tqdm=steps, input_constant=False, new_batch4generator_training=True)

    # dist.spawn(train, nprocs=2)

