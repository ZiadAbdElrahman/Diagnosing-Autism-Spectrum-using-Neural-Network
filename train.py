import time
from collections import defaultdict
import gc
import pandas as pd
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from options.train_options import TrainOptions

from data import create_dataset
from models import create_model


class Trainer:
    def __init__(self):

        self.opt = None
        self.val_opt = None
        self.dataset = None
        self.dataset_size: int = None
        self.val_dataset = None
        self.val_dataset_size: int = None

        self.setup_training()
        if not self.opt.test:
            self.setup_val()

        self.model = create_model(self.opt)  # create a model given opt.model and other options
        self.model.setup(self.opt)
        self.writer = SummaryWriter(self.model.save_dir)

    def setup_training(self):
        self.opt = TrainOptions().parse(is_train=True)
        self.dataset = create_dataset(self.opt)  # create a dataset given opt.dataset_mode and other options
        self.dataset_size = len(self.dataset)  # get the number of images in the dataset.
        print('The number of training images = %d' % self.dataset_size)

    def setup_val(self):
        self.val_opt = TrainOptions().parse(is_train=False)
        self.val_dataset = create_dataset(self.val_opt)  # create a dataset given opt.dataset_mode and other options
        self.val_dataset_size = len(self.val_dataset)  # get the number of images in the dataset.
        print('The number of val images = %d' % self.val_dataset_size)

    def print_itr_loss(self, epoch, data_idx, time, dataset_size, suffix=''):
        itr_losses = self.model.get_current_losses()
        message = f'epoch:{epoch}, batch:{data_idx}/{dataset_size // self.opt.batch_size} time:{time:.4f}s  '
        message += ' '.join(f'{k}{suffix}:{v:.4f} ' for k, v in itr_losses.items())
        message += '\n'

        print(message)

    def run(self):
        if self.opt.test:
            with torch.no_grad():
                self.test_epoch()
        else:
            self.train()

    def train(self):
        for epoch in range(self.opt.epoch_count, self.opt.n_epochs + self.opt.n_epochs_decay + 1):
            epoch_start_time = time.time()  # timer for entire epoch

            self.train_epoch(epoch)
            self.model.save_networks(epoch)
            with torch.no_grad():
                self.val_epoch(epoch)

            message = f'Epoch {epoch} End, time {time.time() - epoch_start_time:.4f} S'
            print(message)

    def train_epoch(self, epoch):
        losses = defaultdict(list)
        model = self.model
        dataset = self.dataset
        opt = self.opt
        epoch_start_time = time.time()  # timer for entire epoch

        for data_idx, data in enumerate(dataset):
            iter_start_time = time.time()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.forward()
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            for k, v in self.model.get_current_losses().items():
                losses[k].append(v)

            if data_idx % opt.print_freq == 0:  # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                self.print_itr_loss(epoch, data_idx, t_comp, self.dataset_size)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

        self.log_losses(losses, epoch)
        print(f'End of training epoch {epoch}, Time Taken: {time.time() - epoch_start_time:.4f} Sec')

    def val_epoch(self, epoch):
        losses = defaultdict(list)
        opt = self.opt
        dataset = self.val_dataset
        model = self.model
        epoch_start_time = time.time()  # timer for entire epoch

        for data_idx, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.forward()
            model.compute_losses()
            for k, v in self.model.get_current_losses().items():
                losses[f'{k}_val'].append(v)

            if data_idx % opt.print_freq == 0:  # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                self.print_itr_loss(epoch, data_idx, t_comp, self.val_dataset_size, '_val')
            gc.collect()
        self.log_losses(losses, epoch)

        print(f'End of val epoch {epoch}, Time Taken: {time.time() - epoch_start_time:.4f} Sec')

    def test_epoch(self):
        submit = pd.read_csv('fcis-asu-autism-disorder-classification/Submit.csv')

        model = self.model
        dataset = self.dataset
        dataset_size = len(self.dataset)
        opt = self.opt
        epoch_start_time = time.time()  # timer for entire epoch
        outputs = []
        names = []
        for data_idx, data in enumerate(dataset):

            iter_start_time = time.time()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.forward()
            outputs += [int(model.get_outputs()[:, 0].detach().cpu().numpy())]
            names += data['name']
            if data_idx % opt.print_freq == 0:  # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                message = f'batch:{data_idx}/{dataset_size // self.opt.batch_size} time:{t_comp:.4f}s '
                print(message)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()
        submit = pd.DataFrame({'Image': names, 'Label': outputs})
        # submit['Image'] = names
        # submit['Label'] = outputs
        submit.to_csv('gray_submit.csv', index=False)

        print(f'End of testing epoch, Time Taken: {time.time() - epoch_start_time:.4f} Sec')

    def log_losses(self, losses, epoch):
        for k, v in losses.items():
            self.writer.add_scalar(k, np.array(v).mean(), epoch)


if __name__ == '__main__':
    Trainer().run()
