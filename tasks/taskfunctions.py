from pyexpat import model
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import analysis.plots as plots
import numpy as np
import itertools

from configs.config_global import DEVICE
from configs.configs import SupervisedLearningBaseConfig, NeuralPredictionConfig
from utils.logger import Logger
from models import Decoder
from typing import List

def data_batch_to_device(data_b, device=DEVICE):
    if type(data_b) is torch.Tensor:
        return data_b.to(device)
    elif type(data_b) is tuple or type(data_b) is list:
        return [data_batch_to_device(data, device) for data in data_b]
    else:
        print("DEBUG BAD TYPE DETECTED")
        print("TYPE:", type(data_b))

        if isinstance(data_b, dict):
            for k, v in data_b.items():
                print("  key:", k, "type:", type(v))

        elif isinstance(data_b, (list, tuple)):
            for i, v in enumerate(data_b):
                print("  idx:", i, "type:", type(v))

        else:
            print("VALUE:", data_b)
        raise NotImplementedError("input type not recognized")

class TaskFunction:

    def __init__(self, config: SupervisedLearningBaseConfig):
        """
        initailize information from config
        """
        pass

    def roll(self, model: nn.Module, data_batch: tuple, phase: str = 'train'):
        """
        Phase can be 'train', 'val', or 'test'.
        In the training mode (train=True), this method should return a scalar representing the training loss.
        In the eval or test mode, this method should return a pair of scalars: the loss and the number of predictions
        """
        raise NotImplementedError

    def after_testing_callback(self, logger: Logger, save_path: str, is_best: bool, batch_num: int, testing: bool = False):
        """
        An optional function called after the test stage
        :param is_best: whether the validation result is the best one; always True for the test set (when testing=True)
        :param batch_num: number of batches used for training
        :param testing: whether the test is on the test set
        """
        pass

    def after_training_callback(self, config, model):
        """
        An optional function called after training is done
        
        :param model: the best model
        """
        pass

class NeuralPrediction(TaskFunction):
    """
    Task function for autoregressive neural prediction
    """

    def __init__(self, config: NeuralPredictionConfig, input_size):
        self.criterion = nn.MSELoss(reduction='sum')
        self.do_analysis = config.do_analysis
        self.loss_mode = config.loss_mode
        self.pred_step = config.pred_length
        self.seq_len = config.seq_length
        self.config = config
        self.mse_baseline = {}

        self.session_ids = {} # session id for each dataset
        n_sessions = 0
        for dataset, size in zip(config.dataset_label, input_size):
            self.session_ids[dataset] = list(range(n_sessions, n_sessions + len(size)))
            n_sessions += len(size)
        print(f'session_ids: {self.session_ids}')
        self.input_size = list(itertools.chain(*input_size))
        self._init_info()

        assert self.loss_mode in ['prediction', 'autoregressive', 'reconstruction', 'reconstruction_prediction']
        if self.loss_mode == 'reconstruction':
            assert self.pred_step == 0, 'reconstruction mode does not support prediction'
        else:
            assert self.pred_step > 0, 'prediction step should be larger than 0'

    def _init_info(self):
        self.sample_trials = {'train': [], 'val': [], 'test': []}
        self.pred_num = {'train': [], 'val': [], 'test': []}
        self.sum_mse = {'train': [], 'val': [], 'test': []}
        self.sum_mae = {'train': [], 'val': [], 'test': []}
        
        for phase in ['train', 'val', 'test']:
            for size in self.input_size:
                self.pred_num[phase].append(np.zeros(size))
                L = self.pred_step if self.pred_step > 0 else self.seq_len
                self.sum_mse[phase].append(np.zeros((L, size)))
                self.sum_mae[phase].append(np.zeros((L, size))) # L, D

    def roll(self, model: nn.Module, data_batch: tuple, phase: str = 'train'):
        
        # print("\n===== DATA INSIDE ROLL =====")
        # print(type(data_batch))
        # if isinstance(data_batch, dict):
        #     for k, v in data_batch.items():
        #         print("key:", k, "type:", type(v))
        # print("============================\n")
        
        train_flag = phase == 'train'
        
        input, target, info_list = data_batch
        print("DEBUG input is:", type(input), len(input) if isinstance(input, list) else "NOT A LIST")
        print("DEBUG input[0] shape:", input[0].shape if isinstance(input, list) else input.shape)
        print("AFTER UNPACK")
        print(type(input), type(target), type(info_list))
        print("input shape:", getattr(input, "shape", None))
        print("\n==== STRUCTURE BEFORE DEVICE MOVE ====")
        print("input type:", type(input))
        print("target type:", type(target))
        print("input[0] shape:", input[0].shape)
        print("target[0] shape:", target[0].shape)

        if isinstance(input, (list, tuple)):
            print("input[0] type:", type(input[0]))
        if isinstance(target, (list, tuple)):
            print("target[0] type:", type(target[0]))

        if isinstance(input, dict):
            print("input keys:", input.keys())
        if isinstance(target, dict):
            print("target keys:", target.keys())

        print("======================================\n")
        input, target = data_batch_to_device((input, target))
        input = [x.to(next(model.parameters()).dtype) for x in input]
        output = model(input)
    
        task_loss = torch.zeros(1).to(DEVICE)
        pred_num = 0

        mean_list = []
        std_list = []

        for idx, (inp, out, tar, info) in enumerate(zip(input, output, target, info_list)):
            if target is None or np.prod(tar.shape) == 0:
                continue

            if self.loss_mode[: 14] == 'reconstruction':
                tar = torch.cat([inp[: 1], tar], dim=0)
            elif self.loss_mode == 'prediction':
                out = out[-self.pred_step: ]
                tar = tar[-self.pred_step: ]
            else:
                assert self.loss_mode == 'autoregressive'

            if len(self.sample_trials[phase]) < len(input):
                self.sample_trials[phase].append((
                    out.detach().float().cpu().numpy(),
                    tar.detach().float().cpu().numpy(),
                    inp.detach().float().cpu().numpy()
                ))
            
            with torch.no_grad():
                if self.pred_step > 0:
                    loss_array = torch.abs(out - tar)[-self.pred_step: ] # shape: L, B, D
                    assert loss_array.shape[0] == self.pred_step
                else:
                    loss_array = torch.abs(out - tar)

                self.sum_mse[phase][idx] += loss_array.pow(2).sum(dim=1).cpu().numpy()
                self.sum_mae[phase][idx] += loss_array.sum(dim=1).cpu().numpy()
                self.pred_num[phase][idx] += loss_array.shape[1]

            task_loss += self.criterion(out, tar) 
            pred_num += np.prod(tar.shape)

            tar = tar[-self.pred_step: ]
            mean_list.append(torch.mean(tar, dim=0).reshape(-1))
            std_list.append(torch.std(tar, dim=0).reshape(-1))

        task_loss = task_loss / max(1, pred_num)

        if train_flag:
            return task_loss
        else:
            return task_loss, pred_num
        
    def plot_prediction(self, save_path: str, phase: str):
        # visualize prediction for selected dimensions
        for pc in [0, 20, 50, -1]:
            if pc >= max(self.input_size):
                continue

            preds = []
            targets = []

            for iter in range(10):
                batch_idx = np.random.randint(len(self.sample_trials[phase]))
                data = self.sample_trials[phase][batch_idx]
                sample_idx = np.random.randint(data[0].shape[1])

                pred = data[0][:, sample_idx, pc]
                target = data[1][:, sample_idx, pc]
                inp = data[2][:, sample_idx, pc]

                if self.loss_mode == 'prediction': # get the whole trial
                    pred = np.concatenate([inp, pred])
                    target = np.concatenate([inp, target])
                preds.append(pred)
                targets.append(target)

            plots.pred_vs_target_plot(
                list(zip(preds, targets)), 
                os.path.join(save_path, f'{phase}_pred_vs_target'), 
                f'best_dim{pc}',
                len(preds[0]) - self.pred_step
            )
        
    def after_testing_callback(self, logger: Logger, save_path: str, is_best: bool, batch_num: int, testing: bool = False):

        # compute mean loss for the prediction part
        for phase in ['train', 'val']:

            if phase == 'val':
                assert self.mse_baseline.get(phase) is not None, 'mse_baseline is not set'
            if np.sum(self.pred_num[phase][0]) == 0:
                continue

            for dataset, session_ids in self.session_ids.items():
                sum_pred_num = 0
                sum_mse = 0
                sum_mae = 0
                avg_mse_score = 0

                for idx in session_ids:
                    sum_pred_num += self.pred_num[phase][idx].sum() * (self.pred_step if self.pred_step > 0 else self.seq_len)
                    sum_mse += self.sum_mse[phase][idx].sum()
                    sum_mae += self.sum_mae[phase][idx].sum()

                    if phase == 'val':
                        if self.pred_num[phase][idx].sum() == 0:
                            print(f"Skipping empty session {idx} in {phase}")
                            continue
                        mse = self.sum_mse[phase][idx].sum() / self.pred_num[phase][idx].sum() / (self.pred_step if self.pred_step > 0 else self.seq_len)
                        avg_mse_score += 1 - mse / self.mse_baseline[phase][idx]

                avg_mse_score /= len(session_ids)

                if phase == 'val':
                    logger.log_tabular(f'{dataset}_{phase}_pred_num', int(sum_pred_num))
                    logger.log_tabular(f'{dataset}_{phase}_score', avg_mse_score)
                
                logger.log_tabular(f'{dataset}_{phase}_mse', sum_mse / sum_pred_num)
                logger.log_tabular(f'{dataset}_{phase}_mae', sum_mae / sum_pred_num)
            
            if is_best and len(self.sample_trials[phase]) > 0:
                self.plot_prediction(save_path, phase)

        if is_best:
            for phase in ['train', 'val']:
                if testing and phase != 'val':
                    continue
                info = {
                    'mse': self.sum_mse[phase],
                    'mae': self.sum_mae[phase],
                    'pred_num': self.pred_num[phase],
                    'sample_trials': self.sample_trials[phase]
                }
                np.save(os.path.join(save_path, f'{phase if not testing else "test"}_best_info.npy'), info)

        self._init_info()
    
    def after_training_callback(self, config, model):
        if not self.do_analysis:
            return