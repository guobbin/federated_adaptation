import argparse
import json
from datetime import datetime
import os
import logging
import torch
import torch.nn.functional as F
from utils.utils import test as utiltest

print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils.image_helper import GateHelper

logger = logging.getLogger("logger")
import yaml
import time
import numpy as np

import random
from utils.utils import *
from copy import deepcopy

criterion = torch.nn.CrossEntropyLoss(reduction='none')


def eval_one_participant(helper, data_source, model):
    model.eval()
    correct = 0.0
    total_test_words = 0.0
    dataset_size = len(data_source.dataset)
    data_iterator = data_source

    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch, evaluation=True)
            output = model(data)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
    return acc


def test_globalmodel_local(helper, data_sets, target_model):
    globalmodel_local_acc = list()
    for model_id in range(len(data_sets)):
        model = target_model
        model.eval()
        _, (current_data_model, test_data) = data_sets[model_id]
        local_acc = eval_one_participant(helper, test_data, model)
        globalmodel_local_acc.append(local_acc)
    return globalmodel_local_acc


def adapt_local(helper, train_data_sets, fisher, target_model, local_model, gate_model, adaptedmodel_local_acc):
    for parame in target_model.parameters():
        parame.requires_grad = False
    for parame in local_model.parameters():
        parame.requires_grad = False
    print("global target model from resume")
    utiltest(helper=helper, data_source=helper.test_data, model=target_model)

    for model_id in tqdm(range(len(train_data_sets))):
        iteration = 0
        # init gate for user_model_id
        model = gate_model
        for m in model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        local_params = torch.load(
            f"{helper.params['repo_path']}/saved_models/{helper.params['local_best_folder']}/local_model{model_id}.pt.tar.best")
        local_model.load_state_dict(local_params['state_dict'])

        optimizer = torch.optim.SGD(model.parameters(), lr=helper.gate_lr,
                                    momentum=helper.gate_momentum,
                                    weight_decay=helper.gate_decay)
        model.train()

        _, (current_data_model, train_data) = train_data_sets[model_id]
        image_trainset_weight = np.zeros(10)
        for ind, x in enumerate(train_data):
            _, label = x
            for labeli in range(10):
                image_trainset_weight[labeli] += (label == labeli).sum()
        image_trainset_weight = image_trainset_weight / image_trainset_weight.sum()

        helper.writer.add_histogram(f'user_{model_id}/data_distribution',
                                 np.array(train_data.dataset.targets)[train_data.sampler.indices],
                                 bins=np.arange(11))

        start_time = time.time()
        total_test_loss, total_test_acc, correct_class_acc = utiltest(helper=helper, data_source=helper.test_data, model=helper.local_model)
        helper.writer.add_scalar(f'user_{model_id}/local_test_acc', (correct_class_acc * image_trainset_weight).sum(),
                                 0)
        helper.writer.add_scalar(f'user_{model_id}/total_test_loss', total_test_loss, 0)
        helper.writer.add_scalar(f'user_{model_id}/total_test_acc', total_test_acc, 0)
        for internal_epoch in range(1, helper.adaptation_epoch + 1):
            model.train()
            data_iterator = train_data
            batch_num = 0
            for batch_id, batch in enumerate(data_iterator):
                iteration += 1
                batch_num += 1
                optimizer.zero_grad()
                data, targets = helper.get_batch(train_data, batch,
                                                 evaluation=False)
                output1 = local_model(data)
                output2 = target_model(data)
                gate = model(data)

                loss1 = F.cross_entropy(output1, targets, reduction='none')
                loss2 = F.cross_entropy(output2, targets, reduction='none')
                gate_label = (loss1 > loss2).long()
                loss = F.cross_entropy(gate, gate_label)

                # output = gate * output1 + (1-gate) * output2
                # loss = criterion(output, targets).mean()
                # loss = (gate.view(batch[1].shape)*criterion(output1, targets) + (1-gate).view(batch[1].shape)*criterion(output2, targets)).mean()
                loss.backward()
                optimizer.step()

            helper.writer.add_scalar(f'user_{model_id}/gate_local_global_loss', loss, internal_epoch)

            if internal_epoch == 1 or internal_epoch % helper.test_each_epochs == 0 or internal_epoch == helper.adaptation_epoch:
                total_test_loss, total_test_acc, correct_class_acc = test(helper=helper, data_source=helper.test_data, model=model)
                helper.writer.add_scalar(f'user_{model_id}/local_test_acc', (correct_class_acc * image_trainset_weight).sum(),
                                         internal_epoch)
                helper.writer.add_scalar(f'user_{model_id}/total_test_loss', total_test_loss, internal_epoch)
                helper.writer.add_scalar(f'user_{model_id}/total_test_acc', total_test_acc, internal_epoch)
        t = time.time()
        logger.info(f'time spent on local adaptation: {t - start_time}')
        logger.info(f'testing adapted model on local testset at model_id: {model_id}')

        _, _, correct_class_acc = test(helper=helper, data_source=helper.test_data, model=model)
        adaptedmodel_local_acc.append((correct_class_acc * image_trainset_weight).sum())

        logger.info(f'time spent on testing: {time.time() - t}')
        if (model_id + 1) % 100 == 0 or (model_id + 1) == len(train_data_sets):
            logger.info(f'Saved adaptedmodel_local_acc at model_id: {model_id}')
            np.save(helper.save_name + '_AdaptedModel_LocalTest_Acc.npy', np.array(adaptedmodel_local_acc))


def test(helper, data_source, model):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    correct_class = np.zeros(10)
    correct_class_acc = np.zeros(10)
    c_class = np.zeros(10)
    loss_class_acc = np.zeros(10)
    correct_class_size = np.zeros(10)
    total_test_words = 0.0
    dataset_size = len(data_source.dataset)
    data_iterator = data_source
    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch, evaluation=True)
            output1 = helper.local_model(data)
            output2 = helper.target_model(data)
            gate = torch.softmax(helper.gate_model(data), dim=1)
            loss1 = F.cross_entropy(output1, targets, reduction='none')
            loss2 = F.cross_entropy(output2, targets, reduction='none')

            # gate = torch.zeros_like(gate)
            total_loss += (torch.cat((loss1.unsqueeze(1), loss2.unsqueeze(1)), 1) * gate).sum()

            output = gate[:, 0].unsqueeze(dim=1) * torch.softmax(output1, dim=1) + gate[:, 1].unsqueeze(dim=1) * torch.softmax(output2, dim=1)
            # total_loss += (gate.view(batch[1].shape) * criterion(output1, targets) + (1 - gate).view(batch[1].shape) * criterion(output2, targets)).sum()
            # output = output1 * gate + output2 * (1-gate)
            # total_loss += criterion(output, targets).sum()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
            for i in range(10):
                class_ind = targets.data.view_as(pred).eq(i*torch.ones_like(pred))
                correct_class_size[i] += class_ind.cpu().sum().item()
                correct_class[i] += (pred.eq(targets.data.view_as(pred))*class_ind).cpu().sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))
        for i in range(10):
            correct_class_acc[i] = (float(correct_class[i]) / float(correct_class_size[i]))
        total_l = total_loss / dataset_size
        print(f'___Test {model.name} , Average loss: {total_l},  '
                    f'Accuracy: {correct}/{dataset_size} ({acc}%)')
        return total_l, acc, correct_class_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='./utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)

    args = parser.parse_args()
    d = datetime.now().strftime('%b.%d_%H.%M.%S')

    with open(args.params) as f:
        params_loaded = yaml.load(f)

    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')

    adaptation_helper = GateHelper(current_time=current_time, params=params_loaded,
                                        name=params_loaded.get('name', 'image_adapt'))

    if not adaptation_helper.random:
        adaptation_helper.fix_random()

    adaptation_helper.load_data()
    adaptation_helper.create_model()

    # configure logging
    wr = SummaryWriter(log_dir=f'{adaptation_helper.repo_path}/runs/gate_adapt_{args.name}_{current_time}')
    adaptation_helper.writer = wr

    if adaptation_helper.log:
        logger = create_logger()
        fh = logging.FileHandler(filename=f'{adaptation_helper.folder_path}/log.txt')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.warning(f'Logging things. current path: {adaptation_helper.folder_path}')
        adaptation_helper.params['tb_name'] = args.name
        with open(f'{adaptation_helper.folder_path}/params.yaml.txt', 'w') as f:
            yaml.dump(adaptation_helper.params, f)
    else:
        logger = create_logger()

    if adaptation_helper.tb:
        table = create_table(adaptation_helper.params)
        adaptation_helper.writer.add_text('Model Params', table)
        print(adaptation_helper.lr, table)

    participant_ids = range(len(adaptation_helper.train_data))
    mean_acc = list()

    # save parameters:
    with open(f'{adaptation_helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(adaptation_helper.params, f)
    if not adaptation_helper.only_eval:
        random.seed(66)
        adaptedmodel_local_acc = list()
        subset_data_chunks = participant_ids
        logger.info(f'Selected adapted models ID: {subset_data_chunks}')
        t1 = time.time()
        adapt_local(helper=adaptation_helper,
                    train_data_sets=[(pos, adaptation_helper.train_data[pos]) for pos in subset_data_chunks],
                    fisher=None, target_model=adaptation_helper.target_model, gate_model=adaptation_helper.gate_model,
                    local_model=adaptation_helper.local_model, adaptedmodel_local_acc=adaptedmodel_local_acc)
        logger.info(f'time spent on local adaptation: {time.time() - t1}')
    logger.info(f"Evaluate the global (target) model on participants' local testdata to get local accuracies of federated learning model")
    _, _, globalmodel_correct_class_acc = test(helper=adaptation_helper, data_source=adaptation_helper.test_data,
                                               model=adaptation_helper.target_model)
    globalmodel_local_acc = (globalmodel_correct_class_acc * adaptation_helper.train_image_weight).sum(-1)
    np.save(adaptation_helper.save_name + '_GlobalModl_LocalTest_Acc.npy', np.array(globalmodel_local_acc))
    logger.info(f"This run has a label: {adaptation_helper.params['current_time']}. ")