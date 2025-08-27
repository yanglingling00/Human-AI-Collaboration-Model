import logging
import random
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_size = 0.7
valid_size = 0.1
test_size = 0.2
Batch_size = 120
TRAIN_BATCH_SIZE = 120
TEST_BATCH_SIZE = 120
num_classes = 2

class ensemble_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ensemble_model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(input_size, hidden_size * 3)
        self.fc2 = nn.Linear(hidden_size * 3, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Softmax(dim=1)

    def forward(self, input):
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc1(input)
        out = self.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc3(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc4(out)

        out = self.activation(out)
        return out

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        self.activation = nn.Softmax(dim=2)

    def forward(self, input):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_()
        # Forward propagate LSTM
        out, _ = self.lstm(input,
                           (h0.detach().to(self.device),
                            c0.detach().to(self.device)))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.activation(out)
        return out

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, feature, human_pre, data_label):
        self.feature = feature
        self.human_pre = human_pre
        self.label = data_label

    def __getitem__(self, index):
        feature = self.feature[index]
        human_pre = self.human_pre[index]
        labels = self.label[index]
        return feature, human_pre, labels
        
    def __len__(self):
        return len(self.feature)

def adjust_shape(dataSet, seq):
    dataSet_x = dataSet[:, :-1]
    dataSet_y = dataSet[:, -1]
    # Features and Labels
    data_feat, data_target = [], []

    # Each data sequence contains seq groups of data
    for index in range(len(dataSet) - seq):
        data_feat.append(dataSet_x[index:index + seq])
        data_target.append(dataSet_y[index + seq - 1])

    data_feat = np.array(data_feat, dtype='float64')
    data_target = np.array(data_target, dtype='float64')
    return data_feat, data_target

class synthetic_expert():
    '''
    If there is no human，synthetic expert
    '''

    def __init__(self, human_ability, num_classes):
        self.n_classes = num_classes
        self.human_ability = human_ability

    def predict(self, input, labels):
        batch_size = len(labels)
        outs = [0] * batch_size
        for i in range(0, batch_size):
            rand = random.random()
            if rand <= self.human_ability:
                outs[i] = labels[i]
            else:
                outs[i] = abs(labels[i] - 1)
        return outs

class early_stop():
    def __init__(self, EARLY_STOP_EPOCH):
        self.best_model = 0
        self.loss_list = [0] * EARLY_STOP_EPOCH
        self.min_loss = 1000000

    def stop_point(self, new_loss, model, seq, seq_index):
        if new_loss < self.min_loss:
            self.min_loss = new_loss
            self.best_model = model
            torch.save(self.best_model, f'best_dense_model_{seq}_{seq_index}.pth')

        self.loss_list = self.loss_list[1:]
        self.loss_list.append(new_loss)
        if min(self.loss_list) > self.min_loss:
            return True

# loss
def softamax_loss(outputs, m, labels, m2, n_classes):
    '''
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (I_{m =y})
    labels: target
    m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    n_classes: number of classes
    '''
    batch_size = outputs.size()[0]  # batch_size
    rc = [n_classes] * batch_size
    outputs = (-m) * torch.log2(outputs[range(batch_size), rc]) - (m2) * torch.log2(
        outputs[range(batch_size), labels.long()])
    return torch.sum(outputs) / batch_size

def metrics_print(model, expert, num_classes, loader, logger):
    up_total = 0
    down_total = 0
    up_correct = 0
    down_correct = 0
    total = 0
    correct_sys = 0
    exp_correct = 0
    exp_total = 0
    ai_correct = 0
    ai_total = 0

    # infer
    with torch.no_grad():
        for data in loader:
            feature, human, labels = data
            feature, labels = feature.to(device), labels.to(device)
            outputs = model(feature)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size
            exp_prediction = human.to(device)
            for i in range(0, batch_size):

                # Calculate whether to defer
                r = (predicted[i].item() == num_classes)
                sys_pre = -1
                if r == 0:
                    ai_correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                    sys_pre = predicted[i]
                    ai_total += 1
                if r == 1:
                    exp_correct += (exp_prediction[i] == labels[i].item())
                    correct_sys += (exp_prediction[i] == labels[i].item())
                    sys_pre = exp_prediction[i]
                    exp_total += 1
                total += 1
                # Calculate the up and down probabilities
                if labels[i] == 1:
                    up_total += 1
                    up_correct += (sys_pre == labels[i]).item()
                else:
                    down_total += 1
                    down_correct += (sys_pre == labels[i]).item()
    cov = str(exp_total) + str(" out of") + str(total)
    to_print = {"defer": cov, "sys_acc": 100 * correct_sys / total,
                "exp_acc": 100 * exp_correct / (exp_total + 0.0002),
                "classifier_acc": 100 * ai_correct / (ai_total + 0.0001),
                "up_acc": 100 * up_correct / (up_total + 0.0001),
                'down_acc': 100 * down_correct / (down_total + 0.0001)}
    logger.info(to_print)
    return 100 * correct_sys / total

def get_data(period, currency, time_frame, SEQ):
    """
   Encapsulate the data
    """
    file = f"period_{period}_GMT_{currency}_{time_frame}.xlsx"
    dataSet = pd.read_excel(file)
    close_index = dataSet.columns.get_loc('Close')
    data_after_close = dataSet.iloc[:, close_index:]
    dataSet = data_after_close.values

    # Divide the dataset
    split_idx_1 = int(len(dataSet) * train_size)  # Calculate the index to split the data
    split_idx_2 = int(len(dataSet) * (train_size + valid_size))  # Calculate the index to split the data
    train_data = dataSet[:split_idx_1]
    min_list = train_data[:, :-1].min(axis=0)
    max_list = train_data[:, :-1].max(axis=0)
    # Normalization
    dataSet[:, :-1] = (dataSet[:, :-1] - min_list) / (max_list - min_list)

    input_size = dataSet.shape[1] - 1
    data_feat, data_target = adjust_shape(dataSet, SEQ)

    data_human = pd.read_csv('human_pre.csv').values
    data_human = data_human.flatten()
    # Build a time window
    trainX = torch.from_numpy(data_feat[:split_idx_1].reshape(-1, SEQ, input_size)).type(torch.Tensor)
    validX = torch.from_numpy(data_feat[split_idx_1:split_idx_2].reshape(-1, SEQ, input_size)).type(torch.Tensor)
    testX = torch.from_numpy(data_feat[split_idx_2:].reshape(-1, SEQ, input_size)).type(torch.Tensor)
    train_human = torch.tensor(data_human[:split_idx_1], dtype=int)
    valid_human = torch.tensor(data_human[split_idx_1:split_idx_2], dtype=int)
    test_human = torch.tensor(data_human[split_idx_2:], dtype=int)
    trainY = torch.tensor(data_target[:split_idx_1], dtype=int)
    validY = torch.tensor(data_target[split_idx_1:split_idx_2], dtype=int)
    testY = torch.tensor(data_target[split_idx_2:], dtype=int)

    train = MyDataset(trainX, train_human, trainY)
    valid = MyDataset(validX, valid_human, validY)
    test = MyDataset(testX, test_human, testY)
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=TRAIN_BATCH_SIZE,
                                               shuffle=False, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid,
                                               batch_size=TEST_BATCH_SIZE,
                                               shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=TEST_BATCH_SIZE,
                                              shuffle=False, drop_last=True)
    return train_loader, valid_loader, test_loader

def calculate_entropy(p0, p1):
    total = p0 + p1
    p0_normalized = p0 / total
    p1_normalized = p1 / total

    entropy = - (p0_normalized * torch.log2(p0_normalized) +
                 p1_normalized * torch.log2(p1_normalized))

    entropy = torch.nan_to_num(entropy, nan=0.0)
    return entropy


def get_new_loader(loader, short_model, middle_model, long_model):

    short_model.eval()
    middle_model.eval()
    long_model.eval()

    new_input = None
    new_human = None
    new_target = None

    with torch.no_grad():
        for short_data, middle_data, long_data in loader:

            # short_data  list:[batch_input,batch_target]
            short_batch_input, short_human, short_batch_target = short_data[0], short_data[1], short_data[2]
            middle_batch_input, middle_batch_target = middle_data[0], middle_data[2]
            long_batch_input, long_batch_target = long_data[0], long_data[2]
            short_batch_input, short_batch_target = short_batch_input.to(device), short_batch_target.to(device)
            middle_batch_input, middle_batch_target = middle_batch_input.to(device), middle_batch_target.to(device)
            long_batch_input, long_batch_target = long_batch_input.to(device), long_batch_target.to(device)

             # [Batch_size,num_classes+1]
            short_batch_output = short_model(short_batch_input)[:, -1, :]  
            middle_batch_output = middle_model(middle_batch_input)[:, -1, :]  
            long_batch_output = long_model(long_batch_input)[:, -1, :]  

            # short
            machine_entropy = calculate_entropy(short_batch_output[:, 0], short_batch_output[:, 1])
            short_batch_output = torch.cat([short_batch_output, machine_entropy.unsqueeze(1)], dim=1)

            h_ai_entropy = calculate_entropy(
                torch.maximum(short_batch_output[:, 0], short_batch_output[:, 1]), short_batch_output[:, 2])
            short_batch_output = torch.cat([short_batch_output, h_ai_entropy.unsqueeze(1)], dim=1)

            # middle  
            machine_entropy = calculate_entropy(middle_batch_output[:, 0], middle_batch_output[:, 1])
            middle_batch_output = torch.cat([middle_batch_output, machine_entropy.unsqueeze(1)], dim=1)

            h_ai_entropy = calculate_entropy(
                torch.maximum(middle_batch_output[:, 0], middle_batch_output[:, 1]), middle_batch_output[:, 2])
            middle_batch_output = torch.cat([middle_batch_output, h_ai_entropy.unsqueeze(1)], dim=1)

            # long
            machine_entropy = calculate_entropy(long_batch_output[:, 0], long_batch_output[:, 1])
            long_batch_output = torch.cat([long_batch_output, machine_entropy.unsqueeze(1)], dim=1)

            h_ai_entropy = calculate_entropy(
                torch.maximum(long_batch_output[:, 0], long_batch_output[:, 1]), long_batch_output[:, 2])
            long_batch_output = torch.cat([long_batch_output, h_ai_entropy.unsqueeze(1)], dim=1)

            # merge
            merge_output = torch.cat((short_batch_output, middle_batch_output, long_batch_output), dim=1)
            if new_input is None:
                new_input = merge_output
            else:
                new_input = torch.cat((new_input, merge_output), dim=0)

            if new_human is None:
                new_human = short_human
            else:
                new_human = torch.cat((new_human, short_human), dim=0)

            if new_target is None:
                new_target = short_batch_target
            else:
                new_target = torch.cat((new_target, short_batch_target), dim=0)

    data_set = MyDataset(new_input, new_human, new_target)
    data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              batch_size=TRAIN_BATCH_SIZE,
                                              shuffle=False, drop_last=True)
    return data_loader

def temp_test(model, expert, loader, logger):
    model.eval()
    metrics_print(model, expert, num_classes, loader, logger)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_control_process(currency, time_frame, short_seq, short_index, m_seq, m_index, l_seq, l_index, human_ability,
                EARLY_STOP_EPOCH, alpha, indexs):
    EPOCHS = 500
    LR = 1e-3

    # If there is no human，synthetic expert
    expert = synthetic_expert(human_ability, num_classes)

    res = {}
    # SEQ_list = [5, 10, 15]
    SEQ_list = [5]
    short_model = torch.load(f'best_model_period_6_{short_seq}_{short_index}.pth')
    middle_model = torch.load(f'best_model_period_12_{m_seq}_{m_index}.pth')
    long_model = torch.load(f'best_model_period_24_{l_seq}_{l_index}.pth')

    logger = logging.getLogger(f'logger_stage')

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(f'stage2_output.txt')
    logger.addHandler(handler)

    logger.info(f"\n===== Running with alpha = {alpha} =====")

    for SEQ in SEQ_list:
        acc_list = []
        for seq_index in range(indexs):
            logger.info(f"---------------SEQ:{SEQ} SEQ_index:{seq_index}-------------")
            # Obtain data at different stages
            short_train_loader, short_valid_loader, short_test_loader = get_data(6, currency, time_frame, SEQ)
            middle_train_loader, middle_valid_loader, middle_test_loader = get_data(12, currency, time_frame, SEQ)
            long_train_loader, long_valid_loader, long_test_loader = get_data(24, currency, time_frame, SEQ)
            train_data_loader = zip(short_train_loader, middle_train_loader, long_train_loader)
            valid_data_loader = zip(short_valid_loader, middle_valid_loader, long_valid_loader)
            test_data_loader = zip(short_test_loader, middle_test_loader, long_test_loader)

            # Extract the new data
            ensemble_train_loader, ensemble_valid_loader, ensemble_test_loader = get_new_loader(
                train_data_loader, short_model, middle_model, long_model), get_new_loader(valid_data_loader,
                                                                                          short_model, middle_model,
                                                                                          long_model), get_new_loader(
                test_data_loader, short_model, middle_model, long_model)

            # Train the new model
            dense_model = ensemble_model(input_size=15, hidden_size=36, output_size=num_classes + 1)

            optimizer = torch.optim.Adam(dense_model.parameters(), lr=LR) 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * train_size)
            loss_fn = softamax_loss  
            Stop_early = early_stop(EARLY_STOP_EPOCH)
            dense_model = dense_model.to(device)
            cudnn.benchmark = True

            # train model
            for epoch in range(1, EPOCHS + 1):
                dense_model.train()
                all_loss = 0
                for i, (batch_input, batch_human, batch_targets) in enumerate(ensemble_train_loader):
                    batch_input = batch_input.to(device)
                    batch_targets = batch_targets.to(device)

                    y_train_pred = dense_model(batch_input)

                    m = batch_human.to(device)
                    m2 = [0] * TRAIN_BATCH_SIZE
                    for j in range(0, TRAIN_BATCH_SIZE):
                        if m[j] == batch_targets[j].item():
                            m[j] = 1
                            m2[j] = alpha
                        else:
                            m[j] = 0
                            m2[j] = 1
                    m = torch.tensor(m)
                    m2 = torch.tensor(m2)
                    m = m.to(device)
                    m2 = m2.to(device)

                    loss = softamax_loss(y_train_pred, m, batch_targets, m2, num_classes)
                    # Forward pass

                    optimizer.zero_grad()
                    loss.backward()
                    # Update
                    optimizer.step()
                    all_loss += loss
                if epoch % 10 == 0:
                    print(epoch)
                    logger.info(f"epoch:{epoch} loss:{all_loss / TRAIN_BATCH_SIZE}")
                    temp_test(dense_model, expert, ensemble_test_loader,logger)

                # -----------------------------Early Stop---------------------------------------------
                # Retain the model with the least loss on valid
                dense_model.eval()
                with torch.no_grad():
                    valid_loss = 0
                    for data in ensemble_valid_loader:
                        x, human, y = data
                        x = x.to(device)
                        y = y.to(device)
                        outputs = dense_model(x)
                        m = human.to(device)
                        m2 = [0] * TRAIN_BATCH_SIZE
                        for j in range(0, TRAIN_BATCH_SIZE):
                            if m[j] == y[j].item():
                                m[j] = 1
                                m2[j] = alpha
                            else:
                                m[j] = 0
                                m2[j] = 1
                        m = torch.tensor(m)
                        m2 = torch.tensor(m2)
                        m = m.to(device)
                        m2 = m2.to(device)

                        loss = softamax_loss(outputs, m, y, m2, num_classes)
                        valid_loss += loss
                    if Stop_early.stop_point((valid_loss / TEST_BATCH_SIZE), dense_model, SEQ, seq_index):
                        logger.info(f"early stop epoch:{epoch}")
                        print(f"early stop epoch")
                        break
                    else:
                        continue
            model = torch.load(f'best_dense_model_{SEQ}_{seq_index}.pth')
            # Output
            model.eval()
            res[(SEQ, seq_index)] = metrics_print(model, expert, num_classes, ensemble_test_loader, logger)

    return res






