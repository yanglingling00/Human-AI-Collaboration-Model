import random
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import os
import logging
import torch.nn.functional as F

class synthetic_expert():
    '''
    synthetic expert on our data based on human_ability
    '''

    def __init__(self, human_ability, num_classes):
        self.n_classes = num_classes
        # Human experts predict the probability of success
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
        out = F.dropout(out, 0.1, training=self.training)
        out = self.fc2(out)
        out = self.relu(out)
        out = F.dropout(out, 0.1, training=self.training)
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

def adjust_shape(dataSet, expert, seq):
    dataSet_x = dataSet[:, :-1]
    dataSet_y = dataSet[:, -1]
    # Features and Labels
    data_feat, data_target = [], []

    # each data sequence have seq groups of data
    for index in range(len(dataSet) - seq):
        data_feat.append(dataSet_x[index:index + seq])
        data_target.append(dataSet_y[index + seq - 1])

    data_feat = np.array(data_feat, dtype='float64')
    data_target = np.array(data_target, dtype='float64')
    data_human = expert.predict(data_feat, data_target)   # The label results predicted by humans
    return data_feat, data_human, data_target

class early_stop():
    def __init__(self, period, EARLY_STOP_EPOCH):
        self.best_model = 0
        self.loss_list = [0] * EARLY_STOP_EPOCH
        self.min_loss = 1000000
        self.period = period

    def stop_point(self, new_loss, model, SEQ, SEQ_index):
        if new_loss < self.min_loss:
            self.min_loss = new_loss
            self.best_model = model
            torch.save(self.best_model, f'best_model_period_{self.period}_{SEQ}_{SEQ_index}.pth')

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

def metrics_print(model, expert, num_classes, loader, device, logger):
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
            feature, human_pre, labels = data
            feature, human_pre, labels = feature.to(device), human_pre.to(device), labels.to(device)
            outputs = model(feature)[:, -1, :]
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size
            exp_prediction = human_pre
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

num_classes = 2

def temp_test(model, expert, loader, device, logger):
    model.eval()
    metrics_print(model, expert, num_classes, loader, device, logger)

def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_monitor_process(period, currency, time_frame, human_ability, EARLY_STOP_EPOCH, alpha, indexs):
    EPOCHS = 500
    LR = 1e-3
    TRAIN_BATCH_SIZE = 120
    TEST_BATCH_SIZE = 120

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    train_size = 0.7
    valid_size = 0.1
    test_size = 0.2
    num_classes = 2

    # Synthetic expert
    expert = synthetic_expert(human_ability, num_classes)

    # Record output
    logger = logging.getLogger(f'logger_{period}')

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    # Configure logging
    handler = logging.FileHandler(f'{period}_out.txt')
    logger.addHandler(handler)

    logger.info(f"\n===== Running with alpha = {alpha} =====")

    res = {}
    SEQ_list = [5,10,15]
    for SEQ in SEQ_list:
        for SEQ_index in range(indexs):
            logger.info(f"---------------SEQ:{SEQ} SEQ_index:{SEQ_index}-------------")
            file = (f"period_{period}_GMT_{currency}_{time_frame}.xlsx")
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
            data_feat, data_human, data_target = adjust_shape(dataSet, expert, SEQ)

            # Obtain the human prediction 
            # check file to see if it exists
            if os.path.exists('human_pre.csv'):
                data_human = pd.read_csv('human_pre.csv').values
                data_human = data_human.flatten()
            else:
                df = pd.DataFrame(data_human)
                df.to_csv('human_pre.csv', index=False)

            # Build a time window
            trainX = torch.from_numpy(data_feat[:split_idx_1].reshape(-1, SEQ, input_size)).type(torch.Tensor)
            validX = torch.from_numpy(data_feat[split_idx_1:split_idx_2].reshape(-1, SEQ, input_size)).type(
                torch.Tensor)
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

            model = Net(input_size=input_size, hidden_size=120, output_size=(num_classes + 1), num_layers=2,
                        device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR) 
            loss_fn = softamax_loss  

            Stop_early = early_stop(period, EARLY_STOP_EPOCH)
            model = model.to(device)
            cudnn.benchmark = True

            # train model
            for epoch in range(1, EPOCHS + 1):
                model.train()
                all_loss = 0
                for i, (batch_input, batch_human, batch_targets) in enumerate(train_loader):
                    batch_input = batch_input.to(device)
                    batch_targets = batch_targets.to(device)
                    # Only obtain the last predicted value of the window
                    y_train_pred = model(batch_input)[:, -1, :]
                    # Get the expert prediction results
                    m = expert.predict(batch_input, batch_targets)
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
                    optimizer.zero_grad()
                    loss.backward()
                    # Update
                    optimizer.step()
                    all_loss += loss
                if epoch % 10 == 0:
                    print(epoch)
                    logger.info(f"epoch:{epoch} loss:{all_loss / TRAIN_BATCH_SIZE}")
                    temp_test(model, expert, test_loader, device, logger)
                # -----------------------------Early Stop---------------------------------------------
                # Retain the model with the least loss on valid
                model.eval()
                with torch.no_grad():
                    all_loss = 0
                    for data in valid_loader:
                        x, human, y = data
                        x = x.to(device)
                        human = human.to(device)
                        y = y.to(device)
                        outputs = model(x)[:, -1, :] 
                        m = human
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
                        all_loss += loss
                    if Stop_early.stop_point((all_loss / TEST_BATCH_SIZE), model, SEQ, SEQ_index):
                        print("early stop")
                        logger.info(f"early stop epoch:{epoch}")
                        break
                    else:
                        continue
            # Load the best model
            model = torch.load(f'best_model_period_{period}_{SEQ}_{SEQ_index}.pth')
            # Output
            model.eval()
            res[(SEQ, SEQ_index)] = metrics_print(model, expert, num_classes, test_loader, device, logger)

    return max(res, key=res.get)


