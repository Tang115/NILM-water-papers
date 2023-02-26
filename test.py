import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime
import copy
import time


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, self.hidden_size).cuda()
        out, _ = self.lstm(x.to(torch.float32).cuda(), (h0, c0))
        out = self.fc(out)
        return out


class NILM_related_tasks(object):
    def __init__(self):
        self.set_hyper_para()

    def set_hyper_para(self):
        # We will set all the parameters in this function

        # data process related
        self.seq_length = 64    # This number is the window size of LSTM
        self.ratio_of_sets = 0.5   # ratio≈(train size)/(size of total data)

        # LSTM related
        self.input_size = self.seq_length
        self.hidden_size = 256
        self.output_size = 1
        self.num_layers = 3

        # train related
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.batch_size = 500

    def load_dataset(self, train_or_test = "Train", skip_dp = True):
        if skip_dp == False:
            self.data_process()

        data = np.load('Concise_House_2.npy', allow_pickle=True)
        seq_length = self.seq_length
        X = []  # model input
        y = []  # label
        for i in range(data.shape[0] - seq_length + 1):
            X.append(data[i:i + seq_length, 1])
            y.append(data[i + seq_length - 1, 2])
            print(i)
        X = np.array(X)
        y = np.array(y)

        train_size = X.shape[0]*self.ratio_of_sets
        train_size = int(np.ceil(train_size/self.batch_size) * self.batch_size)
        test_size = X.shape[0] - train_size
        test_size = int(np.floor(test_size / self.batch_size) * self.batch_size)

        if train_or_test == "Train":
            X = X[0:train_size, :]
            y = y[0:train_size]

        if train_or_test == "Test":
            X = X[-test_size:, :]
            y = y[-test_size:]

        self.X = torch.from_numpy(X.astype(float)).view(-1, self.batch_size, X.shape[1]).cuda()
        self.y = torch.from_numpy(y.astype(float)).view(-1, self.batch_size, 1).cuda()


    def data_process(self):
        # the original data is for every 6s, this data process is to aggregate the data to min-level

        df = pd.read_csv(os.getcwd() + '/House_2.csv')
        self.data = df.to_numpy()

        record_rows = []
        Concise_data = []

        for row in range(self.data.shape[0]):
            print(row)
            current_time = datetime.datetime.strptime(self.data[row, 0], '%Y-%m-%d %H:%M:%S')
            current_time = current_time.replace(second=0, microsecond=0)

            if row == 0:
                record_time = copy.deepcopy(current_time)

            if current_time > record_time:
                one_min_data = np.asarray(self.data[record_rows, 2:], dtype=np.float32)
                one_min_data = np.around(np.mean(one_min_data, axis=0))
                time_str = record_time.strftime('%Y-%m-%d %H:%M:%S')

                Concise_data.append(np.concatenate(([time_str], one_min_data)))

                record_time = current_time
                record_rows = [row]
            else:
                record_rows.append(row)

        Concise_data = np.array(Concise_data)
        np.save(os.getcwd() + '/Concise_House_2.npy', Concise_data)

    def train(self):
        model = LSTM(self.input_size, self.hidden_size, self.output_size, self.num_layers).cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        losses = []
        for epoch in range(self.num_epochs):
            for i, batch in enumerate(self.X):
                labels = self.y[i]
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs.to(torch.float64).cuda(), labels)
                loss.backward()
                optimizer.step()
                if (i + 1) % 1 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, self.num_epochs, i + 1, len(self.X), loss.item()))
                losses.append(loss.cpu().detach().numpy())
        self.model = model
        self.losses = losses

    def test(self):
        model = self.model
        model.eval()
        with torch.no_grad():
            predicts = []
            ground_trues = []
            errors = []
            for i in range(self.X.shape[0]):
                output = model(self.X[i])
                for j in range(output.shape[0]):
                    predict = output.data[j].cpu().detach().numpy()
                    ground_true = self.y.data[i,j].cpu().detach().numpy()
                    predicts.append(predict[0])
                    ground_trues.append(ground_true[0])
                    errors.append(predict[0] - ground_true[0])
                    print("predict: ", predict, "ground-true: ", ground_true)
        print("mean_abs_error: ", np.mean(np.abs(errors)))

        self.predicts = predicts
        self.ground_trues = ground_trues
        self.errors = errors

    def render(self):
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # not sure why I have this issue, you may ignore
        # loss
        w = 10  #window_size
        moving_avg_loss = np.convolve(self.losses, np.ones(w), 'valid') / w
        for i in range(w-1):
            moving_avg_loss = np.append(moving_avg_loss, moving_avg_loss[-1])
        plt.figure()
        plt.plot(self.losses, label='Loss')
        plt.plot(moving_avg_loss, label=f'Moving Average Loss (Window Size={w})')
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.savefig('loss.png')
        #plt.show()

        #  predict and true
        plt.figure()
        plt.plot(self.ground_trues, label='Ground Trues', linewidth=0.2)
        plt.plot(self.predicts, label='Predictions', linewidth=0.2)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('W')
        plt.savefig('predict_true.png')
        #plt.show()


def main():
    t1 = time.time()
    NILM = NILM_related_tasks()

    NILM.load_dataset(train_or_test="Train", skip_dp=True)
    NILM.train()

    NILM.load_dataset(train_or_test="Test")
    NILM.test()

    NILM.render()
    t2 = time.time()
    print('Time cost: ', t2 - t1, 's')


if __name__ == "__main__":
    main()





