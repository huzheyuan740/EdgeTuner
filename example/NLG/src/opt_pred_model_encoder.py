import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import sys, os
import time


class CustomDatasetHDF5(Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        self.f = h5py.File(data_file, 'r')
        self.num_samples = self.f['matrix'].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_x = torch.from_numpy(self.f['input_x'][idx])
        base_parm = torch.from_numpy(self.f['base_parm'][idx])
        lora_parm = torch.from_numpy(self.f['lora_parm'][idx])
        lora_grad = torch.from_numpy(self.f['lora_grad'][idx])
        matrix = torch.from_numpy(self.f['matrix'][idx])
        epoch = torch.tensor(self.f['epoch'][idx])
        return input_x, base_parm, lora_parm, lora_grad, matrix, epoch
    


class PredictNetwork(nn.Module):
    def __init__(self, k, m_size, seq_length, embedding_dim, layer_num, head_num):
        super(PredictNetwork, self).__init__()
        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # CNN for base and lora
        self.conv = nn.Conv2d(2+k, 16, kernel_size=3, padding=1)
        conv_output_dim = 16 * layer_num * head_num
        time_output_dim = seq_length * embedding_dim
        self.fc1 = nn.Linear(time_output_dim * m_size*k + conv_output_dim, 128)
        self.fc2 = nn.Linear(128, layer_num * head_num)

    def forward(self, time_data, base_features, lora_pram, lora_grad):
        batch_size = time_data.size(0)
        input_transformer = time_data.permute(1, 0, 2, 3)
        input_transformer = input_transformer.reshape(input_transformer.size(0), -1, input_transformer.size(3))
        time_features = self.transformer(input_transformer)
        time_features = time_features.permute(1, 0, 2).contiguous().view(batch_size, -1)

        # stack base and lora
        combined_conv_features = torch.stack((base_features, lora_pram, lora_grad), dim=1)
        conv_features = F.relu(self.conv(combined_conv_features))
        conv_features = conv_features.view(conv_features.size(0), -1)

        # fusion feature
        combined_features = torch.cat((time_features, conv_features), dim=1)

        x = F.relu(self.fc1(combined_features))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), -1)
        return x


class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, num_epochs, save_path='./model/iHeadPruner'):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.save_path = save_path

    def train(self):
        for epoch in range(self.num_epochs):
            train_step = 0
            running_loss = 0.0
            for input_x, base_parm, lora_parm, lora_grad, matrix, epoch in self.train_loader:

                print(f"Epoch: {epoch.squeeze()}")
                print(f"Input_x shape: {input_x.shape}")
                print(f"Base_parm shape: {base_parm.shape}")
                print(f"Lora_parm shape: {lora_parm.shape}")
                print(f"Lora_grad shape: {lora_grad.shape}")
                print(f"Matrix shape: {matrix.shape}")
                print("-" * 50)
                lora_grad_avg = torch.mean(lora_grad, dim=1)
                target = matrix[:, -1, :, :]
                input_x = torch.mean(input_x, dim=-2)


                # forward
                begin_time = time.time()
                output = self.model(input_x, base_parm, lora_parm, lora_grad_avg)
                print("infer:", time.time() - begin_time)

                loss = self.criterion(output, target.view(target.size(0), -1))
                running_loss += loss.item()

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f'Epoch [{epoch}/{self.num_epochs}], train_step:{train_step}, Loss: {loss.item():.4f}')
                train_step += 1

            avg_loss = running_loss / len(self.train_loader)
            print(f'Epoch [{epoch}/{self.num_epochs}], Average Loss: {avg_loss:.4f}')
            
            # save model
            epoch_model_path = os.path.join(self.save_path, f'model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
            }, epoch_model_path)


if __name__ == "__main__":
    dataset_path = './custom_data'
    dataset_name = 'dataset.h5' #'dataset.h5' lora_dataset_batch_4_model.h5
    dataset_file_name = os.path.join(dataset_path, dataset_name)
    predict_dataset = CustomDatasetHDF5(dataset_file_name)

    train_batch_size = 4

    predict_data_loader = DataLoader(
        predict_dataset, batch_size=train_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=True,
    )

    input_x, base_parm, lora_parm, lora_grad, matrix, epoch = next(iter(predict_data_loader))
    k = input_x.shape[1]
    input_x_batch = input_x.shape[2]
    seq_length = input_x.shape[3]
    embedding_dim = input_x.shape[4]
    layer_num = lora_grad.shape[-2]
    head_num = lora_grad.shape[-1]
    model = PredictNetwork(k, input_x_batch, seq_length, embedding_dim, layer_num, head_num)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = int(1e2)

    trainer = Trainer(model, predict_data_loader, criterion, optimizer, num_epochs)
    trainer.train()
    