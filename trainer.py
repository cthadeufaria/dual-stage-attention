import torch
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from loss import Loss
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Class to train the model.
    Built using instructions available @ https://pytorch.org/tutorials/beginner/introyt/trainingyt.html.
    """
    def __init__(self, model, optimizer, dataset, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.device = model.device
        self.loss_function = loss_function

        training_length = int(len(dataset) * 0.8)
        training_dataset, validation_dataset = random_split(dataset, [training_length, len(dataset) - training_length])

        print('Training and Validation datasets created.')
        print('Training set has {} instances.'.format(len(training_dataset)))
        print('Validation set has {} instances.'.format(len(validation_dataset)))

        self.training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
        self.validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    def get(self, data):
            video_content_inputs = []
            for v in data['video_content']:
                video_content_inputs.append([
                    [b[0].to(self.device), b[1].to(self.device)] if type(b) == list else b.to(self.device) for b in v
                ])

            qos_features = data['qos'].to(self.device)

            overall_prediction = data['overall_QoE'].to(self.device)
            continuous_prediction = torch.stack(data['continuous_QoE']).to(self.device)

            inputs = (video_content_inputs, qos_features)
            labels = (overall_prediction, continuous_prediction)

            return inputs, labels
    
    def train_step(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(self.training_dataloader):
            inputs, labels = self.get(data)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss_function(outputs, labels)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()  # TODO: understand this structure below.
            if i % 100 == 99:
                last_loss = running_loss / 100 # loss per batch
                print('batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.training_dataloader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def train(self, EPOCHS):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/DUAL_ATTENTION_LIVENFLX_II_{}'.format(timestamp))
        epoch_number = 0

        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_train_loss = self.train_step(epoch_number, writer)

            running_vloss = 0.0

            self.model.eval()

            with torch.no_grad():
                for i, vdata in enumerate(self.validation_dataloader):
                    vinputs, vlabels = self.get(vdata)
                    voutputs = self.model(vinputs)
                    vloss = self.loss_function(voutputs, vlabels)
                    running_vloss += vloss

            avg_val_loss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_train_loss, avg_val_loss))

            # Log
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_train_loss, 'Validation' : avg_val_loss },
                            epoch_number + 1)
            writer.flush()

            if avg_val_loss < best_vloss:
                best_vloss = avg_val_loss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

            print(torch.cuda.memory_summary())  # Detailed breakdown
            print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    

        ###############
        # for epoch in range(n_epochs):
        #     for x, y in self.train_loader:
        #         loss = self.train_step(x, y)
        #     val_loss = self.val_step()
        #     print(f"Epoch: {epoch}, Loss: {loss}, Val Loss: {val_loss}")
        # print("Training is done")

    def val_step(self):
        self.model.eval()

        # for module in dual_attention.modules.values():
        #     module.eval()

        with torch.no_grad():

            for x, y in self.val_loader:
                y_hat = self.model(x)
                loss = self.loss_fn(y, y_hat)

        return loss.item()