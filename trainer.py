import torch

from datetime import datetime
from torch.utils.data import DataLoader, random_split
from loss import Loss


class Trainer:
    """
    Class to train the model.
    Built using instructions available @ https://pytorch.org/tutorials/beginner/introyt/trainingyt.html.
    """
    def __init__(self, model, optimizer, dataset):
        self.model = model
        self.optimizer = optimizer
        self.device = model.device
        self.loss_function = Loss().to(self.device)

        training_length = int(len(dataset) * 0.8)
        training_dataset, validation_dataset = random_split(dataset, [training_length, len(dataset) - training_length])

        print('traininging set has {} instances'.format(len(training_dataset)))
        print('validationidation set has {} instances'.format(len(validation_dataset)))

        self.training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
        self.validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    def train_step(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(self.training_dataloader):
            inputs, labels = data  # TODO: update this to fit the dataset.

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            # dual_attention((video_content_inputs, qos_features))

            loss = self.loss_function(outputs, labels)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100 # loss per batch
                print('batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.training_dataloader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def val_step(self):
        self.model.eval()
        with torch.no_grad():
            for x, y in self.val_loader:
                y_hat = self.model(x)
                loss = self.loss_fn(y, y_hat)
        return loss.item()

    def train(self, n_epochs):  # TODO: update the whole training and evaluation.
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0

        EPOCHS = 5

        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss = train_one_epoch(epoch_number, writer)


            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(validation_loader):
                    vinputs, vlabels = vdata
                    voutputs = model(vinputs)
                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)

            epoch_number += 1

        ###############
        for epoch in range(n_epochs):
            for x, y in self.train_loader:
                loss = self.train_step(x, y)
            val_loss = self.val_step()
            print(f"Epoch: {epoch}, Loss: {loss}, Val Loss: {val_loss}")
        print("Training is done")

        ################### 
        inputs = next(iter(self.training_dataloader))

        video_content_inputs = []

        for v in inputs['video_content']:
            video_content_inputs.append([
                [b[0].to(self.device), b[1].to(self.device)] if type(b) == list else b.to(self.device) for b in v
            ])

        qos_features = inputs['qos'].to(self.device)

        overall_prediction = inputs['overall_QoE'].to(self.device)
        continuous_prediction = inputs['continuous_QoE'].to(self.device)
        ######################