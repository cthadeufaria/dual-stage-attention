import torch
from datetime import datetime
from torch.utils.data import DataLoader, random_split
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

        print('Training and Validation datasets created')
        print('Training set has {} instances'.format(len(training_dataset)))
        print('Validation set has {} instances'.format(len(validation_dataset)))

        self.training_dataloader = DataLoader(
            training_dataset, 
            batch_size=4, 
            shuffle=True, 
            collate_fn=self.collate_function
        )
        self.validation_dataloader = DataLoader(
            validation_dataset, 
            batch_size=4, 
            shuffle=False, 
            collate_fn=self.collate_function
        )

    def collate_function(self, batch: list) -> list:
        return [self.get(data) for data in batch]

    def get(self, data):
        video_content_inputs = data['video_content']

        qos_features = data['qos'].to(self.device)

        overall_labels = data['overall_QoE'].to(self.device)
        continuous_labels = data['continuous_QoE'].to(self.device)

        inputs = [video_content_inputs, qos_features]
        labels = [overall_labels, continuous_labels]

        return inputs, labels

    def train_step(self, epoch, tb_writer):
        self.model.train(True)
        running_loss = 0.

        for i, data in enumerate(self.training_dataloader):
            print('Processing training for batch', (i + 1))

            inputs, labels = [
                value[0] for value in data
            ], [
                value[1] for value in data
            ]

            self.optimizer.zero_grad()

            outputs = []
            for input in inputs:  # TODO: How to pass batched inputs to the model? Check model input/output shapes.
                outputs.append(self.model(input))

            loss = self.loss_function(outputs, labels)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            print('Training batch {} average loss: {} --------------------'.format(i + 1, avg_loss))

            tb_x = epoch * len(self.training_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', avg_loss, tb_x)

        return avg_loss

    def val_step(self, epoch, tb_writer):
        self.model.eval()
        running_loss = 0.

        with torch.no_grad():
            for i, data in enumerate(self.validation_dataloader):
                print('Processing validation for batch', (i + 1))

                inputs, labels = [
                    value[0] for value in data
                ], [
                    value[1] for value in data
                ]

                outputs = []
                for input in inputs:  # TODO: How to pass batched inputs to the model? Check model input/output shapes.
                    outputs.append(self.model(input))

                loss = self.loss_function(outputs, labels)
                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
                print('Validation batch {} average loss: {} --------------------'.format(i + 1, avg_loss))

        tb_x = epoch * len(self.validation_dataloader) + i + 1
        tb_writer.add_scalar('Loss/val', avg_loss, tb_x)

        return avg_loss

    def train_and_validate(self, EPOCHS):
        print('Starting training...')

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        writer = SummaryWriter('runs/summaries/DUAL_ATTENTION_LIVENFLX_II_SUMMARY_{}'.format(timestamp))
        best_loss = float('inf')

        print('Timestamp:', timestamp)
        print('Total epochs:', EPOCHS)

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch + 1))

            avg_train_loss = self.train_step(epoch + 1, writer)
            avg_val_loss = self.val_step(epoch + 1, writer)

            # Log
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_train_loss, 'Validation' : avg_val_loss },
                            epoch + 1)
            writer.flush()

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                model_path = 'runs/models/DUAL_ATTENTION_LIVENFLX_II_{}_EPOCH_{}'.format(timestamp, epoch)
                torch.save(self.model.state_dict(), model_path)
                print('Model saved to', model_path)

        print('Training and validation finished')
        delta = datetime.now() - timestamp
        print(f"Elapsed time: {delta}")
        writer.close()