import torch


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        y_hat = self.model(x)
        loss = self.loss_fn(y, y_hat)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self):
        self.model.eval()
        with torch.no_grad():
            for x, y in self.val_loader:
                y_hat = self.model(x)
                loss = self.loss_fn(y, y_hat)
        return loss.item()

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            for x, y in self.train_loader:
                loss = self.train_step(x, y)
            val_loss = self.val_step()
            print(f"Epoch: {epoch}, Loss: {loss}, Val Loss: {val_loss}")
        print("Training is done")