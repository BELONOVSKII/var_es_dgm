import torch
from torch import nn
from tqdm.auto import tqdm

from .epsilon_theta import EpsilonTheta


class TimeGrad(nn.Module):
    def __init__(
        self,
        target_dim,
        input_size,
        scheduler,
        num_layers=2,
        hidden_size=5,
        dropout_rate=0.8,
        num_inference_steps=100,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.unet = EpsilonTheta(target_dim=target_dim, cond_dim=hidden_size)
        self.criterion = nn.MSELoss()
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.target_dim = target_dim

    def forward(self, x, prediction_length=1, num_parallel_samples=1):
        batch_size = x.shape[0]

        _, (h, c) = self.rnn(x)

        next_sample = self.sample(h[-1]).reshape(batch_size, 1, -1)
        future_samples = next_sample

        if prediction_length > 1:
            for _ in tqdm(range(1, prediction_length), desc="prediction step"):
                _, (h, c) = self.rnn(next_sample, (h, c))
                next_sample = self.sample(h[-1]).reshape(batch_size, 1, -1)
                future_samples = torch.cat((future_samples, next_sample), dim=1)
        return future_samples

    def sample(self, context):
        """Algorithm 2 from paper"""

        # initialize random noise
        x = torch.randn((context.shape[0], self.target_dim), device=context.device)

        self.scheduler.set_timesteps(self.num_inference_steps)

        # denoise x step-by-step
        for t in self.scheduler.timesteps:
            predicted_noise = self.unet(
                x.reshape(x.shape[0], 1, -1),
                t,
                context.reshape(context.shape[0], 1, -1),
            )
            predicted_noise = predicted_noise.reshape(predicted_noise.shape[0], -1)
            x = self.scheduler.step(predicted_noise, t, x).prev_sample

        # return x^0
        return x

    def fit(self, train_loader, optimizer, n_epochs=50, device="cpu", verbose=True):
        losses = list()
        self.train()

        # iterate over all epochs
        with tqdm(range(n_epochs), desc="Epochs", disable=not verbose) as t:
            for _ in t:
                total_loss = 0
                for train, target in train_loader:

                    # compute loss
                    loss = self.loss(train.to(device), target.to(device))

                    # optimize weights
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.detach().cpu().item()
                t.set_postfix(total_loss=(total_loss / len(train_loader)))
                losses.append(total_loss / len(train_loader))
        return losses

    def loss(self, x_context, x_prediction, verbose=False):
        """Algorithm 1 from paper"""

        batch_size = x_context.shape[0]

        # obtain h_{t-1} from context
        _, (h, _) = self.rnn(x_context)
        h_t_minus_1 = h[-1]

        # sample n
        n = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=x_context.device,
        )

        # noise
        noise = torch.randn(x_prediction.shape, device=x_prediction.device)
        # noisy output for unet
        x_noisy = self.scheduler.add_noise(x_prediction, noise, n)

        # model's output
        model_output = self.unet(
            x_noisy.reshape(batch_size, 1, -1),
            n,
            h_t_minus_1.reshape(batch_size, 1, -1),
        )
        if verbose:
            print(x_noisy)
            print(noise)
            print(model_output)
        # criterion
        return self.criterion(model_output, noise)
