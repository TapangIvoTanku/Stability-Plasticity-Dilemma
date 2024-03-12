import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from copy import deepcopy

class EWC:
    def __init__(self, m: nn.Module, dataset: list):

        self.model = m
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        # print("starting diag fisher")
        self._precision_matrices = self._diag_fisher()

        # print("Deepcoping")
        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.clone().detach().requires_grad_(True)

    def _diag_fisher(self):
        precision_matrices = {}
        # print("inside diag_fisher deepcopy")
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.clone().detach().requires_grad_(True)

        self.model.eval()
        for src in tqdm(self.dataset, total=len(self.dataset)):
            # print("performing inference on each sample")
            self.model.zero_grad()
            input_ids = tokenizer(src, return_tensors="pt", padding=True).input_ids.to(device)
            decoder_input_ids = torch.tensor(0, device=device).reshape(1,1)
            while decoder_input_ids[:, -1] != 1:
                output = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits
                decoder_input_ids = torch.cat((decoder_input_ids, torch.argmax(output[:, -1, :].unsqueeze(1), dim=2)), dim=1)

            loss = F.nll_loss(F.log_softmax(output, dim=2).squeeze(), decoder_input_ids.squeeze()[1:])
            # print("Computed NLL_LOSS")
            loss.backward()
            del input_ids
            del output

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss