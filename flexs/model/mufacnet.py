import torch
import torch.nn as nn
import numpy as np
from . import torch_model, register_model
from utils.seq_utils import sequences_to_mutation_sets

class MuFacNet(nn.Module):
    """
        Mutation Factorization Network (MuFacNet)
    """
    
    def __init__(self, input_dim, latent_dim=32, num_filters=32, hidden_dim=128, kernel_size=5):
        super().__init__()
        self.mutation_context_encoder = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, kernel_size),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(1),
            nn.Linear(num_filters, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.joint_effect_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, mutation_sets, mutation_sets_mask):
        # Input:  - mutation_sets:      [batch_size, max_mutation_num, input_dim, context_width]
        #         - mutation_sets_mask: [batch_size, max_mutation_num]
        # Output: - predictions:        [batch_size, 1]
        
        batch_size, max_mutation_num, input_dim, context_width = list(mutation_sets.size())
        element_embeddings = self.mutation_context_encoder(mutation_sets.view(batch_size*max_mutation_num, input_dim, context_width))
        element_embeddings = element_embeddings.view(batch_size, max_mutation_num, -1) * torch.unsqueeze(mutation_sets_mask, dim=-1)
        set_embeddings = torch.sum(element_embeddings, dim=1)
        predictions = self.joint_effect_decoder(set_embeddings)
        return predictions

@register_model("mufacnet")
class MutationFactorizationModel(torch_model.TorchModel):
    def __init__(self, args, alphabet, starting_sequence, **kwargs):
        super().__init__(
            args, alphabet,
            net = MuFacNet(
                input_dim = len(alphabet),
                latent_dim = args.latent_dim 
            )
        )
        self.wt_sequence = starting_sequence
        self.context_radius = args.context_radius
        self.latent_dim = args.latent_dim
        self.lr = args.lr
        self.alphabet = alphabet

    def get_data_loader(self, sequences, labels, rank_coefficient=0.0):
        # Input:  - sequences:    [dataset_size, sequence_length]
        #         - labels:       [dataset_size]
        # Output: - loader_train: torch.utils.data.DataLoader
        
        mutation_sets, mutation_sets_mask = sequences_to_mutation_sets(sequences, self.alphabet, self.wt_sequence, self.context_radius)
        labels = torch.from_numpy(labels).float()
        dataset_train = torch.utils.data.TensorDataset(mutation_sets, mutation_sets_mask, labels)
        if rank_coefficient > 0.:
            ranks = torch.argsort(torch.argsort(-1 * labels))
            weights = 1.0 / (rank_coefficient * len(labels) + ranks)
            sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=self.args.batch_size, sampler=sampler)
        else:
            loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=self.args.batch_size, shuffle=True)
        return loader_train

    def compute_loss(self, data):
        # Input:  - mutation_sets:      [batch_size, max_mutation_num, alphabet_size, context_width]
        #         - mutation_sets_mask: [batch_size, max_mutation_num]
        #         - labels:             [batch_size]
        # Output: - loss:               [1]
        
        mutation_sets, mutation_sets_mask, labels = data
        outputs = torch.squeeze(self.net(mutation_sets.to(self.device), mutation_sets_mask.to(self.device)), dim=-1)
        loss = self.loss_func(outputs, labels.to(self.device))
        return loss

    def get_fitness(self, sequences):
        # Input:  - sequences:   [batch_size, sequence_length]
        # Output: - predictions: [batch_size]
        
        self.net.eval()
        with torch.no_grad():
            mutation_sets, mutation_sets_mask = sequences_to_mutation_sets(sequences, self.alphabet, self.wt_sequence, self.context_radius)
            predictions = self.net(mutation_sets.to(self.device), mutation_sets_mask.to(self.device)).cpu().numpy()
        predictions = np.squeeze(predictions, axis=-1)
        return predictions
    
    def train_prioritized(self, sequences, labels, rank_coefficient=0.01):
        # Input: - sequences: [dataset_size, sequence_length]
        #        - labels:    [dataset_size]
            
        self.net.train()
        loader_train = self.get_data_loader(sequences, labels, rank_coefficient=rank_coefficient)
        best_loss, num_no_improvement = np.inf, 0
        while num_no_improvement < self.args.patience:
            loss_List = []
            for data in loader_train:
                loss = self.compute_loss(data)
                loss_List.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            current_loss = np.mean(loss_List)
            if current_loss < best_loss:
                best_loss = current_loss
                num_no_improvement = 0
            else:
                num_no_improvement += 1
