import torch
import torch.nn as nn
from torch.nn.init import normal_

class SensitiveAttributeNet(nn.Module):
    """MLP to learn expanded sensitive weights from item embedding."""
    def __init__(self, item_dim, hidden_dim, expanded_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(item_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, expanded_dim)  # output is 2k + h
        )

    def forward(self, item_emb):
        return self.net(item_emb)


class FairPMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim,
                 num_sensitive_attrs,
                 extra_dim=16,     # = h
                 hidden_dim=128,
                 fair_lambda=0.0):
        super().__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

        # Expanded fairness dimension = 2k + h
        self.k = num_sensitive_attrs
        self.expanded_dim = 2 * num_sensitive_attrs + extra_dim

        # Sensitive attribute expansion network
        self.sensitive_net = SensitiveAttributeNet(
            item_dim=latent_dim,
            hidden_dim=hidden_dim,
            expanded_dim=self.expanded_dim
        )

        # Final MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(latent_dim + self.expanded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        normal_(self.user_embedding.weight, std=0.01)
        normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user, item, item_cat_vector):
        """
        item_cat_vector: [batch, k]
        """
        batch_size = item_cat_vector.size(0)
        # ---------------------- 1. Base Vector Score ----------------------
        user_e = self.user_embedding(user)  # [batch, latent_dim]
        item_e = self.item_embedding(item)  # [batch, latent_dim]

        # Elementwise product produces a vector
        base_vec = user_e * item_e         # [batch, latent_dim]

        # ---------------------- 2. Fairness Vector ------------------------
        w_s_expanded = self.sensitive_net(item_e)  # [batch, 2k+h]

        # expand indicator for multiplication
        expanded_indicator = torch.cat([
            item_cat_vector,  # k
            torch.zeros(batch_size, self.expanded_dim - item_cat_vector.size(1),
                        device=item_cat_vector.device)
        ], dim=1)

        fairness_vec = w_s_expanded * expanded_indicator  # [batch, 2k+h]

        # ---------------------- 3. Concat base + fairness -----------------
        z = torch.cat([base_vec, fairness_vec], dim=1)  # [batch, latent_dim + 2k + h]

        # ---------------------- 4. Final Prediction -----------------------
        out = self.final_mlp(z)  # scalar
        return self.sigmoid(out).squeeze(1)
