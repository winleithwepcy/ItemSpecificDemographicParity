# model/fair_biasedmf.py
import torch
import torch.nn as nn
from torch.nn.init import normal_


# ------------------------------------------------
# Sensitive Attribute Expansion Network
# ------------------------------------------------
class SensitiveAttributeNet(nn.Module):
    """
    Input: item embedding
    Output: expanded fairness vector of size (2k + h)
    """
    def __init__(self, item_dim, hidden_dim, expanded_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(item_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, expanded_dim)
        )

    def forward(self, x):
        return self.net(x)  # [batch, expanded_dim]
        

# ------------------------------------------------
# FairBiasedMF (NEW VERSION)
# ------------------------------------------------
class FairBiasedMF(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        latent_dim,
        num_sensitive_attrs,     # = k
        extra_dim=16,            # = h
        sensitive_hidden_dim=128,
        dropout_rate=0.2,
        fair_lambda=0.0
    ):
        super().__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))

        # Expanded fairness dimension = 2k + h
        self.k = num_sensitive_attrs 
        self.expanded_dim = 2 * num_sensitive_attrs + extra_dim

        # Sensitive attribute expansion network
        self.sensitive_net = SensitiveAttributeNet(
            item_dim=latent_dim,
            hidden_dim=sensitive_hidden_dim,
            expanded_dim=self.expanded_dim
        )

        # Final MLP (maps combined vector → scalar)
        self.final_mlp = nn.Sequential(
            nn.Linear(latent_dim + self.expanded_dim, sensitive_hidden_dim),
            nn.ReLU(),
            nn.Linear(sensitive_hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()


    def _init_weights(self):
        normal_(self.user_embedding.weight, std=0.01)
        normal_(self.item_embedding.weight, std=0.01)
        normal_(self.user_bias.weight, std=0.01)
        normal_(self.item_bias.weight, std=0.01)


    def forward(self, user, item, item_cat_vector):
        """
        item_cat_vector: [batch, k]
        """

        u = user.view(-1)
        i = item.view(-1)

        # ---------------------- Embeddings ----------------------
        u_e = self.dropout(self.user_embedding(u))   # [batch, latent_dim]
        i_e = self.dropout(self.item_embedding(i))   # [batch, latent_dim]

        # ---------------------- Base PMF Vector ----------------------
        base_vec = u_e * i_e                         # [batch, latent_dim]

        # ---------------------- Fairness Expansion ----------------------
        w_s_expanded = self.sensitive_net(i_e)       # [batch, 2k + h]

        #batch, k = item_cat_vector.shape
        batch = item_cat_vector.size(0)

        expanded_indicator = torch.cat([
            item_cat_vector,                        # k                      
            torch.zeros(batch, self.expanded_dim - item_cat_vector.size(1),
                        device=item_cat_vector.device)
        ], dim=1)

        fairness_vec = w_s_expanded * expanded_indicator  # [batch, 2k + h]

        # ---------------------- Combine vectors ----------------------
        z = torch.cat([base_vec, fairness_vec], dim=1)     # [batch, latent_dim + 2k + h]

        # ---------------------- Final score via MLP ----------------------
        mlp_score = self.final_mlp(z).view(-1)             # [batch]

        # Add biases
        score = (
            self.global_bias +
            self.user_bias(u).view(-1) +
            self.item_bias(i).view(-1) +
            mlp_score
        )

        return self.sigmoid(score)
