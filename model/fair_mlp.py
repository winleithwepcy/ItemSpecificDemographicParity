import torch
import torch.nn as nn
from torch.nn.init import normal_


# ------------------------------------------------
# Sensitive Attribute Expansion Network (same as in FairBiasedMF & FairNeuMF)
# ------------------------------------------------
class SensitiveAttributeNet(nn.Module):
    """
    Input : item embedding
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
        return self.net(x)  # [batch, 2k + h]


# ------------------------------------------------
# FairMLP – MLP-based recommender with combinational fairness (new version)
# ------------------------------------------------
class FairMLP(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        layers=[64, 32, 16, 8],
        num_sensitive_attrs=3,      # k
        extra_dim=16,               # h → extra expressive capacity
        sensitive_hidden_dim=128,
        dropout_rate=0.2,
        fair_lambda=0.0
    ):
        super().__init__()

        self.embed_dim = layers[0] // 2
        self.layers = layers
        self.k = num_sensitive_attrs
        self.expanded_dim = 2 * num_sensitive_attrs + extra_dim  # 2k + h

        # ---------------------- User & Item Embeddings ----------------------
        self.user_embedding = nn.Embedding(num_users, self.embed_dim)
        self.item_embedding = nn.Embedding(num_items, self.embed_dim)

        # ---------------------- Base MLP Path ----------------------
        mlp_modules = []
        for i in range(len(layers) - 1):
            mlp_modules.append(nn.Linear(layers[i], layers[i + 1]))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout_rate))
        mlp_modules.append(nn.Linear(layers[-1], 1))  # base score head
        self.base_mlp = nn.Sequential(*mlp_modules)

        # ---------------------- Fairness Expansion Network ----------------------
        self.sensitive_net = SensitiveAttributeNet(
            item_dim=self.embed_dim,
            hidden_dim=sensitive_hidden_dim,
            expanded_dim=self.expanded_dim
        )

        # ---------------------- Final Fusion MLP ----------------------
        final_input_dim = layers[-1] + self.expanded_dim  # base MLP output + fairness vector
        self.final_mlp = nn.Sequential(
            nn.Linear(final_input_dim, sensitive_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(sensitive_hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.fair_lambda = float(fair_lambda)

        self._init_weights()

    def _init_weights(self):
        normal_(self.user_embedding.weight, std=0.01)
        normal_(self.item_embedding.weight, std=0.01)
        for layer in self.base_mlp:
            if isinstance(layer, nn.Linear):
                normal_(layer.weight, std=0.01)
        for layer in self.final_mlp:
            if isinstance(layer, nn.Linear):
                normal_(layer.weight, std=0.01)

    def forward(self, user, item, item_cat_vector):
        """
        Args:
            user: LongTensor [batch]
            item: LongTensor [batch]
            item_cat_vector: FloatTensor [batch, k]  – one-hot/multi-hot sensitive categories
        """
        user = user.view(-1)
        item = item.view(-1)

        # ---------------------- Embeddings ----------------------
        u_emb = self.dropout(self.user_embedding(user))   # [batch, embed_dim]
        i_emb = self.dropout(self.item_embedding(item))   # [batch, embed_dim]

        # ---------------------- Base MLP Path ----------------------
        mlp_input = torch.cat([u_emb, i_emb], dim=-1)       # [batch, layers[0]]
        base_features = self.base_mlp[:-1](mlp_input)       # up to last hidden layer
        base_score = self.base_mlp[-1](base_features).view(-1)  # [batch]

        # ---------------------- Combinational Fairness Expansion ----------------------
        w_s_expanded = self.sensitive_net(i_emb)             # [batch, 2k + h]

        batch_size = item_cat_vector.size(0)
        expanded_indicator = torch.cat([
            item_cat_vector,                                 # k                                # another k → 2k
            torch.zeros(batch_size, self.expanded_dim - item_cat_vector.size(1),
                        device=item_cat_vector.device)       # h zeros
        ], dim=1)                                             # [batch, 2k + h]

        fairness_vec = w_s_expanded * expanded_indicator     # [batch, 2k + h]

        # ---------------------- Final Fusion ----------------------
        combined = torch.cat([base_features, fairness_vec], dim=-1)  # [batch, last_hidden + 2k + h]
        fairness_adjustment = self.final_mlp(combined).view(-1)       # [batch]

        # ---------------------- Final Score ----------------------
        #total_score = base_score + fairness_adjustment
        total_score = fairness_adjustment

        return self.sigmoid(total_score)