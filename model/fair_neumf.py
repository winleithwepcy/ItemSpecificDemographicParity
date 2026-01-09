import torch
import torch.nn as nn
from torch.nn.init import normal_


# ------------------------------------------------
# Sensitive Attribute Expansion Network (same as in FairBiasedMF)
# ------------------------------------------------
class SensitiveAttributeNet(nn.Module):
    """
    Input : item embedding (from GMF path)
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
# FairNeuMF – NeuMF with combinational sensitive attributes
# ------------------------------------------------
class FairNeuMF(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        gmf_dim=8,
        mlp_layers=[64, 32, 16, 8],
        num_sensitive_attrs=3,      # k
        extra_dim=16,               # h
        sensitive_hidden_dim=128,
        dropout_rate=0.2,
        fair_lambda=0.0
    ):
        super().__init__()

        self.gmf_dim = gmf_dim
        self.mlp_layers = mlp_layers
        self.k = num_sensitive_attrs
        self.expanded_dim = 2 * num_sensitive_attrs + extra_dim  # 2k + h

        # ---------------------- GMF Part ----------------------
        self.gmf_user_embedding = nn.Embedding(num_users, gmf_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, gmf_dim)

        # ---------------------- MLP Part ----------------------
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_layers[0] // 2)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_layers[0] // 2)

        mlp_modules = []
        for i in range(len(mlp_layers) - 1):
            mlp_modules.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout_rate))
        self.mlp = nn.Sequential(*mlp_modules)

        # ---------------------- NeuMF Fusion ----------------------
        # Original NeuMF concatenates GMF vector + last MLP layer → 1
        self.neumf_fusion = nn.Linear(gmf_dim + mlp_layers[-1], 1)

        # ---------------------- Fairness Expansion ----------------------
        self.sensitive_net = SensitiveAttributeNet(
            item_dim=gmf_dim,                     # using GMF item embedding as input (same as original FairNeuMF)
            hidden_dim=sensitive_hidden_dim,
            expanded_dim=self.expanded_dim
        )

        # Final MLP that combines NeuMF base vector with fairness vector
        final_input_dim = gmf_dim + mlp_layers[-1] + self.expanded_dim
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
        normal_(self.gmf_user_embedding.weight, std=0.01)
        normal_(self.gmf_item_embedding.weight, std=0.01)
        normal_(self.mlp_user_embedding.weight, std=0.01)
        normal_(self.mlp_item_embedding.weight, std=0.01)
        for layer in self.neumf_fusion.modules():
            if isinstance(layer, nn.Linear):
                normal_(layer.weight, std=0.01)
        for layer in self.final_mlp.modules():
            if isinstance(layer, nn.Linear):
                normal_(layer.weight, std=0.01)

    def forward(self, user, item, item_cat_vector):
        """
        Args:
            user: LongTensor [batch]
            item: LongTensor [batch]
            item_cat_vector: FloatTensor [batch, k]  – one-hot or multi-hot sensitive categories
        """
        user = user.view(-1)
        item = item.view(-1)

        # ---------------------- GMF Path ----------------------
        gmf_u = self.dropout(self.gmf_user_embedding(user))      # [batch, gmf_dim]
        gmf_i = self.dropout(self.gmf_item_embedding(item))      # [batch, gmf_dim]
        gmf_vec = gmf_u * gmf_i                                   # [batch, gmf_dim]

        # ---------------------- MLP Path ----------------------
        mlp_u = self.dropout(self.mlp_user_embedding(user))
        mlp_i = self.dropout(self.mlp_item_embedding(item))
        mlp_input = torch.cat([mlp_u, mlp_i], dim=-1)              # [batch, mlp_layers[0]]
        mlp_vec = self.mlp(mlp_input)                             # [batch, mlp_layers[-1]]

        # ---------------------- Base NeuMF vector ----------------------
        base_neumf_vec = torch.cat([gmf_vec, mlp_vec], dim=-1)    # [batch, gmf_dim + last_mlp]

        # ---------------------- Fairness Expansion (combinational) ----------------------
        # Predict expanded weight vector w_s ∈ ℝ^{2k + h} from item GMF embedding
        w_s_expanded = self.sensitive_net(gmf_i)                  # [batch, 2k + h]

        batch_size = item_cat_vector.size(0)
        # Build expanded indicator: [one-hot, one-hot, zeros_h]
        expanded_indicator = torch.cat([
            item_cat_vector,                                      # k dims  # another k → total 2k
            torch.zeros(batch_size, self.expanded_dim - item_cat_vector.size(1),
                        device=item_cat_vector.device)
        ], dim=1)                                                  # [batch, 2k + h]

        fairness_vec = w_s_expanded * expanded_indicator          # [batch, 2k + h]

        # ---------------------- Final combination ----------------------
        combined_vec = torch.cat([base_neumf_vec, fairness_vec], dim=-1)
        # Use the same style final MLP as in FairBiasedMF
        adjustment = self.final_mlp(combined_vec).view(-1)        # [batch]

        # Original NeuMF prediction (optional – you can keep or remove it)
        base_score = self.neumf_fusion(base_neumf_vec).view(-1)

        # Final prediction = base NeuMF + fairness adjustment
        #score = base_score + adjustment
        score = adjustment

        return self.sigmoid(score)