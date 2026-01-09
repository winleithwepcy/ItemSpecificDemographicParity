import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
#from model.fair_pmf import FairPMF

def evaluate_model(model, test_ratings, test_negatives, top_k, device, dataset=None):
    """
    dataset: pass Dataset instance to get user attributes and item-sensitive vectors
    """
    hits, ndcgs = [], [] 
    model.eval()

    for idx, (rating, negatives) in tqdm(
        enumerate(zip(test_ratings, test_negatives)),
        total=len(test_ratings),
        desc="Evaluating",
        unit="user",
        leave=False,
    ):
        user = rating[0]
        item = rating[1]
        items = negatives + [item]
        users = [user] * len(items)

        # Convert to tensors
        user_tensor = torch.LongTensor(users).to(device)
        item_tensor = torch.LongTensor(items).to(device)

        # --- FairPMF requires extra attributes ---
        # user attributes
        user_age = torch.FloatTensor([dataset.user_attributes[u][0] for u in users]).to(device)
        user_gender = torch.FloatTensor([dataset.user_attributes[u][1] for u in users]).to(device)
        # item-sensitive vectors
        item_cat_vector = torch.FloatTensor(dataset.item_sensitive_vectors[items]).to(device)
        pred_scores = model(user_tensor, item_tensor, item_cat_vector)
        scores = pred_scores.squeeze()

        # Safety checks
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        top_k_effective = min(top_k, scores.size(0))
        _, indices = torch.topk(scores, top_k_effective)
        top_items = [items[i] for i in indices.cpu().numpy()]  # flat list

        hits.append(hit_at_k(top_items, item))
        ndcgs.append(ndcg_at_k(top_items, item))

    return np.mean(hits), np.mean(ndcgs)

def evaluate_ISDP(model, test_ratings, test_negatives, user_attributes, movie_categories, dataset, device):
    """
    Evaluate Item-Specific Demographic Parity (ISDP) fairness metric.
    Works with FairPMF.
    """
    model.eval()
    item_group_preds_cat1 = defaultdict(lambda: defaultdict(list))
    item_group_preds_cat0 = defaultdict(lambda: defaultdict(list))
    group_scores_cat1 = defaultdict(list)
    group_scores_cat0 = defaultdict(list)

    with torch.no_grad():
        for (user, item), negatives in zip(test_ratings, test_negatives):
            if user not in user_attributes or item not in movie_categories:
                continue

            user_age, user_gender = user_attributes[user]
            category = movie_categories[item]
            group_cat1 = (user_age, user_gender)
            group_cat0 = user_gender

            items = negatives + [item]
            users = [user] * len(items)

            # Convert to tensors
            user_tensor = torch.LongTensor(users).to(device)
            item_tensor = torch.LongTensor(items).to(device)

            # user attributes
            user_age = torch.FloatTensor([dataset.user_attributes[u][0] for u in users]).to(device)
            user_gender = torch.FloatTensor([dataset.user_attributes[u][1] for u in users]).to(device)
            # item-sensitive vectors
            item_cat_vector = torch.FloatTensor(dataset.item_sensitive_vectors[items]).to(device)

            if isinstance(model, torch.nn.Module) and hasattr(model, 'forward'):
                pred_scores = model(user_tensor, item_tensor, item_cat_vector)
                scores = pred_scores.squeeze().cpu().numpy()
            else:
                scores = model(user_tensor, item_tensor).squeeze().cpu().numpy()

            true_item_score = scores[-1]  # last item is test item

            if category == 1:
                item_group_preds_cat1[item][group_cat1].append(true_item_score)
                group_scores_cat1[group_cat1].append(true_item_score)
            else:
                item_group_preds_cat0[item][group_cat0].append(true_item_score)
                group_scores_cat0[group_cat0].append(true_item_score)

    def compute_avg_max_diff(item_group_preds):
        max_diffs = []
        for item, group_preds in item_group_preds.items():
            group_means = {g: np.mean(v) for g, v in group_preds.items() if len(v) > 0}
            groups = list(group_means.keys())
            if len(groups) < 2:
                continue
            diffs = [abs(group_means[groups[i]] - group_means[groups[j]]) 
                     for i in range(len(groups)) for j in range(i+1, len(groups))]
            if diffs:
                max_diffs.append(max(diffs))
        return float(np.mean(max_diffs)) if max_diffs else 0.0, len(max_diffs)

    avg_max_diff_cat1, num_items_cat1 = compute_avg_max_diff(item_group_preds_cat1)
    avg_max_diff_cat0, num_items_cat0 = compute_avg_max_diff(item_group_preds_cat0)

    group_avg_scores_cat1 = {g: round(np.mean(v), 4) for g, v in group_scores_cat1.items() if len(v) > 0}
    group_avg_scores_cat0 = {g: round(np.mean(v), 4) for g, v in group_scores_cat0.items() if len(v) > 0}

    return avg_max_diff_cat1, avg_max_diff_cat0, group_avg_scores_cat1, group_avg_scores_cat0, num_items_cat1, num_items_cat0

def evaluate_value(model, test_ratings, test_negatives, user_attributes, dataset, device):
    model.eval()
    item_group_preds = defaultdict(lambda: defaultdict(list))
    item_group_trues = defaultdict(lambda: defaultdict(list))
    group_pred_scores = defaultdict(list)
    group_true_scores = defaultdict(list)

    with torch.no_grad():
        for (user, item), negatives in zip(test_ratings, test_negatives):
            if user not in user_attributes:
                continue
            user_age, user_gender = user_attributes[user]
            items = negatives + [item]
            users = [user] * len(items)

            # Convert to tensors
            user_tensor = torch.LongTensor(users).to(device)
            item_tensor = torch.LongTensor(items).to(device)

            # user attributes
            user_age = torch.FloatTensor([dataset.user_attributes[u][0] for u in users]).to(device)
            user_gender = torch.FloatTensor([dataset.user_attributes[u][1] for u in users]).to(device)
            # item-sensitive vectors
            item_cat_vector = torch.FloatTensor(dataset.item_sensitive_vectors[items]).to(device)

            if isinstance(model, torch.nn.Module) and hasattr(model, 'forward'):
                pred_scores  = model(user_tensor, item_tensor, item_cat_vector)
                scores = pred_scores.squeeze().cpu().numpy()
            else:
                scores = model(user_tensor, item_tensor).squeeze().cpu().numpy()

            true_pred = scores[-1]
            true_label = 1.0

            item_group_preds[item][user_age].append(true_pred)
            item_group_trues[item][user_age].append(true_label)
            group_pred_scores[user_age].append(true_pred)
            group_true_scores[user_age].append(true_label)

    value_diffs = []
    for item in item_group_preds.keys():
        group_means_pred = {g: np.mean(v) for g, v in item_group_preds[item].items() if len(v) > 0}
        group_means_true = {g: np.mean(v) for g, v in item_group_trues[item].items() if len(v) > 0}
        if len(group_means_pred) < 2:
            continue
        groups = list(group_means_pred.keys())
        diff_g = {g: group_means_pred[g] - group_means_true[g] for g in groups}
        pair_diffs = [abs(diff_g[g1] - diff_g[g2]) for i, g1 in enumerate(groups) for g2 in groups[i+1:]]
        if pair_diffs:
            value_diffs.append(max(pair_diffs))

    avg_value_unfairness = float(np.mean(value_diffs)) if value_diffs else 0.0
    return round(avg_value_unfairness, 4)

def evaluate_absolute(model, test_ratings, test_negatives, user_attributes, dataset, device):
    model.eval()
    item_group_abs = defaultdict(lambda: defaultdict(list))
    group_abs_errors = defaultdict(list)

    with torch.no_grad():
        for (user, item), negatives in zip(test_ratings, test_negatives):
            if user not in user_attributes:
                continue
            user_age, user_gender = user_attributes[user]
            items = negatives + [item]
            users = [user] * len(items)

            # Convert to tensors
            user_tensor = torch.LongTensor(users).to(device)
            item_tensor = torch.LongTensor(items).to(device)
            # user attributes
            user_age = torch.FloatTensor([dataset.user_attributes[u][0] for u in users]).to(device)
            user_gender = torch.FloatTensor([dataset.user_attributes[u][1] for u in users]).to(device)
            # item-sensitive vectors
            item_cat_vector = torch.FloatTensor(dataset.item_sensitive_vectors[items]).to(device)

            if isinstance(model, torch.nn.Module) and hasattr(model, 'forward'):
                pred_scores = model(user_tensor, item_tensor, item_cat_vector)
                scores = pred_scores.squeeze().cpu().numpy()
            else:
                scores = model(user_tensor, item_tensor).squeeze().cpu().numpy()

            true_pred = scores[-1]
            true_label = 1.0
            abs_err = (true_pred - true_label) ** 2
            item_group_abs[item][user_age].append(abs_err)
            group_abs_errors[user_age].append(abs_err)

    abs_diffs = []
    for item in item_group_abs.keys():
        group_means_abs = {g: np.mean(v) for g, v in item_group_abs[item].items() if len(v) > 0}
        if len(group_means_abs) < 2:
            continue
        groups = list(group_means_abs.keys())
        pair_diffs = [abs(group_means_abs[groups[i]] - group_means_abs[groups[j]])
                      for i in range(len(groups)) for j in range(i+1, len(groups))]
        if pair_diffs:
            abs_diffs.append(max(pair_diffs))

    avg_abs_unfairness = float(np.mean(abs_diffs)) if abs_diffs else 0.0
    return round(avg_abs_unfairness, 4)

def evaluate_absolute_gpwise(model,test_ratings,test_negatives,user_attributes,dataset,device):
    """
    Item-conditional absolute unfairness.

    For each item:
      - compute group-wise mean squared error
      - take maximum pairwise difference across sensitive groups
    Final score = average over items
    """

    model.eval()

    # item -> gender -> age -> [absolute errors]
    item_group_abs = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    with torch.no_grad():
        for (user, pos_item), negatives in zip(test_ratings, test_negatives):

            # Skip users without sensitive attributes
            if user not in user_attributes:
                continue

            # ---------- scalar attributes (for grouping) ----------
            age_scalar, gender_scalar = user_attributes[user]

            # ---------- model inputs ----------
            items = negatives + [pos_item]
            users = [user] * len(items)

            user_tensor = torch.LongTensor(users).to(device)
            item_tensor = torch.LongTensor(items).to(device)

            # user-sensitive tensors
            user_age_tensor = torch.FloatTensor(
                [dataset.user_attributes[u][0] for u in users]
            ).to(device)

            user_gender_tensor = torch.FloatTensor(
                [dataset.user_attributes[u][1] for u in users]
            ).to(device)

            # item-sensitive vectors
            item_cat_vector = torch.FloatTensor(
                dataset.item_sensitive_vectors[items]
            ).to(device)

            # ---------- prediction ----------
            if hasattr(model, "forward"):
                preds = model(
                    user_tensor,
                    item_tensor,
                    item_cat_vector
                )
            else:
                preds = model(user_tensor, item_tensor)

            scores = preds.squeeze().cpu().numpy()

            # ---------- absolute error (positive item only) ----------
            true_pred = float(scores[-1])
            true_label = 1.0
            abs_err = (true_pred - true_label) ** 2

            # ---------- group aggregation ----------
            item_group_abs[pos_item][gender_scalar][age_scalar].append(abs_err)

    # ---------- compute item-level unfairness ----------
    item_unfairness = []

    for item, gender_dict in item_group_abs.items():

        # flatten gender × age → group means
        group_means = []
        for gender in gender_dict:
            for age in gender_dict[gender]:
                errs = gender_dict[gender][age]
                if len(errs) > 0:
                    group_means.append(np.mean(errs))

        if len(group_means) < 2:
            continue

        # max pairwise disparity
        max_gap = max(
            abs(group_means[i] - group_means[j])
            for i in range(len(group_means))
            for j in range(i + 1, len(group_means))
        )

        item_unfairness.append(max_gap)

    return round(float(np.mean(item_unfairness)) if item_unfairness else 0.0, 4)

def evaluate_value_gpwise(model,test_ratings,test_negatives,user_attributes,dataset,device):
    """
    Item-conditional value unfairness.

    For each item:
      - compute group-wise mean prediction bias (prediction − ground truth)
      - take maximum pairwise difference across sensitive groups
    Final score = average over items
    """

    model.eval()

    # item -> gender -> age -> [predictions / truths]
    item_group_preds = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    item_group_trues = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    with torch.no_grad():
        for (user, pos_item), negatives in zip(test_ratings, test_negatives):

            if user not in user_attributes:
                continue

            # ---------- scalar attributes (for grouping) ----------
            age_scalar, gender_scalar = user_attributes[user]

            # ---------- model inputs ----------
            items = negatives + [pos_item]
            users = [user] * len(items)

            user_tensor = torch.LongTensor(users).to(device)
            item_tensor = torch.LongTensor(items).to(device)

            # user-sensitive tensors
            user_age_tensor = torch.FloatTensor(
                [dataset.user_attributes[u][0] for u in users]
            ).to(device)

            user_gender_tensor = torch.FloatTensor(
                [dataset.user_attributes[u][1] for u in users]
            ).to(device)

            # item-sensitive vectors
            item_cat_vector = torch.FloatTensor(
                dataset.item_sensitive_vectors[items]
            ).to(device)

            # ---------- prediction ----------
            if hasattr(model, "forward"):
                preds = model(
                    user_tensor,
                    item_tensor,
                    item_cat_vector
                )
            else:
                preds = model(user_tensor, item_tensor)

            scores = preds.squeeze().cpu().numpy()

            # ---------- positive item only ----------
            true_pred = float(scores[-1])
            true_label = 1.0

            # ---------- group aggregation ----------
            item_group_preds[pos_item][gender_scalar][age_scalar].append(true_pred)
            item_group_trues[pos_item][gender_scalar][age_scalar].append(true_label)

    # ---------- compute item-level value unfairness ----------
    value_diffs = []

    for item, gender_dict in item_group_preds.items():

        group_biases = []

        for gender in gender_dict:
            for age in gender_dict[gender]:

                preds = item_group_preds[item][gender][age]
                trues = item_group_trues[item][gender][age]

                if len(preds) == 0:
                    continue

                # mean(pred − true)
                bias = np.mean(preds) - np.mean(trues)
                group_biases.append(bias)

        if len(group_biases) < 2:
            continue

        # max pairwise bias gap
        max_gap = max(
            abs(group_biases[i] - group_biases[j])
            for i in range(len(group_biases))
            for j in range(i + 1, len(group_biases))
        )

        value_diffs.append(max_gap)

    return round(float(np.mean(value_diffs)) if value_diffs else 0.0, 4)

def hit_at_k(ranked_list, ground_truth):
    return int(ground_truth in ranked_list)

def ndcg_at_k(ranked_list, ground_truth):
    if ground_truth in ranked_list:
        idx = ranked_list.index(ground_truth)
        return 1.0 / np.log2(idx + 2)
    return 0.0
