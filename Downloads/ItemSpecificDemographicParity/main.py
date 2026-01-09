import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import yaml
import argparse
import time
import os
from tqdm import tqdm
import random
import torch.nn.functional as F

from dataset import Dataset
from model.fair_mlp import FairMLP
from model.fair_neumf import FairNeuMF
from model.fair_pmf import FairPMF
from model.fair_biasedmf import FairBiasedMF
from evaluate_ML import evaluate_model,evaluate_ISDP, evaluate_value, evaluate_absolute, evaluate_absolute_gpwise,evaluate_value_gpwise

# -----------------------------
# Utilities
# -----------------------------
def compute_ISDP_loss(predictions, ages, genders, categories):
    """
    Differentiable ISDP fairness loss for a batch.

    - Category 1 (more sensitive): penalizes the *maximum pairwise difference*
      among the means of all (age, gender) groups present in the batch.

    - Category 0: penalizes mean difference between male and female groups.

    Works directly on prediction logits or probabilities.
    """
    device = predictions.device
    fairness_loss = torch.tensor(0.0, device=device)

    # --- Category 1: subgroup fairness across 4 groups (age ∈ {0,1}, gender ∈ {0,1}) ---
    cat1_mask = (categories == 1)
    if cat1_mask.any():
        cat1_preds = predictions[cat1_mask]
        cat1_ages = ages[cat1_mask]
        cat1_genders = genders[cat1_mask]
        
        group_means = []
        for age_val in [0, 1]:
            for gender_val in [0, 1]:
                mask = (cat1_ages == age_val) & (cat1_genders == gender_val)
                if mask.any():
                    group_means.append(cat1_preds[mask].mean())

        # If 2+ groups appear, compute max absolute difference
        if len(group_means) > 1:
            group_means = torch.stack(group_means)
            diffs = torch.abs(group_means.unsqueeze(0) - group_means.unsqueeze(1))
            fairness_loss = fairness_loss + diffs.max()

    # --- Category 0: binary gender fairness ---
    cat0_mask = (categories == 0)
    if cat0_mask.any():
        cat0_preds = predictions[cat0_mask]
        cat0_genders = genders[cat0_mask]

        male_mask = (cat0_genders == 0)
        female_mask = (cat0_genders == 1)

        if male_mask.any() and female_mask.any():
            male_mean = cat0_preds[male_mask].mean()
            female_mean = cat0_preds[female_mask].mean()
            fairness_loss = fairness_loss + torch.abs(male_mean - female_mean)

    return fairness_loss

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_model(model_type, num_users, num_items, latent_dim=None):
    """Create model based on configuration"""
    if model_type.upper() == 'FAIRMLP':
        return FairMLP(num_users=num_users,num_items=num_items,num_sensitive_attrs=3, fair_lambda=args.fair_lambda)
    elif model_type.upper() == 'FAIRNEUMF':
        return FairNeuMF(num_users=num_users,num_items=num_items,gmf_dim=latent_dim,num_sensitive_attrs=3, fair_lambda=args.fair_lambda)
    elif model_type.upper() == 'FAIRPMF':
        return FairPMF(num_users=num_users,num_items=num_items,latent_dim=latent_dim,num_sensitive_attrs=3, fair_lambda=args.fair_lambda)  # age & gender
    elif model_type.upper() == 'FAIRBIASEDMF':
        return FairBiasedMF(num_users=num_users,num_items=num_items,latent_dim=latent_dim,num_sensitive_attrs=3, fair_lambda=args.fair_lambda)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_device(device_config):
    """Get device based on configuration"""
    if device_config == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_config)

def set_seed(seed):
    """Set seed for reproducibility"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Seed set to: {seed}")
    else:
        print("No seed set - using random initialization")


# -----------------------------
# Training Loop (MODIFIED)
# -----------------------------
def main(args):
    # Config + Setup
    config = load_config(args.config)
    device = get_device(args.device)
    set_seed(args.seed)
    
    # Dataset (keep the same)
    dataset = Dataset(args.train_file, args.test_file, args.neg_file, args.user_file, args.movie_file)
    num_users, num_items = dataset.num_users, dataset.num_items
    users, items, labels, ages, genders, categories = dataset.get_train_instances()
    
    # Model + Optimizer
    model = get_model(args.model, num_users, num_items, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.BCELoss() # Standard BCE loss
    
    # Save directory (keep the same)
    save_path = f"log/model_{args.model}_seed{args.seed}.log"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_hr = 0.0
    best_ndcg = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_epoch = 0
    
    # Epoch loop
    epoch_pbar = tqdm(range(args.epochs), desc="Training Progress", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        
        # --- Generate training instances (keep the same) ---
        user_input, item_input, labels, ages, genders, categories = dataset.get_train_instances(args.num_negatives)
        # --- Get item-sensitive vectors for each item in the training data (keep the same) ---
        item_sensitive_vecs = torch.FloatTensor(dataset.item_sensitive_vectors[item_input])
        train_dataset = TensorDataset(
            torch.LongTensor(user_input), torch.LongTensor(item_input), torch.FloatTensor(labels),
            torch.FloatTensor(ages), torch.FloatTensor(genders), torch.FloatTensor(categories),item_sensitive_vecs)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        # Batch loop
        epoch_loss = 0
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", leave=False)
        
        for batch in batch_pbar:
            optimizer.zero_grad()
            
            # --- Forward & Loss ---
            batch = [x.to(device) for x in batch]
            users, items, lbls, ages, genders, cats, item_vecs = batch
            preds = model(users, items, item_vecs)
            preds = preds.squeeze(-1)
            
            # 1. Base Prediction Loss
            pred_loss = loss_function(preds, lbls)

            # 2. ISDP fairness loss
            isdp_loss = compute_ISDP_loss(preds, ages, genders, cats)
            #fairness_loss = torch.norm(w_s, p=2)  # L2 norm on w_s

            # ---------------------------------------
            # 5. Total Fairness Loss
            # ---------------------------------------
            total_loss = pred_loss + args.fair_lambda * isdp_loss
         
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            batch_pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
            
        # Evaluation (keep the same)
        hr, ndcg  = evaluate_model(model,dataset.test_ratings,dataset.test_negatives,args.top_k,device=device,dataset=dataset)
        avg_max_diff_cat1,avg_max_diff_cat0,group_avg_scores_cat1,group_avg_scores_cat0,num_items_cat1,num_items_cat0 = evaluate_ISDP(model, dataset.test_ratings, dataset.test_negatives, dataset.user_attributes, dataset.movie_categories, device=device,dataset=dataset)
        #avg_value_gpwise = evaluate_value_gpwise(model, dataset.test_ratings, dataset.test_negatives, dataset.user_attributes, device=device,dataset=dataset)
        #avg_absolute_gpwise = evaluate_absolute_gpwise(model, dataset.test_ratings, dataset.test_negatives, dataset.user_attributes,device=device,dataset=dataset)
        avg_absolute = evaluate_absolute(model, dataset.test_ratings, dataset.test_negatives, dataset.user_attributes,device=device,dataset=dataset)
        avg_value = evaluate_value(model, dataset.test_ratings, dataset.test_negatives, dataset.user_attributes,device=device,dataset=dataset)
        if hr > best_hr:
            best_hr = hr
            best_epoch = epoch  # Track the epoch of best metrics
            torch.save(model.state_dict(), save_path)
            tqdm.write(f"New best model saved with HR: {best_hr:.4f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_epoch = epoch  # Track the epoch of best metrics
            tqdm.write(f"New best NDCG: {best_ndcg:.4f}")       

        # Update progress bar with best NDCG
        epoch_pbar.set_postfix({
            'Loss': f'{epoch_loss:.4f}',
            'HR': f'{hr:.4f}', 
            'NDCG': f'{ndcg:.4f}', 
            'BestHR': f'{best_hr:.4f}',
            'BestNDCG': f'{best_ndcg:.4f}'
        })

    epoch_pbar.close()
    print(f"Training completed. Best HR: {best_hr:.4f}, Best NDCG: {best_ndcg:.4f} at epoch {best_epoch}")
    print(f"Best model saved to: {save_path}")
    print("Average group scores of Movies(R)", group_avg_scores_cat0)
    print("Average group scores of General", group_avg_scores_cat1)
    print("ISDP Max Difference of Movies(R)", avg_max_diff_cat0) 
    print("ISDP Max Difference of Movies(General)", avg_max_diff_cat1) 
    print("Value Unfairness between Genders", avg_value)
    print("Absolute Unfairness between Genders", avg_absolute)
    #print("Absolute Unfairness between Genders", avg_absolute_gpwise)
   #print("Value Unfairness between Genders", avg_value_gpwise)
# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train recommendation model')
    parser.add_argument('--config', type=str, default='properties/overall.yaml')
    parser.add_argument('--train_file', type=str, default='data/ml_1m/train.rating')
    parser.add_argument('--test_file', type=str, default='data/ml_1m/test.rating')
    parser.add_argument('--user_file', type=str, default='data/ml_1m/users.dat')
    parser.add_argument('--movie_file', type=str, default='data/ml_1m/movies.dat')
    parser.add_argument('--neg_file', type=str, default='data/ml_1m/test.negative')
    parser.add_argument("--group_attribute", type=str, default='ageandgender', choices=['gender', 'age','ageandgender'],help="Sensitive attribute to evaluate fairness")
    parser.add_argument("--fair_attribute", type=str, default='gender', choices=['gender', 'age','ageandgender'],help="Sensitive attribute to evaluate fairness")
    parser.add_argument('--seed', type=int, default=2053)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.000889898935614925)
    parser.add_argument('--fair_lambda', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_negatives', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--fair_objective', type=str, default='none', help='Fairness objective (none, single, rf)')
    parser.add_argument('--model', type=str, choices=['FairPMF', 'FairMLP', 'FairNeuMF', 'FairBiasedMF'], default='FairPMF')
    args = parser.parse_args()
    main(args)