import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Seed configuration for reproducibility
np.random.seed(22)
random.seed(22)

def get_lognormal_params(mean, std):
    """
    Convert mean and standard deviation to log-normal parameters mu and sigma_sq.
    """
    variance = std**2
    # Sigma squared computation (log-normal variance)
    sigma_sq = np.log(1 + (variance / (mean**2)))
    sigma = np.sqrt(sigma_sq)
    
    # Mu computation (log-normal mean)
    mu = np.log(mean) - 0.5 * sigma_sq
    
    return mu, sigma

def assign_profiles(num_customers, profiles_config, assignment_weights):
    """
    Generates customer IDs and assigns a behavioral profile to each, 
    calculating selection probabilities based on frequency weights.
    """
   
    id_width = len(str(num_customers))
    
    # Generate dynamic IDs: CUS-0001, CUS-0002...
    customers = [f"CUS-{i+1:0{id_width}d}" for i in range(num_customers)]
    
    profile_names = list(profiles_config.keys())
    
    assigned_profiles = np.random.choice(profile_names, size=num_customers, p=assignment_weights) # Assign profiles based on provided weights
    
    customer_assignments = {}
    customer_selection_weights = []
    
    for i in range(num_customers):
        customer_assignments[customers[i]] = assigned_profiles[i]
        # Use frequency weight from config
        customer_selection_weights.append(profiles_config[assigned_profiles[i]]["frequency_weight"])

    selection_probs = np.array(customer_selection_weights) / sum(customer_selection_weights) # Normalize weights to sum to 1 for selection
    return customers, customer_assignments, selection_probs


