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

def precompute_lognormal_params(profiles_config):
    """
    Calculates the mu and sigma parameters for each combination of Profile and Category.
    Returns a nested dictionary for fast lookup.
    """
    precomputed = {}
    for profile_name, config in profiles_config.items():
        precomputed[profile_name] = {}
        for category, (mean, std) in config["behaviors"].items():
            mu, sigma = get_lognormal_params(mean, std)
            precomputed[profile_name][category] = (mu, sigma)
    return precomputed

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

def assign_locations(customers, locations, city_weights):
    """
    Assigns a Home Location to each customer based on demographic weights.
    """
    customer_locations = {}
    
    # Assign a fixed home city using the provided weights
    assigned_cities = np.random.choice(locations, size=len(customers), p=city_weights)
    
    for i, cust_id in enumerate(customers):
        customer_locations[cust_id] = assigned_cities[i]
        
    return customer_locations

def get_disc_category_day_probs(discretionary_categories, daily_multipliers):
    """
    Calculates the probability that a discretionary category occurs on each day of the week.
    """
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    cat_day_probs = {}
    
    for cat in discretionary_categories:
        weights = []
        for day in weekdays:
            day_configs = daily_multipliers.get(day, {})
            multiplier = day_configs.get(cat, 1.0)
            weights.append(multiplier)
        
        cat_day_probs[cat] = np.array(weights) / sum(weights) # Normalize weights
        
    return cat_day_probs

def generate_bimodal_hour():
    """
    Generates a decimal hour using a mixture of two normal distributions.
    No restrictions: values outside [0, 24] will be handled via date overflow.
    """
    # Parameters of the two peaks (morning and noon)
    p1 = (10, 2.5)  # mu=10, sigma=2.5 (morning)
    p2 = (20, 2.0)  # mu=20, sigma=2.0 (noon)
    p_weight = 0.45 # Weight for the first peak (morning)
    
    # Choose a peak and generate the decimal hour
    if random.random() < p_weight:
        return random.gauss(p1[0], p1[1])
    else:
        return random.gauss(p2[0], p2[1])

def generate_disc_category_timestamp(start_date, discretionary_category, cat_day_probs, days_range=30):
    """
    Generates a timestamp based on the temporal preferences of the discretionary category.
    """
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    target_day = np.random.choice(weekdays, p=cat_day_probs[discretionary_category]) # Select day name based on category probabilities
    
    possible_dates = []
    for d in range(days_range):
        candidate_date = start_date + timedelta(days=d) # Check each date in the range
        if candidate_date.strftime('%A') == target_day:
            possible_dates.append(candidate_date) # Collect matching dates
            
    base_date = random.choice(possible_dates) # Randomly select one of the matching dates
    
    decimal_hour = generate_bimodal_hour() # Generate decimal hour
    return base_date + timedelta(hours=decimal_hour)

def generate_fixed_category_timestamp(start_date, day_range):
    """
    Generates a timestamp for a fixed expense (direct debit).
    
    Args:
        start_date: Start date of the month.
        day_range: Tuple (min_day, max_day) of the month where the charge occurs.
    """
    # Select a random day within the specified range
    day_offset = random.randint(day_range[0], day_range[1])

    # Direct debit charges are typically processed in batches at first hour (8:00 - 10:00 AM)
    hour = random.randint(8, 9)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    
    return start_date + timedelta(days=day_offset-1, hours=hour, minutes=minute, seconds=second)

def generate_amount(profile, category, precomputed_params):
    """
    Generates the transaction amount based on the profile and category.
    """
    mu, sigma = precomputed_params[profile][category]
    return round(np.random.lognormal(mean=mu, sigma=sigma), 2)

def generate_fixed_expenses(customers, customer_assignments, customer_locations, profiles_config, fixed_categories, fixed_penetration, fixed_ranges, start_date, tx_width, start_tx_id):
    """
    Generates fixed expenses. The location is always the customer's home city.
    """
    data = []
    current_id = start_tx_id
    
    for cust_id in customers:
        prof_name = customer_assignments[cust_id]
        home_city = customer_locations[cust_id] 
        
        for cat in fixed_categories:
            penetration = fixed_penetration.get(cat, 1.0)
            if random.random() < penetration:
                day_range = fixed_ranges.get(cat, (1, 28))
                date = generate_fixed_category_timestamp(start_date, cat, day_range)
                amount = generate_amount(prof_name, cat, profiles_config)
                
                data.append({
                    "Transaction_ID": f"TXN-{current_id:0{tx_width}d}",
                    "Customer_ID": cust_id,
                    "Customer_Profile": prof_name,
                    "Customer_Home": home_city,
                    "Amount": amount,
                    "Timestamp": date,
                    "Terminal_ID": f"FIXED-{cust_id[-4:]}",
                    "Category": cat,
                    "Location": home_city,
                    "Is_Fixed": 1, "Is_Fraud": 0
                })
                current_id += 1
    return data, current_id

def calculate_profile_category_weights(profiles_config, discretionary_categories):
    """
    Extracts the weights of the discretionary categories for each profile.
    Assumes that the input configuration already sums to 1.0 among the discretionary categories.
    """
    extracted_weights_by_profile = {}
    
    # Extract the order of categories used in the configuration
    sample_profile = next(iter(profiles_config))
    all_categories = list(profiles_config[sample_profile]["behaviors"].keys())
    
    for profile_name, config in profiles_config.items():
        # Identify the indices of the categories we are interested in
        indices = [all_categories.index(cat) for cat in discretionary_categories]
        
        # Extract directly the values from the configuration vector
        weights = np.array([config["category_weights"][i] for i in indices])
        
        # Save the resulting vector
        extracted_weights_by_profile[profile_name] = weights
        
    return extracted_weights_by_profile

def generate_discretionary_expenses(num_transactions, customers, selection_probs, customer_assignments, customer_locations,
                                   precomputed_params, discretionary_categories, cat_day_probs, 
                                   profile_cat_weights, start_date, locations, tx_width, start_tx_id):
    """
    Generates discretionary expenses data.
    """
    data = []
    current_id = start_tx_id

    for i in range(num_transactions):
        cust_id = np.random.choice(customers, p=selection_probs)
        prof_name = customer_assignments[cust_id]
        home_city = customer_locations[cust_id]
        category_weights = profile_cat_weights[prof_name]
        category = np.random.choice(discretionary_categories, p=category_weights)
        date = generate_disc_category_timestamp(start_date, category, cat_day_probs)
        amount = generate_amount(prof_name, category, precomputed_params)
        
        # Location assignment with 95% chance of being home city
        if random.random() < 0.95:
            tx_location = home_city
        else:
            possible_locs = [loc for loc in locations if loc != home_city]
            tx_location = random.choice(possible_locs)
        
        data.append({
            "Transaction_ID": f"TXN-{current_id:0{tx_width}d}",
            "Customer_ID": cust_id,
            "Customer_Profile": prof_name,
            "Customer_Home": home_city,
            "Amount": amount,
            "Timestamp": date,
            "Terminal_ID": f"TERM_{random.randint(100, 999)}",
            "Category": category,
            "Location": tx_location,
            "Is_Fixed": 0, "Is_Fraud": 0
        })
        current_id += 1
    return data, current_id