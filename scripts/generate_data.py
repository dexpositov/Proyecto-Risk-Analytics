import pandas as pd
import numpy as np
import random
import os
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

def generate_fixed_expenses(customers, customer_assignments, customer_locations, profiles_config, 
                            fixed_categories, fixed_penetration, fixed_ranges, start_date, tx_width, start_tx_id):
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
                date = generate_fixed_category_timestamp(start_date, day_range)
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

def get_fraud_date(start_date, days_range, hour_min, hour_max):
    """
    Generates a timestamp for fraudulent transactions within specified day and hour ranges.
    """
    min_day, max_day = days_range
        
    day_offset = random.randint(min_day, max_day)
    
    if hour_min > hour_max: # Overnight range
        possible_hours = list(range(hour_min, 24)) + list(range(0, hour_max + 1))
        hour = random.choice(possible_hours)
    else:
        hour = random.randint(hour_min, hour_max)
        
    return start_date + timedelta(days=day_offset, hours=hour, minutes=random.randint(0, 59))

def generate_velocity_fraud(target_client, home_city, category, start_date, days_range, tx_width, current_id, batch_suffix, attack_location):
    """
    Generates a velocity fraud attack.
    """
    anomalies = []
    base_time = get_fraud_date(start_date, days_range, 2, 6)
    
    num_attacks = random.randint(5, 15)
    
    for j in range(num_attacks): 
        anomalies.append({
            "Transaction_ID": f"FRD-{current_id:0{tx_width}d}",
            "Customer_ID": target_client, 
            "Customer_Profile": "Unknown",
            "Customer_Home": home_city,
            "Amount": round(random.uniform(5, 25), 2), 
            "Timestamp": base_time + timedelta(minutes=j*1),
            "Terminal_ID": f"TERM_VEL_{batch_suffix}",
            "Category": category, 
            "Location": attack_location,
            "Is_Fixed": 0, 
            "Is_Fraud": 1
        })
        current_id += 1
    return anomalies, current_id

def generate_magnitude_fraud(target_client, profile_name, home_city, category, start_date, days_range, tx_width, current_id, batch_suffix, attack_location, lognormal_params):
    """
    Generates a magnitude fraud attack relative to the client's profile.
    Calculates a normal expense for their profile and multiplies it by a risk factor (10x - 20x).
    """
    anomalies = []
    date = get_fraud_date(start_date, days_range, 3, 7)

    # Generates a "normal" amount for that client in that category
    base_amount = generate_amount(profile_name, category, lognormal_params)

    fraud_multiplier = random.uniform(10, 20) # Applies a massive multiplier to create the anomaly
    fraud_amount = round(base_amount * fraud_multiplier, 2)
    
    anomalies.append({
        "Transaction_ID": f"FRD-{current_id:0{tx_width}d}",
        "Customer_ID": target_client, 
        "Customer_Profile": profile_name,
        "Customer_Home": home_city,
        "Amount": fraud_amount, 
        "Timestamp": date,
        "Terminal_ID": f"TERM_BIG_{batch_suffix}",
        "Category": category, 
        "Location": attack_location, 
        "Is_Fixed": 0, 
        "Is_Fraud": 1
    })
    current_id += 1
    return anomalies, current_id

def create_anomalies(customers, customer_locations, discretionary_categories, locations, start_date, tx_width, 
                     start_tx_id, lognormal_params, customer_assignments, num_anomalies_sets=120, days_range=(0, 30)):
    """
    Generates a set of fraud anomalies (velocity and magnitude) for random customers.
    """
    all_anomalies = []
    current_id = start_tx_id
    
    retail_cat = "Retail" if "Retail" in discretionary_categories else discretionary_categories[-1]
    
    for i in range(num_anomalies_sets):
        # Numeric selection of the type of fraud (0: Velocity, 1: Magnitude)
        anomaly_type = random.choice([0, 1])
        batch_suffix = f"{i+1:03d}"
        target_client = random.choice(customers)
        home_city = customer_locations[target_client] 
        profile_name = customer_assignments[target_client]
        
        new_anomalies = []
        
        if anomaly_type == 0: # Velocity Fraud
            attack_loc = random.choice(locations)
            new_anomalies, current_id = generate_velocity_fraud(target_client, home_city, retail_cat, 
                                start_date, days_range, tx_width, current_id, batch_suffix, attack_loc)
            
        elif anomaly_type == 1: # Magnitude Fraud
            # Magnitude fraud in a different city
            possible_locs = [loc for loc in locations if loc != home_city]
            attack_loc = random.choice(possible_locs) if possible_locs else home_city
            
            new_anomalies, current_id = generate_magnitude_fraud(target_client, profile_name, home_city, 
            retail_cat, start_date, days_range, tx_width, current_id, batch_suffix, attack_loc, lognormal_params)
            
        all_anomalies.extend(new_anomalies)
            
    return all_anomalies

def create_all_transactions_data(customers, customer_assignments, customer_locations, selection_probs, 
                             locations, fixed_categories, discretionary_categories, 
                             fixed_penetration, fixed_ranges, num_transactions, start_date, 
                             profiles_config, daily_category_multipliers, tx_width, start_tx_id, 
                             num_anomalies_sets=120, days_range=(0, 30)):
    """
    Principal orchestrator. Returns the data and the next available transaction ID.
    """
    lognormal_params = precompute_lognormal_params(profiles_config)
    profile_cat_weights = calculate_profile_category_weights(profiles_config, discretionary_categories)
    cat_day_probs = get_disc_category_day_probs(discretionary_categories, daily_category_multipliers)
    
    # Fixed Expenses
    fixed_data, next_id_fixed = generate_fixed_expenses(
        customers, customer_assignments, customer_locations, lognormal_params, fixed_categories, 
        fixed_penetration, fixed_ranges, start_date, tx_width, start_tx_id)

    # Discretionary Expenses
    variable_data, next_id_final = generate_discretionary_expenses(
        num_transactions, customers, selection_probs, customer_assignments, customer_locations,
        lognormal_params, discretionary_categories, cat_day_probs, 
        profile_cat_weights, start_date, locations, tx_width, next_id_fixed
    )
    # Anomalies
    anomalous_data = create_anomalies(
        customers, customer_locations, discretionary_categories, locations, start_date,
        tx_width, next_id_final, lognormal_params, customer_assignments, 
        num_anomalies_sets, days_range
    )

    return fixed_data + variable_data + anomalous_data

def main():
    # Creation of the configuration parameters
    fixed_categories = ["Housing", "Utilities"]
    discretionary_categories = ["Transportation", "Retail", "Groceries", "Leisure", "Others"]
    fixed_penetration = {"Housing": 0.75, "Utilities": 0.85}
    fixed_ranges = {"Housing": (1, 5), "Utilities": (1, 10)}
    
    daily_category_multipliers = {
        "Wednesday": {"Others": 1.5, "Retail": 1.2},
        "Friday":    {"Leisure": 2.0, "Transportation": 1.5},
        "Saturday":  {"Leisure": 3.0, "Groceries": 2.5, "Retail": 1.8},
        "Sunday":    {"Leisure": 2.5, "Transportation": 2.0}
    }

    profiles_config = {
        "Thrifty": {
            "frequency_weight": 0.35,
            "category_weights": [0.0, 0.0, 0.05, 0.15, 0.50, 0.20, 0.10],
            "behaviors": {
                "Utilities": (60, 5), "Housing": (550, 20), "Transportation": (40, 10), 
                "Retail": (25, 10), "Groceries": (50, 15), "Leisure": (20, 10), "Others": (10, 5)
            }
        },
        "Standard": {
            "frequency_weight": 1.0,
            "category_weights": [0.0, 0.0, 0.15, 0.20, 0.30, 0.20, 0.15],
            "behaviors": {
                "Utilities": (90, 15), "Housing": (900, 50), "Transportation": (80, 20), 
                "Retail": (80, 40), "Groceries": (80, 25), "Leisure": (70, 30), "Others": (30, 15)
            }
        },
        "Well-off": {
            "frequency_weight": 2.0,
            "category_weights": [0.0, 0.0, 0.60, 0.15, 0.10, 0.10, 0.05],
            "behaviors": {
                "Utilities": (220, 40), "Housing": (3500, 300), "Transportation": (200, 80), 
                "Retail": (700, 300), "Groceries": (200, 50), "Leisure": (450, 150), "Others": (150, 60)
            }
        },
        "Techie": {
            "frequency_weight": 1.75,
            "category_weights": [0.0, 0.0, 0.10, 0.60, 0.10, 0.15, 0.05],
            "behaviors": {
                "Utilities": (120, 25), "Housing": (1200, 100), "Transportation": (100, 30), 
                "Retail": (900, 500), "Groceries": (70, 20), "Leisure": (100, 40), "Others": (60, 30)
            }
        }
    }
    
    profile_assignment_weights = [0.25, 0.55, 0.08, 0.12]
    num_clients = 2900
    num_discretionary_tx = 100000 
    start_date = datetime(2026, 1, 1)
    locations = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Bilbao", "Zaragoza", "Malaga"]
    city_weights = [0.3, 0.25, 0.15, 0.1, 0.05, 0.05, 0.1]
    
    customers, customer_assignments, selection_probs = assign_profiles(
        num_clients, profiles_config, profile_assignment_weights)
    
    customer_locations = assign_locations(customers, locations, city_weights)
    
    estimated_total = num_discretionary_tx + (len(customers) * len(fixed_categories)) + 1000
    tx_width = len(str(estimated_total))
    
    all_data = create_all_transactions_data(
        customers, customer_assignments, customer_locations, selection_probs, 
        locations, fixed_categories, discretionary_categories, 
        fixed_penetration, fixed_ranges, num_discretionary_tx, start_date, 
        profiles_config, daily_category_multipliers, tx_width, start_tx_id=1, 
        num_anomalies_sets=120, days_range=(0, 30)
    )

    df = pd.DataFrame(all_data)
    df = df.sort_values(by="Timestamp").reset_index(drop=True)
    
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, "transactions_simulated.csv")
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()