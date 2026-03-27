import pandas as pd
import random
from datetime import date, timedelta

def get_synthetic_data():
    first = random.choice(['James', 'Mary', 'Robert', 'Patricia', 'John', 'Jennifer'])
    last = random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia'])
    dom = random.choice(['gmail.com', 'outlook.com', 'healthmail.org'])
    days_old = random.randint(0, 36500) 
    dob = date.today() - timedelta(days=days_old)
    return {
        'first': first,
        'last': last,
        'email': f"{first.lower()}.{last.lower()}@{dom}",
        'phone': f"{random.randint(100,999)}-555-{random.randint(1000,9999)}",
        'address': f"{random.randint(100, 999)} Maple St, Springfield",
        'dob': dob
    }

def calculate_exact_age(birthdate):
    today = date.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

def generate_patient_records(num_records):
    records = []
    for i in range(num_records):
        data = get_synthetic_data()
        records.append({
            'PatientID': 1000 + i,
            'FirstName': data['first'],
            'LastName': data['last'],
            'Gender': random.choice(['Male', 'Female', 'Non-binary']),
            'Birthdate': data['dob'],
            'Age': calculate_exact_age(data['dob']),
            'Email': data['email'],
            'Phone': data['phone'],
            'Address': data['address'],
            'BloodType': random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']),
            'HasInsurance': random.choice([True, False])
        })
    return records

# 1. Generate the data
NUM_RECORDS = 1000
df = pd.DataFrame(generate_patient_records(NUM_RECORDS))

# 2. THE CSV FUNCTION: Save the DataFrame to a file
# index=False prevents pandas from adding an extra column of numbers
csv_filename = "patient_data.csv"
df.to_csv(csv_filename, index=False)

# 3. Print verification to console
print(f"--- Processed {NUM_RECORDS} Records ---")
print(df[['PatientID', 'FirstName', 'LastName', 'Age', 'BloodType']].head().to_string(index=False))
print(f"\n✅ Success! File saved as: {csv_filename}")