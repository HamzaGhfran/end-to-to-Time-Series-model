
from datetime import datetime, timedelta


def make_short_form(text):
    words = text.split()
    short_form = ''.join(word[0].upper() for word in words if len(word) >= 4)
    return short_form
# Extract season based on Pakistan's seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'

# Function to determine if Eid ul-Adha is in the given month or one month before/after
eid_ul_adha_dates = {
    2020: datetime(2020, 7, 31),
    2021: datetime(2021, 7, 20),
    2022: datetime(2022, 7, 9),
    2023: datetime(2023, 6, 27)
}

def is_eid_ul_adha(year, month):
    if year not in eid_ul_adha_dates:
        return 0
    eid_date = eid_ul_adha_dates[year]
    eid_month = eid_date.month
    return 1 if month in {eid_month, (eid_month % 12) + 1, (eid_month - 2) % 12 + 1} else 0

# Function to determine if Ramdhan is in the given month
ramadan_start_end_month = {
    2020: (4, 5),
    2021: (4, 5),
    2022: (4, 5),
    2023: (3, 4),
    2024: (3, 4)
}
def is_ramadan(year, month):
    if year not in ramadan_start_end_month:
        return 0
    start_month, end_month = ramadan_start_end_month[year]
    return 1 if month in {start_month, end_month} else 0

# Monsoon season (June to September)
def is_monsoon(month):
    return 1 if month in [6, 7, 8, 9] else 0



# Merge
def standardize_unique_id(unique_id):
    unique_id_lower = unique_id.lower()
    replacements = {
        'drp': 'drip',
        'syrinje': 'syringe'
    }
    for key, value in replacements.items():
        unique_id_lower = unique_id_lower.replace(key, value)
    return unique_id_lower

def extract_medicine_type(unique_id, data_types):
    unique_id_lower = standardize_unique_id(unique_id)
    for data_type in data_types:
        if data_type.lower() in unique_id_lower:
            return unique_id_lower.replace(data_type.lower(), '').strip(), data_type.lower()
    return unique_id_lower, 'misc'

def insert_medicine_type(unique_id, medicine_type):
    parts = unique_id.split('_', 2)
    if parts[2].startswith('.'):
        parts[2] = parts[2][1:]  # Remove the first character if it's a dot
    return f"{parts[0]}_{parts[1]}_{medicine_type}#{parts[2]}"
