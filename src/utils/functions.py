import csv
import os


def save_to_csv(path, name, data):
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(path, exist_ok=True)
    file = os.path.join(path, name)
    exists = os.path.isfile(file)
    with open(file, 'a', newline='') as f:
        row = dict(**data)
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)
