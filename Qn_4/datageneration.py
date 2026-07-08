import numpy as np
import pandas as pd
def generate_student_ids(n):
    branches = ["ch", "ce", "ma", "ai", "cs", "me", "bt", "ee"]
    years = ["22", "23", "24", "25"]
    rolls = [f"110{str(i).zfill(2)}" for i in range(1, 51)]

    all_ids = []
    for branch in branches:
        for year in years:
            for roll in rolls:
                all_ids.append(f"{branch}{year}btech{roll}")

    np.random.shuffle(all_ids)
    return all_ids[:n]
def generate_dataset(n_samples=500, seed=42):
    student_ids = generate_student_ids(600)
    np.random.seed(seed)
    data = []

    for i in range(n_samples):
        student_id = student_ids[np.random.randint(0,600)]
        day_of_week = np.random.randint(0, 7)
        meal_time = np.random.randint(0, 3)
        meal_type = np.random.randint(0, 2)

        is_weekend = 1 if day_of_week >= 5 else 0
        class_day = 0 if day_of_week < 5 else 1
        assignment_deadline = np.random.choice([0, 1], p=[0.8, 0.2])

        # Weather
        temperature = np.random.normal(30, 5)
        humidity = np.random.uniform(40, 90)
        windspeed = np.random.uniform(0, 20)
        rain = np.random.choice([0, np.random.uniform(1, 20)], p=[0.7, 0.3])
        airquality = np.random.uniform(50, 200)

        rising_time = np.random.randint(300, 600)
        sleeping_time = np.random.randint(1320, 1440)

        # Base duration
        if meal_time == 0:
            mess_duration = np.random.randint(15, 30)
        elif meal_time == 1:
            mess_duration = np.random.randint(30, 50)
        else:
            mess_duration = np.random.randint(35, 60)

        # Context effects
        if is_weekend:
            mess_duration += np.random.randint(5, 15)

        if assignment_deadline:
            mess_duration -= np.random.randint(5, 10)

        if rain > 10:
            mess_duration += 5

        if airquality > 150:
            mess_duration -= 3

        mess_duration = max(10, mess_duration)

        data.append([
            student_id, day_of_week, meal_time, meal_type,
            is_weekend, class_day, assignment_deadline,
            temperature, humidity, windspeed, rain, airquality,
            rising_time, sleeping_time, mess_duration
        ])

    columns = [
        "student_id","day_of_week","meal_time","meal_type",
        "is_weekend","class_day","assignment_deadline",
        "temperature","humidity","windspeed","rain","airquality",
        "rising_time","sleeping_time","mess_duration"
    ]

    return pd.DataFrame(data, columns=columns)


# Generate base dataset
df_500 = generate_dataset(500)
df_500.to_csv("mess_dataset_500.csv", index=False)

# Generate 5000-sample dataset with Gaussian noise
df_5000 = df_500.loc[df_500.index.repeat(10)].reset_index(drop=True)

noise_features = [
    "temperature","humidity","windspeed","airquality","rising_time","sleeping_time"
]

for feature in noise_features:
    df_5000[feature] += np.random.normal(0, 2, size=len(df_5000))
df_5000["rain"] += np.random.normal(0, 1, size=len(df_5000))

# Ensure rain is not negative
df_5000["rain"] = np.clip(df_5000["rain"], 0, None)
df_5000["windspeed"] = np.clip(df_5000["rain"], 0, None)
df_5000.to_csv("mess_dataset_5000.csv", index=False)
print("500-sample dataset shape:", df_500.shape)
print("5000-sample dataset shape:", df_5000.shape)