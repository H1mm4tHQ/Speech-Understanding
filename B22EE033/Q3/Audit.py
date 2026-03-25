from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "cv-valid-test.csv"

# Load local Common Voice sample metadata.
df = pd.read_csv(CSV_PATH)

df["audio_path"] = df["filename"].apply(lambda x: str(BASE_DIR / x))

# -------------------------------
# Documentation Debt
# -------------------------------
cols = ['gender', 'age', 'accent']
missing = df[cols].isnull().mean()

print("\nMissing Metadata (%):")
print(missing * 100)

# -------------------------------
# Representation Bias
# -------------------------------
gender_dist = df['gender'].value_counts(normalize=True)
age_dist = df['age'].value_counts(normalize=True)

print("\nGender Distribution:")
print(gender_dist)

print("\nAge Distribution:")
print(age_dist)

# -------------------------------
# Plotting
# -------------------------------
plt.figure()
gender_dist.plot(kind='bar', title='Gender Distribution')
plt.tight_layout()
plt.savefig(BASE_DIR / "gender_distribution.png")

plt.figure()
age_dist.plot(kind='bar', title='Age Distribution')
plt.tight_layout()
plt.savefig(BASE_DIR / "age_distribution.png")

# -------------------------------
# Save summary
# -------------------------------
with open(BASE_DIR / "audit_summary.txt", "w", encoding="utf-8") as f:
    f.write("Missing Metadata:\n")
    f.write(str(missing) + "\n\n")
    f.write("Gender Distribution:\n")
    f.write(str(gender_dist) + "\n\n")
    f.write("Age Distribution:\n")
    f.write(str(age_dist))
