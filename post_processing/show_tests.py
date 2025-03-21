import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data from your testing results
data = {
    "combination": [
        "Left Wrist Only", "Right Wrist Only", "Left Pocket Only", "Right Pocket Only", "Head Only",
        "Both Wrists", "Left Wrist + Left Pocket", "Left Wrist + Right Pocket", "Left Wrist + Head",
        "Right Wrist + Left Pocket", "Right Wrist + Right Pocket", "Right Wrist + Head",
        "Both Pockets", "Left Pocket + Head", "Right Pocket + Head",
        "Both Wrists + Left Pocket", "Both Wrists + Right Pocket", "Both Wrists + Head"
    ],
    "num_sensors": [
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3
    ],
    "mse": [
        0.574109, 0.577276, 0.569932, 0.569164, 0.566130,
        0.574908, 0.564122, 0.565514, 0.565254, 0.571568, 0.577925, 0.574526,
        0.559116, 0.555689, 0.561112,
        0.572576, 0.572750, 0.574538
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Add sensor types for additional analysis
df["has_wrist"] = df["combination"].apply(lambda x: "Wrist" in x)
df["has_pocket"] = df["combination"].apply(lambda x: "Pocket" in x)
df["has_head"] = df["combination"].apply(lambda x: "Head" in x)

# Calculate statistics by number of sensors
sensor_stats = df.groupby("num_sensors")["mse"].agg(["mean", "std", "min", "max"]).reset_index()

# Plot 1: MSE by number of sensors
plt.figure(figsize=(12, 8))

# Boxplot of MSE by number of sensors
plt.subplot(2, 2, 1)
sns.boxplot(x="num_sensors", y="mse", data=df)
plt.title("MSE by Number of Sensors")
plt.xlabel("Number of Sensors")
plt.ylabel("Mean Squared Error")

# Plot 2: Line plot with error bars
plt.subplot(2, 2, 2)
plt.errorbar(sensor_stats["num_sensors"], sensor_stats["mean"], 
             yerr=sensor_stats["std"], marker='o', capsize=5)
plt.title("Mean MSE by Number of Sensors (with Std Dev)")
plt.xlabel("Number of Sensors")
plt.ylabel("Mean Squared Error")
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 3: Top 5 best combinations
plt.subplot(2, 2, 3)
top_5 = df.sort_values("mse").head(5)
sns.barplot(x="combination", y="mse", data=top_5)
plt.title("Top 5 Sensor Combinations")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Mean Squared Error")

# Plot 4: Impact of specific sensors
plt.subplot(2, 2, 4)
sensor_impact = pd.DataFrame({
    "Sensor Type": ["Wrist", "Pocket", "Head"],
    "Average MSE": [
        df[df["has_wrist"]]["mse"].mean(),
        df[df["has_pocket"]]["mse"].mean(),
        df[df["has_head"]]["mse"].mean()
    ]
})
sns.barplot(x="Sensor Type", y="Average MSE", data=sensor_impact)
plt.title("Average MSE by Sensor Type")
plt.ylabel("Mean Squared Error")

plt.tight_layout()
plt.savefig("sensor_analysis.png", dpi=300, bbox_inches="tight")

# Display the best combination for each number of sensors
best_per_count = df.loc[df.groupby("num_sensors")["mse"].idxmin()]
print("\nBest Combination per Sensor Count:")
for _, row in best_per_count.iterrows():
    print(f"{row['num_sensors']} sensors: {row['combination']} (MSE: {row['mse']:.6f})")

# Calculate improvement percentages
single_sensor_avg = sensor_stats[sensor_stats["num_sensors"] == 1]["mean"].values[0]
two_sensor_avg = sensor_stats[sensor_stats["num_sensors"] == 2]["mean"].values[0]
three_sensor_avg = sensor_stats[sensor_stats["num_sensors"] == 3]["mean"].values[0]

improvement_1_to_2 = (single_sensor_avg - two_sensor_avg) / single_sensor_avg * 100
improvement_2_to_3 = (two_sensor_avg - three_sensor_avg) / two_sensor_avg * 100
improvement_1_to_3 = (single_sensor_avg - three_sensor_avg) / single_sensor_avg * 100

print(f"\nImprovement from 1 to 2 sensors: {improvement_1_to_2:.2f}%")
print(f"Improvement from 2 to 3 sensors: {improvement_2_to_3:.2f}%")
print(f"Overall improvement from 1 to 3 sensors: {improvement_1_to_3:.2f}%")