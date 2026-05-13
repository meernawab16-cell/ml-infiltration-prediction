
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 1) Load dataset
df = pd.read_csv("soil_ml_dataset.csv")
print("Dataset loaded successfully.")
print(df.head())
print("\nShape:", df.shape)

# 2) Define inputs and target
X = df[["soil_type", "tss_mg_l", "lapam_mg_l", "moisture_pct"]]
y = df["reduction_pct"]

# 3) Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["soil_type"])
    ],
    remainder="passthrough"
)

# 4) Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# 5) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []

for name, model in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results).sort_values("RMSE")
print("\nModel comparison:")
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

final_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", best_model)
])

final_pipe.fit(X_train, y_train)
final_pred = final_pipe.predict(X_test)

print(f"\nBest model: {best_model_name}")
print("Final MAE:", mean_absolute_error(y_test, final_pred))
print("Final RMSE:", mean_squared_error(y_test, final_pred) ** 0.5)
print("Final R2:", r2_score(y_test, final_pred))

# 6) Save prediction table
pred_df = X_test.copy()
pred_df["actual_reduction_pct"] = y_test.values
pred_df["predicted_reduction_pct"] = final_pred
pred_df.to_csv("predictions_output.csv", index=False)
print("\nSaved predictions_output.csv")

# 7) Plot 1: Actual vs Predicted
plt.figure(figsize=(7,5))
plt.scatter(y_test, final_pred)
plt.xlabel("Actual Reduction (%)")
plt.ylabel("Predicted Reduction (%)")
plt.title(f"Actual vs Predicted ({best_model_name})")
plt.grid(True)
plt.tight_layout()
plt.savefig("figure_actual_vs_predicted.png", dpi=300)
plt.show()

# 8) Plot 2: Reduction vs TSS
group_tss = df.groupby("tss_mg_l")["reduction_pct"].mean()
plt.figure(figsize=(7,5))
plt.plot(group_tss.index, group_tss.values, marker='o')
plt.xlabel("TSS (mg/L)")
plt.ylabel("Mean Reduction (%)")
plt.title("Reduction in Infiltration vs TSS")
plt.grid(True)
plt.tight_layout()
plt.savefig("figure_reduction_vs_tss.png", dpi=300)
plt.show()

# 9) Plot 3: Reduction vs LA-PAM
group_lapam = df.groupby("lapam_mg_l")["reduction_pct"].mean()
plt.figure(figsize=(7,5))
plt.plot(group_lapam.index, group_lapam.values, marker='o')
plt.xlabel("LA-PAM (mg/L)")
plt.ylabel("Mean Reduction (%)")
plt.title("Reduction in Infiltration vs LA-PAM")
plt.grid(True)
plt.tight_layout()
plt.savefig("figure_reduction_vs_lapam.png", dpi=300)
plt.show()

# 10) Plot 4: Soil type comparison
group_soil = df.groupby("soil_type")["reduction_pct"].mean()
plt.figure(figsize=(7,5))
plt.bar(group_soil.index, group_soil.values)
plt.xlabel("Soil Type")
plt.ylabel("Mean Reduction (%)")
plt.title("Soil Type Comparison")
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("figure_soil_type_comparison.png", dpi=300)
plt.show()

# 11) Cross-validation on Random Forest
cv_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(cv_pipe, X, y, cv=kf, scoring="neg_root_mean_squared_error")
cv_rmse = -cv_scores

print("\nCross-validation RMSE scores:", cv_rmse)
print("Mean CV RMSE:", cv_rmse.mean())

print("\nAll done.")
