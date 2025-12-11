import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="FoldFocus: K-Fold Visualizer", layout="wide")

st.title("📊 FoldFocus: K-Fold Cross-Validation Visualizer")
st.markdown("""
This app demonstrates **accuracy variation** across different data splits. 
Adjust the parameters in the sidebar to see how stable your model is!
""")

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("1. Data Parameters")
n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300, step=50)
noise_level = st.sidebar.slider("Noise Level (Difficulty)", 0.0, 1.0, 0.2)

st.sidebar.header("2. Model Configuration")
model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
k_folds = st.sidebar.slider("Number of Folds (K)", 2, 20, 5)

# --- 3. DATA GENERATION ---
# We use synthetic data so the app works out-of-the-box without file uploads
X, y = make_classification(
    n_samples=n_samples,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    flip_y=noise_level, # Adds noise/error to labels
    random_state=42
)

# Show a snippet of data
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Synthetic Data Preview")
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(1, 6)])
    df['Target'] = y
    st.dataframe(df.head(5), use_container_width=True)

# --- 4. MODEL TRAINING & VALIDATION ---
if model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier()
else:
    model = RandomForestClassifier()

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Run Cross Validation
# We loop manually to capture accuracy per fold easily for plotting
accuracies = []
fold_indices = []

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    accuracies.append(score)
    fold_indices.append(f"Fold {i+1}")

# --- 5. VISUALIZATION ---
with col2:
    st.subheader(f"Performance across {k_folds} Folds")
    
    # create the plot
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(fold_indices, accuracies, color='#4CAF50', alpha=0.7)
    
    # Add a red line for the average accuracy
    mean_acc = np.mean(accuracies)
    ax.axhline(mean_acc, color='red', linestyle='--', label=f'Mean Accuracy: {mean_acc:.2f}')
    
    # Formatting
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy Score")
    ax.set_title(f"Variation: {(np.max(accuracies) - np.min(accuracies)):.2f} (Max - Min)")
    ax.legend()
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    st.pyplot(fig)

# --- 6. STATS SUMMARY ---
st.markdown("---")
st.subheader("📝 Summary Statistics")
m1, m2, m3 = st.columns(3)
m1.metric("Mean Accuracy", f"{mean_acc:.2%}")
m2.metric("Standard Deviation", f"{np.std(accuracies):.2%}")
m3.metric("Variance", f"{np.var(accuracies):.4f}")

if np.std(accuracies) > 0.05:
    st.warning("⚠️ High variance detected! Your model performance depends heavily on which data split is used.")
else:
    st.success("✅ Low variance. Your model is stable across different folds.")