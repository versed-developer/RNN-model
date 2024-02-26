import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

columns = [f"t{i}" for i in range(1, 1441)]
# Load data from 'Full' sheet for features
data_full_train = pd.read_excel('training.xlsx', sheet_name='Full', index_col=0, names=columns)

# Load data from 'G Truth' sheet for labels
g_truth_data = pd.read_excel('training.xlsx', sheet_name='G truth')

# Create a mapping from nodes to phases
node_to_phase = {}
for phase, nodes in g_truth_data.items():
    for node in nodes.dropna():
        node_to_phase[node] = phase

# Create labels based on the mapping
data_full_train['PHASE'] = data_full_train.index.map(node_to_phase)

# Preprocess data
X = data_full_train.drop(columns=['PHASE'])
y = data_full_train['PHASE']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Build and train RNN model with dropout and regularization
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                      alpha=0.0001, max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model on validation set
y_val_pred = model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy}')

# Load new data from 'Full' sheet
data_full_new = pd.read_excel('testing.xlsx', sheet_name='Full', index_col=0, names=columns)

# Preprocess new data
X_new_scaled = scaler.transform(data_full_new)

# Predict phases for new data
predicted_phases = model.predict(X_new_scaled)

# Create a new DataFrame for predicted phases
predicted_df = pd.DataFrame(predicted_phases, index=data_full_new.index, columns=['Predicted_PHASE'])

# Write the predicted phases to a new sheet in 'testing.xlsx'
with pd.ExcelWriter('testing.xlsx', mode='a', engine='openpyxl') as writer:
    predicted_df.to_excel(writer, sheet_name='G truth')
