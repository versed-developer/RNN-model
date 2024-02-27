import os

import pandas as pd
from tensorflow.keras.models import load_model

from lstm import load_data, get_phase_labels, add_or_update_excel_sheet

model_path = './lstm_model.keras'
phase_list = ['A', 'B', 'C']


def evaluate_saved_models():
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Model loaded successfully.")

        # Final evaluation of the model using testing.xlsx
        x_test, y_test, index = load_data('testing.xlsx')
        scores = model.evaluate(x_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        predicted_values = model.predict(x_test)
        phase_labels = get_phase_labels(predicted_values)
        predicted_df = pd.DataFrame(phase_labels, index=index, columns=['Predicted_PHASE'])
        add_or_update_excel_sheet(f'testing.xlsx', 'Predicted_Phases', predicted_df)
    else:
        print(f"The file '{model_path}' does not exist.")


if __name__ == '__main__':
    evaluate_saved_models()
