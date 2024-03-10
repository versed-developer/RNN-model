import os

import pandas as pd
from tensorflow.keras.models import load_model

from lstm import load_data, write_results

model_path = './lstm_model.keras'


def evaluate_saved_models():
    test_file_name = input('Please enter excel file name that includes test data:(testing?)').strip() or 'testing'
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Model loaded successfully.")

        # Final evaluation of the model using testing.xlsx
        x_test, y_test, index, g_truth = load_data(f'{test_file_name}.xlsx')
        scores = model.evaluate(x_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        predicted_values = model.predict(x_test)
        write_results(predicted_values, g_truth, index, f'{test_file_name}_result.xlsx')
    else:
        print(f"The file '{model_path}' does not exist.")


if __name__ == '__main__':
    evaluate_saved_models()
