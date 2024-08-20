import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

class LifelongLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lifelong Learning System")

        self.csv_file = 'data.csv'
        self.model = LinearRegression()

        # Initialize CSV file
        if not os.path.exists(self.csv_file):
            df = pd.DataFrame(columns=['x', 'y'])
            df.to_csv(self.csv_file, index=False)

        # Input frame
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(pady=10)

        self.input_label = tk.Label(self.input_frame, text="Enter new data point (x,y):")
        self.input_label.grid(row=0, column=0)

        self.input_entry = tk.Entry(self.input_frame)
        self.input_entry.grid(row=0, column=1)

        self.add_button = tk.Button(self.input_frame, text="Add Data", command=self.add_data)
        self.add_button.grid(row=0, column=2)

        # Update button
        self.update_button = tk.Button(self.root, text="Update Model", command=self.update_model)
        self.update_button.pack(pady=10)

        # Output frame
        self.output_frame = tk.Frame(self.root)
        self.output_frame.pack(pady=10)

        self.output_label = tk.Label(self.output_frame, text="Enter a value for prediction (x):")
        self.output_label.grid(row=0, column=0)

        self.prediction_entry = tk.Entry(self.output_frame)
        self.prediction_entry.grid(row=0, column=1)

        self.predict_button = tk.Button(self.output_frame, text="Predict", command=self.predict)
        self.predict_button.grid(row=0, column=2)

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 12))
        self.result_label.pack(pady=20)

    def add_data(self):
        try:
            x, y = map(float, self.input_entry.get().split(','))
            df = pd.read_csv(self.csv_file)
            new_data = pd.DataFrame({'x': [x], 'y': [y]})
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(self.csv_file, index=False)
            messagebox.showinfo("Info", "Data point added successfully!")
            self.input_entry.delete(0, tk.END)
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values separated by a comma.")

    def update_model(self):
        df = pd.read_csv(self.csv_file)
        if len(df) > 1:
            X = df[['x']].values
            y = df['y'].values
            self.model.fit(X, y)
            messagebox.showinfo("Info", "Model updated successfully!")
        else:
            messagebox.showerror("Error", "Not enough data to update the model. Please add more data points.")

    def predict(self):
        try:
            x = float(self.prediction_entry.get())
            prediction = self.model.predict([[x]])[0]
            self.result_label.config(text=f"Predicted value for x={x}: {prediction:.2f}")
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter a numeric value.")
        except:
            messagebox.showerror("Error", "Model not trained yet. Please add data and update the model.")

if __name__ == "__main__":
    root = tk.Tk()
    app = LifelongLearningApp(root)
    root.mainloop()
