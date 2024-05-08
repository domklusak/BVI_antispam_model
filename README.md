# Anti-Spam Email Classifier

This project is an anti-spam email classifier built using a dataset of 5000 spam and ham emails. The model is trained to distinguish between spam (unwanted email) and ham (non-spam email) using Python and machine learning techniques.

## Project Structure

- `main.py`: The main script that runs the classifier.
- `spam_detector.pkl`: The trained model file.
- `requirements.txt`: A list of libraries required to run the project.

## Dataset

The dataset used in this project consists of 5000 labeled emails divided into spam and ham categories. This data was split into training and testing sets to evaluate the model's performance.

## Installation

To run this project, you will need Python installed on your machine. You can download Python [here](https://www.python.org/downloads/). After installing Python, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://your-repository-url
   cd path-to-your-project
   ```
   
2. Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

To run the spam classifier, execute the following command from the project directory:
 ```bash
  pip install -r requirements.txt
 ```

## Saving the Trained Model

The trained model is saved using Python's `pickle` module. Here's the snippet used to export the model:

```python
import pickle

# Save the model
with open('spam_detector.pkl', 'wb') as file:
    pickle.dump(model, file)
```

You can use this exported model to classify new emails and integrate the classifier into other applications as an anti-spam system.
