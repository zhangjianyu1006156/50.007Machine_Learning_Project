# ML_Project Group 33
leader:
Zhang Jianyu 1006156

team member:
Liu Yu 1005621 
Wang Yongjie 1006155


## Installation
download or git clone from the GitHub

Pip install the requirements file [pip](https://pip.pypa.io/en/stable/) to install all the requirements.

```bash
pip install -r requirements.txt
```

After that, just run:
```bash
python Part_1.py
python Part_2.py
python Part_3.py
python Part_4.py
```

## Scroll all the way done once finished you should have the result shown below
This is part 2 screenshot
![image](https://github.com/Dr123Ake/ML_Project/assets/50765120/6e73bc91-fcf1-4edd-a846-e58c66226bdb)


## Contributing
Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)


# Sentiment Analysis System README

This repository contains Python scripts for a sentiment analysis system that performs sentiment tagging on text data using Hidden Markov Models (HMMs). The system estimates emission and transition parameters from training data and applies the Viterbi algorithm for tagging text data with sentiment labels.

## Part_1.py

This script performs sentiment tagging using emission parameters and a simple Maximum Likelihood Estimation (MLE) approach. It includes the following functionalities:


- **Data Reading and Processing:**

  - `read_training_data(file_path)`: Reads training data and processes it into a list of sentences and their respective sentiment labels.

  - `estimate_emission_parameters(training_data, k)`: Estimates emission parameters from training data, considering the emission of sentiment labels from words.

  - `read_dev_in(file_path)`: Reads development data (input) and processes it into a list of sentences.

  - `predict_sentiment(word, emission_parameters)`: Predicts the sentiment label for a given word using the emission parameters.

  - `write_predicted_labels_to_file(predicted_labels, output_path)`: Writes predicted sentiment labels to an output file.
  
  - `read_gold_standard_outputs(file_path)`: Reads gold-standard sentiment labels from a file and processes them.


- **Sentiment Tagging and Evaluation:**

  - Sentiment tagging is performed using emission parameters and predicted sentiment labels are written to output files.

  - `calculate_scores(predicted_labels, gold_standard_outputs)`: Calculates precision, recall, and F-score of the sentiment tagging system based on predicted and gold-standard sentiment labels.


- **Usage:**

  - The script estimates emission parameters using training data and applies sentiment tagging to development data.

  - Run the script to perform sentiment tagging and evaluation for both the ES and RU datasets.



## Part_2.py

This script extends the sentiment tagging system to incorporate transition parameters and the Viterbi algorithm. It includes the following functionalities:

- **Transition Parameter Estimation:**

  - `estimate_transition_data(training_data)`: Estimates transition parameters between sentiment labels using Maximum Likelihood Estimation (MLE) from training data.

- **Viterbi Algorithm:**

  - `viterbi_algo(sentences, tags, trans_prob, emit_prob)`: Implements the Viterbi algorithm to find the most likely sequence of sentiment labels given the input sentences, emission parameters, and transition parameters.

- **Sentiment Tagging with Transition Probabilities:**

  - The script reads training data, estimates emission and transition parameters, and applies the Viterbi algorithm for sentiment tagging.
  
  - Predicted sentiment labels are written to output files.

- **Usage:**

  - The script extends the sentiment tagging system to incorporate transition probabilities and applies the Viterbi algorithm for sentiment tagging.

  - Run the script to perform sentiment tagging using both emission and transition parameters.


## Running the Scripts:

1. Make sure you have Python installed on your system.

2. Adjust the `k` value and file paths as needed in the script.

3. Run the scripts to perform sentiment tagging and evaluation.
