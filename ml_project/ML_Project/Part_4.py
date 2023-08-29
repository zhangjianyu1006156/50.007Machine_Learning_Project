def read_training_data(file_path):
    training_data = []
    current_sentence = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1)
                if len(parts) >= 2:
                    word = parts[0]
                    label = parts[1]
                    current_sentence.append((word, label))
                else:
                    print(f"Invalid line: {line}")
            else:
                if current_sentence:
                    training_data.append(current_sentence)
                    current_sentence = []

    return training_data

def read_dev_in(file_path):
    dev_in_data = []
    current_sentence = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            word = line.strip()
            if word:
                current_sentence.append(word)
            else:
                if current_sentence:
                    dev_in_data.append(current_sentence)
                    current_sentence = []

    return dev_in_data

# Write predicted labels to output files
def write_predicted_labels_to_file(predicted_labels, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        for word, tag in predicted_labels:
            file.write(f"{word} {tag}\n")

def read_gold_standard_outputs(file_path):
    data = []
    current_sentence = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1)
                if len(parts) >= 2:
                    word = parts[0]
                    label = parts[1]
                    current_sentence.append((word, label))
                else:
                    print(f"Invalid line: {line}")
            else:
                if current_sentence:
                    data.append(current_sentence)
                    current_sentence = []

    return data

def calculate_scores(predicted_labels, gold_standard_outputs):
    correct_entities = 0
    predicted_entities = 0
    gold_entities = 0

    for (predicted_word, predicted_tag), gold_sentence in zip(predicted_labels, gold_standard_outputs):
        predicted_entities += len(predicted_word)
        gold_entities += len(gold_sentence)

        i = 0
        j = 0
        while i < len(predicted_word) and j < len(gold_sentence):
            # Handle the case where predicted_tag is None (neutral sentiment)
            if predicted_tag is None:
                predicted_tag = 'O'

            for gold_word, gold_tag in gold_sentence:
                if predicted_word == gold_word and predicted_tag == gold_tag:
                    correct_entities += 1

            # Handle transition from O to I-negative
            if predicted_tag.startswith('O') and gold_tag.startswith('I-negative'):
                j += 1
            elif predicted_tag.startswith('I-negative') and gold_tag.startswith('O'):
                i += 1
            else:
                i += 1
                j += 1

    precision = correct_entities / predicted_entities 
    recall = correct_entities / gold_entities 
    f_score = 2 * (precision * recall) / (precision + recall) 

    return precision, recall, f_score

# Define the function to extract features from a word
def extract_features(word):
    features = {}
    
    # Add features based on word length
    features['word_length'] = len(word)
    
    # Add features based on uppercase/lowercase
    features['is_uppercase'] = 1 if word.isupper() else 0
    features['is_lowercase'] = 1 if word.islower() else 0
    
    # You can add more features here, such as word embeddings, prefixes, suffixes, etc.
    # Example:
    features['prefix_2'] = word[:2]
    features['suffix_2'] = word[-2:]
    
    return features


# Define the function to estimate transition parameters using feature weights
def estimate_transition_parameters(training_data, feature_weights):
    transition_counts = {}
    tag_counts = {}
    
    for sentence in training_data:
        prev_tag = 'start'
        for _, tag in sentence:
            transition_counts[(prev_tag, tag)] = transition_counts.get((prev_tag, tag), 0) + 1
            tag_counts[prev_tag] = tag_counts.get(prev_tag, 0) + 1
            prev_tag = tag
    
    transition_parameters = {}
    for (prev_tag, tag), count in transition_counts.items():
        transition_parameters[(prev_tag, tag)] = count / tag_counts.get(prev_tag, 1)  # Avoid division by zero
    
    return transition_parameters


# Define the function to predict sentiment labels using CRF
def predict_sentiment_crf(sentence, feature_weights, transition_parameters):
    predicted_labels = []
    prev_tag = 'start'

    for word in sentence:
        features = extract_features(word)
        max_score = -float('inf')
        best_label = None

        for label in possible_labels:
            transition_score = transition_parameters.get((prev_tag, label), 0.0)
            emission_score = sum(feature_weights.get((word, label, feature), 0.0) * float(value) for feature, value in features.items())
            total_score = transition_score + emission_score

            if total_score > max_score:
                max_score = total_score
                best_label = label

        predicted_labels.append((word, best_label))
        prev_tag = best_label

    return predicted_labels


# Load training data
es_train_path = 'Data/ES/train'
ru_train_path = 'Data/RU/train'
es_training_data = read_training_data(es_train_path)
ru_training_data = read_training_data(ru_train_path)

# Train CRF model
feature_weights = {}  # Initialize feature weights

for sentence in es_training_data + ru_training_data:
    for word, label in sentence:
        features = extract_features(word)
        for feature, value in features.items():
            feature_weights[(word, label, feature)] = value

possible_labels = set(tag for _, tag in es_training_data[0])  # Assuming labels are consistent across datasets
es_transition_parameters = estimate_transition_parameters(es_training_data, feature_weights)
ru_transition_parameters = estimate_transition_parameters(ru_training_data, feature_weights)

# Load development data
es_dev_data = read_dev_in('Data/ES/dev.in')
ru_dev_data = read_dev_in('Data/RU/dev.in')

# Predict sentiment labels using CRF for ES
es_predicted_labels = []

for sentence in es_dev_data:
    es_predicted_labels.extend(predict_sentiment_crf(sentence, feature_weights, es_transition_parameters))

# Predict sentiment labels using CRF for RU
ru_predicted_labels = []

for sentence in ru_dev_data:
    ru_predicted_labels.extend(predict_sentiment_crf(sentence, feature_weights, ru_transition_parameters))

# Write predicted labels to output files for ES and RU datasets
write_predicted_labels_to_file(es_predicted_labels, 'Data/ES/dev.p4.out')
write_predicted_labels_to_file(ru_predicted_labels, 'Data/RU/dev.p4.out')

# Read gold-standard outputs for ES and RU datasets
es_gold_standard_outputs = read_gold_standard_outputs('Data/ES/dev.out')
ru_gold_standard_outputs = read_gold_standard_outputs('Data/RU/dev.out')

# Calculate scores for ES dataset
es_precision, es_recall, es_f_score = calculate_scores(es_predicted_labels, es_gold_standard_outputs)
print("For ES Dataset:")
print(f"Precision: {es_precision:.2f}")
print(f"Recall: {es_recall:.2f}")
print(f"F-score: {es_f_score:.2f}")

# Calculate scores for RU dataset
ru_precision, ru_recall, ru_f_score = calculate_scores(ru_predicted_labels, ru_gold_standard_outputs)
print("\nFor RU Dataset:")
print(f"Precision: {ru_precision:.2f}")
print(f"Recall: {ru_recall:.2f}")
print(f"F-score: {ru_f_score:.2f}")
