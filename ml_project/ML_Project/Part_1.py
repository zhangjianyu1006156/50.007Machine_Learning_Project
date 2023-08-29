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


# Old version
def estimate_emission_parameters(training_data):
    # Key:(word, tag), Value:Number of times we see observation tag generated from word
    emission_counts = {}  
    # Key:tag, Value:Number of times we see the state u in the training set
    tag_counts = {}  
    
    for sentence in training_data:
        for word, tag in sentence:
            if (word, tag) in emission_counts:
                emission_counts[(word, tag)] += 1
            else:
                emission_counts[(word, tag)] = 1

            if tag in tag_counts:
                tag_counts[tag] += 1
            else:
                tag_counts[tag] = 1

    # Calculate emission parameters 
    emission_parameters = {}
    for (word, tag), count in emission_counts.items():
        tag_count = tag_counts[tag]
        emission_parameters[(word, tag)] = count/tag_count
    return emission_parameters


# New version
def estimate_emission_parameters(training_data,k):
    # Key:(word, tag), Value:Number of times we see observation tag generated from word
    emission_counts = {}  
    # Key:tag, Value:Number of times we see the state u in the training set
    tag_counts = {} 

    for sentence in training_data:
        for word, tag in sentence:
            if (word, tag) in emission_counts:
                emission_counts[(word, tag)] += 1
            else:
                emission_counts[(word, tag)] = 1

            if tag in tag_counts:
                tag_counts[tag] += 1
            else:
                tag_counts[tag] = 1

    # Calculate emission parameters
    emission_parameters = {}
    for (word, tag), count in emission_counts.items():
        tag_count = tag_counts[tag]

        if word == "#UNK#":
            emission_parameters[(word, tag)] = k / (tag_count + k)
        else:
            emission_parameters[(word, tag)] = count / (tag_count + k)

    return emission_parameters


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


def predict_sentiment(word, emission_parameters):
    highest_param = 0.0
    highest_parameter_tag = None
    
    for (w, tag), param in emission_parameters.items():
        if w == word and param > highest_param:
            highest_param = param
            highest_parameter_tag = tag
    
    return highest_parameter_tag


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

# Set k to 1
k = 1

# For all the datasets RU, ES, learn these parameters with train
es_train_path = 'Data/ES/train'
ru_train_path = 'Data/RU/train'
es_training_data = read_training_data(es_train_path)
ru_training_data = read_training_data(ru_train_path)

# Calculate the emission parameters for 2 dataset
es_emission_parameters = estimate_emission_parameters(es_training_data,k)
ru_emission_parameters = estimate_emission_parameters(ru_training_data,k)


#print(es_emission_parameters)
#print(ru_emission_parameters)

#print('ES emission parameters:')
#for (word, tag), param in es_emission_parameters.items():
#    print(f"Word: {word}, Tag: {tag}, Emission Parameter: {param:.4f}")

print('RU emission parameters:')
for (word, tag), param in ru_emission_parameters.items():
    print(f"Word: {word}, Tag: {tag}, Emission Parameter: {param:.4f}")

es_predicted_labels = []
ru_predicted_labels = []
es_dev_in_path = 'Data/ES/dev.in'
ru_dev_in_path = 'Data/RU/dev.in'
es_dev_data = read_dev_in(es_dev_in_path)
ru_dev_data = read_dev_in(ru_dev_in_path)

for sentence in es_dev_data:
    for word in sentence:
        label = predict_sentiment(word, es_emission_parameters)
        es_predicted_labels.append((word,label))

for sentence in ru_dev_data:
    for word in sentence:
        label = predict_sentiment(word, ru_emission_parameters)
        ru_predicted_labels.append((word,label))
        
#print(es_predicted_labels)
#print(ru_predicted_labels)

# Write predicted labels to output files for ES dataset
es_output_path = 'Data/ES/dev.p1.out'
write_predicted_labels_to_file(es_predicted_labels, es_output_path)
 
# Write predicted labels to output files for RU dataset
ru_output_path = 'Data/RU/dev.p1.out'
write_predicted_labels_to_file(ru_predicted_labels, ru_output_path)

# Read gold-standard outputs
es_gold_standard_outputs = read_gold_standard_outputs('Data/ES/dev.out')
ru_gold_standard_outputs = read_gold_standard_outputs('Data/RU/dev.out')

#print(es_gold_standard_outputs)
#print(ru_gold_standard_outputs)

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