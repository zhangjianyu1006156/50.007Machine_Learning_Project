#import function from part1
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

file_path = 'Data/ES/train'  
training_data = read_training_data(file_path)

# for sentence in training_data:
#     for word, label in sentence:
#         print(f"Word: {word}, Label: {label}")
#     print()

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

# # Set k to 1
k = 1
emission_parameters = estimate_emission_parameters(training_data,k)
# for (word, tag), param in emission_parameters.items():
#     print(f"Word: {word}, Tag: {tag}, Emission Parameter: {param:.4f}")
# For all the datasets RU, ES, learn these parameters with train
es_train_path = 'Data/ES/train'
ru_train_path = 'Data/RU/train'

es_training_data = read_training_data(es_train_path)
ru_training_data = read_training_data(ru_train_path)

es_emission_parameters = estimate_emission_parameters(es_training_data,k)
ru_emission_parameters = estimate_emission_parameters(ru_training_data,k)

#print(es_emission_parameters)

# for (word, tag), param in es_emission_parameters.items():
#     print(f"Word: {word}, Tag: {tag}, Emission Parameter: {param:.4f}")

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

# es_predicted_labels = []
# ru_predicted_labels = []
# es_dev_in_path = 'Data/ES/dev.in'
# ru_dev_in_path = 'Data/RU/dev.in'
# es_dev_data = read_dev_in(es_dev_in_path)
# ru_dev_data = read_dev_in(ru_dev_in_path)

# for sentence in es_dev_data:
#     for word in sentence:
#         label = predict_sentiment(word, es_emission_parameters)
#         es_predicted_labels.append((word,label))

# for sentence in ru_dev_data:
#     for word in sentence:
#         label = predict_sentiment(word, ru_emission_parameters)
#         ru_predicted_labels.append((word,label))
        
# print(es_predicted_labels)

# Write predicted labels to output files
def write_predicted_labels_to_file(predicted_labels, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        for word, tag in predicted_labels:
            file.write(f"{word} {tag}\n")

# # Write predicted labels to output files for ES dataset
# es_output_path = 'Data/ES/dev.p1.out'
# write_predicted_labels_to_file(es_predicted_labels, es_output_path)
 
# # Write predicted labels to output files for RU dataset
# ru_output_path = 'Data/RU/dev.p1.out'
# write_predicted_labels_to_file(ru_predicted_labels, ru_output_path)
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

# Read gold-standard outputs
es_gold_standard_outputs = read_gold_standard_outputs('Data/ES/dev.out')
ru_gold_standard_outputs = read_gold_standard_outputs('Data/RU/dev.out')

# print(es_gold_standard_outputs)
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

# # Calculate scores for ES dataset
# es_precision, es_recall, es_f_score = calculate_scores(es_predicted_labels, es_gold_standard_outputs)
# print("For ES Dataset:")
# print(f"Precision: {es_precision:.2f}")
# print(f"Recall: {es_recall:.2f}")
# print(f"F-score: {es_f_score:.2f}")

# # Calculate scores for RU dataset
# ru_precision, ru_recall, ru_f_score = calculate_scores(ru_predicted_labels, ru_gold_standard_outputs)
# print("\nFor RU Dataset:")
# print(f"Precision: {ru_precision:.2f}")
# print(f"Recall: {ru_recall:.2f}")
# print(f"F-score: {ru_f_score:.2f}")

#Estimates the transition parameters from the training set using MLE
def estimate_transition_data(training_data):
    transition_counts={}
    tags_counts={}
    for sentence in training_data:

        start_tag = sentence[0][1]
        transition_counts[('start', start_tag)] = transition_counts.get(('start', start_tag), 0) + 1
        tags_counts['start'] = tags_counts.get('start', 0) + 1
        
        stop_tag=sentence[-1][1]
        transition_counts[(stop_tag, 'stop')] = transition_counts.get((stop_tag, 'stop'), 0) + 1
        tags_counts[stop_tag] = tags_counts.get(stop_tag, 0) + 1

        for i in range(0,len(sentence)-1):
            first_tag,second_tag = sentence[i][1],sentence[i+1][1]
            transition_counts[(first_tag, second_tag)] = transition_counts.get((first_tag, second_tag), 0) + 1
            tags_counts[first_tag] = tags_counts.get(first_tag, 0) + 1
    
    transition_parameters={}
    for (first,second), count in transition_counts.items():
        tags_count = tags_counts[first]
        transition_parameters[(first,second)]=count/tags_count

    return transition_parameters

transition_parameters = estimate_transition_data(training_data)

# for (first, second), param in transition_parameters.items():
    
#     print(f"First: {first}, Second: {second}, Transition Parameter: {param:.4f}")
#Implement the Viterbi algorithm
def viterbi_algo(sentences, tags, trans_prob, emit_prob):
    n = len(sentences)
    count_tags = len(tags)

    #Used to store the maximum probability of each state at each timestep
    # Each dictionary element corresponds to a timestep
    # key:state,value:maximum probability of arriving at the state from the starting state
    pi = [{}]

    #Used to store the optimal path for each state at the current timestep
    #key:state,value:optimal path
    path = {}

    #Initialization
    # if "start" in tags:
    #     pi[0]["start"] = 1.0
    #     path["start"] = ["start"]
    
    for tag in tags:
        # if tag != "start":
        pi[0][tag] = 0.0  # Initialize with zero probability
        path[tag] = []

        if trans_prob.get(('start',tag),0.0)!=0:
            pi[0][tag] = trans_prob.get('start',tag)
            
    #Recurision
    

    for i in range(n):
        pi.append({})
        new_path = {}

        for u in tags:
            
            max_prob = -1
            max_prev_tag = None

            for v in tags:
                if i == 0:
                    prev_prob = 1.0
                else:
                    prev_prob = pi[i][v]
                trans = trans_prob.get((v,u),0.0)
                emit = emit_prob.get((sentences[i],u),0.0)
                prob = prev_prob * trans * emit

                if prob > max_prob:
                    max_prob = prob
                    max_prev_tag = v

            pi[i + 1][u] = max_prob
            new_path[u] = path[max_prev_tag] + [u] #update optimal path for current state in current timestep
        
        path = new_path #update the path after each timestep
        # print(path)

    #Final Step
    max_prob = -1
    max_prev_tag = None
    for v in tags:
        final_prob = pi[n][v] * trans_prob.get((v,'stop'),0.0)
        if final_prob > max_prob:
            max_prob = final_prob
            max_prev_tag = v

    final_max_tag = max_prev_tag

    return path[final_max_tag]

# For all the datasets RU, ES, learn these parameters with train

es_train_path = 'Data/ES/train'
ru_train_path = 'Data/RU/train'

es_training_data = read_training_data(es_train_path)
ru_training_data = read_training_data(ru_train_path)

es_emission_parameters = estimate_emission_parameters(es_training_data,k)
ru_emission_parameters = estimate_emission_parameters(ru_training_data,k)

es_transition_parameters = estimate_transition_data(es_training_data)
ru_transition_parameters = estimate_transition_data(ru_training_data)



def k_best_sequences(sentences, tags, trans_prob, emit_prob, k):
    n = len(sentences)

    # Initialize a list to store all sequences at each timestep
    all_sequences_list = [{} for _ in range(n + 1)]
    all_sequences_list[0]["start"] = [([], 1.0)]  # Initial starting state

    # Dynamic programming to find all sequences
    for i in range(n):
        all_sequences_list[i + 1] = {}
        for u in tags:
            all_sequences_list[i + 1][u] = []

            max_seq_probs = []  # To store sequence probabilities for ranking

            for seq, prev_prob in all_sequences_list[i].get(u, []):  # Use .get() to handle missing tags

                for v in tags:
                
                    trans = trans_prob.get((v, u), 1e-10)  # Handle zero probability transitions
                    emit = emit_prob.get((sentences[i], u), 0.0)
                    prob = prev_prob * trans * emit

                    max_seq_probs.append((seq + [u], prob))  # Append new sequence and its probability
                    print(max_seq_probs)
                    print(v for v in tags)

            # Sort the max_seq_probs based on probabilities and keep only top-k
            max_seq_probs.sort(key=lambda x: x[1], reverse=True)
            k_best_seq_probs = max_seq_probs[:k]

            for new_seq, prob in k_best_seq_probs:
                all_sequences_list[i + 1][u].append((new_seq, prob))

    # Final Step: Find the k-th best sequences at the last timestep
    k_best_output_sequences = []
    for v in tags:
        for seq, prob in all_sequences_list[n].get(v, []):
            if seq:  # Check if the sequence is non-empty
                k_best_output_sequences.append(seq)

    return k_best_output_sequences


k = 2  # Change this to the desired k value
es_dev_in = read_dev_in('Data/ES/dev.in')  # Read input sentences
ru_dev_in = read_dev_in('Data/RU/dev.in')

es_k_best_sequences = []
ru_k_best_sequences = []

# es tages
es_tags=()
for (first, second), param in es_transition_parameters.items():
    if first not in es_tags and first !='start':
        es_tags = es_tags+(first,)
    if second not in es_tags and second !='start':
        es_tags = es_tags+(second,)
print(es_tags)

# ru tags
ru_tags=()
for (first, second), param in ru_transition_parameters.items():
    if first not in ru_tags and first != 'start':
        ru_tags = ru_tags + (first,)
    if second not in ru_tags and second != 'start':
        ru_tags = ru_tags + (second,)
print(ru_tags)


for sentences in es_dev_in:
    k_best_seqs = k_best_sequences(sentences, es_tags, es_transition_parameters, es_emission_parameters, k)
    es_k_best_sequences.append(k_best_seqs)
    print("K-Best Sequences for ES:")
    for i, seqs in enumerate(es_k_best_sequences):
        print(f"Sequence {i}: {seqs}")



for sentences in ru_dev_in:
    k_best_seqs = k_best_sequences(sentences, ru_tags, ru_transition_parameters, ru_emission_parameters, k)
    ru_k_best_sequences.append(k_best_seqs)



    
# Write k-th best outputs to files
def write_k_best_outputs_to_file(output_sequences, file_path):
    with open(file_path, 'w') as f:
        for seqs in output_sequences:
            for seq in seqs:
                f.write(" ".join(seq) + "\n")
            f.write("\n")

write_k_best_outputs_to_file(es_k_best_sequences, 'Data/ES/dev.p3.8th.out')
write_k_best_outputs_to_file(ru_k_best_sequences, 'Data/RU/dev.p3.8th.out')

# Read gold-standard outputs
es_gold_standard_outputs = read_gold_standard_outputs('Data/ES/dev.out')
ru_gold_standard_outputs = read_gold_standard_outputs('Data/RU/dev.out')

# Calculate scores for ES dataset
print("Length of predicted sequences:", len(es_k_best_sequences))
print("Length of gold standard sequences:", len(es_gold_standard_outputs))

# for gold_sentence in es_gold_standard_outputs:
#     print("Gold Sentence:", gold_sentence)
# es_k_best_precision, es_k_best_recall, es_k_best_f_score = calculate_scores(es_k_best_sequences, es_gold_standard_outputs)
# ru_k_best_precision, ru_k_best_recall, ru_k_best_f_score = calculate_scores(ru_k_best_sequences, ru_gold_standard_outputs)

# print("For ES Dataset (8th Best):")
# print(f"Precision: {es_k_best_precision:.2f}")
# print(f"Recall: {es_k_best_recall:.2f}")
# print(f"F-score: {es_k_best_f_score:.2f}")

# print("For RU Dataset (8th Best):")
# print(f"Precision: {ru_k_best_precision:.2f}")
# print(f"Recall: {ru_k_best_recall:.2f}")
# print(f"F-score: {ru_k_best_f_score:.2f}")


