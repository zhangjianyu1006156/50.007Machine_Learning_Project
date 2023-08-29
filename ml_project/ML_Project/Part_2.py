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

# # Set k to 1
k = 1

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
# # Write predicted labels to output files for ES dataset
# es_output_path = 'Data/ES/dev.p1.out'
# write_predicted_labels_to_file(es_predicted_labels, es_output_path)
 
# # Write predicted labels to output files for RU dataset
# ru_output_path = 'Data/RU/dev.p1.out'
# write_predicted_labels_to_file(ru_predicted_labels, ru_output_path)

# Read gold-standard outputs
es_gold_standard_outputs = read_gold_standard_outputs('Data/ES/dev.out')
ru_gold_standard_outputs = read_gold_standard_outputs('Data/RU/dev.out')

#print(es_gold_standard_outputs)

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

#transition_parameters = estimate_transition_data(training_data)

#for (first, second), param in transition_parameters.items():   
#    print(f"First: {first}, Second: {second}, Transition Parameter: {param:.4f}")


es_train_path = 'Data/ES/train'
ru_train_path = 'Data/RU/train'

es_training_data = read_training_data(es_train_path)
ru_training_data = read_training_data(ru_train_path)

es_emission_parameters = estimate_emission_parameters(es_training_data,k)
ru_emission_parameters = estimate_emission_parameters(ru_training_data,k)

es_transition_parameters = estimate_transition_data(es_training_data)
ru_transition_parameters = estimate_transition_data(ru_training_data)

#print('ES transition parameters:')
#for (word, tag), param in es_transition_parameters.items():
#    print(f"Word: {word}, Tag: {tag}, Transition parameters: {param:.4f}")

#print('RU transition parameters:')
#for (word, tag), param in ru_transition_parameters.items():
#    print(f"Word: {word}, Tag: {tag}, Transition parameters: {param:.4f}")

#Run the Viterbi algorithm on ES
es_dev_in = read_dev_in('Data/ES/dev.in')
ru_dev_in = read_dev_in('Data/RU/dev.in')

es_tags=()
for (first, second), param in es_transition_parameters.items():
    if first not in es_tags and first!='start':
        es_tags=es_tags+(first,)
    if second not in es_tags and second!='start':
        es_tags=es_tags+(second,)

#print(es_tags)

es_predicted_tags=[]

for sentences in es_dev_in:
    path = viterbi_algo(sentences, es_tags, es_transition_parameters, es_emission_parameters)
    # print(path)
    for i in range(len(sentences)):
        es_predicted_tags.append((sentences[i],path[i]))

#print(es_predicted_tags)

ru_tags=()
for (first, second), param in ru_transition_parameters.items():
    if first not in ru_tags and first!='start':
        ru_tags=ru_tags+(first,)
    if second not in ru_tags and second!='start':
        ru_tags=ru_tags+(second,)

#print(es_tags)

ru_predicted_tags=[]

for sentences in ru_dev_in:
    path = viterbi_algo(sentences, ru_tags, ru_transition_parameters, ru_emission_parameters)
    # print(path)
    for i in range(len(sentences)):
        ru_predicted_tags.append((sentences[i],path[i]))

print(ru_predicted_tags)

#write in dev_out
write_predicted_labels_to_file(es_predicted_tags, 'Data/ES/dev.p2.out')



#Run the Viterbi algorithm on RU
ru_dev_in = read_dev_in('Data/RU/dev.in')

ru_tags=()
for (first, second), param in ru_transition_parameters.items():
    if first not in ru_tags and first!='start':
        ru_tags=ru_tags+(first,)
    if second not in es_tags and second!='start':
        ru_tags=ru_tags+(second,)
#print(ru_tags)

ru_predicted_tags=[]

for sentences in ru_dev_in:
    path = viterbi_algo(sentences, ru_tags, ru_transition_parameters, ru_emission_parameters)
    # print(path)
    for i in range(len(sentences)):
        ru_predicted_tags.append((sentences[i],path[i]))

#write in dev_out
write_predicted_labels_to_file(ru_predicted_tags, 'Data/RU/dev.p2.out')

# Read gold-standard outputs
es_gold_standard_outputs = read_gold_standard_outputs('Data/ES/dev.out')
ru_gold_standard_outputs = read_gold_standard_outputs('Data/RU/dev.out')

# Calculate scores for ES dataset
es_precision, es_recall, es_f_score = calculate_scores(es_predicted_tags, es_gold_standard_outputs)
print("For ES Dataset:")
print(f"Precision: {es_precision:.2f}")
print(f"Recall: {es_recall:.2f}")
print(f"F-score: {es_f_score:.2f}")

# Calculate scores for RU dataset
ru_precision, ru_recall, ru_f_score = calculate_scores(ru_predicted_tags, ru_gold_standard_outputs)
print("\nFor RU Dataset:")
print(f"Precision: {ru_precision:.2f}")
print(f"Recall: {ru_recall:.2f}")
print(f"F-score: {ru_f_score:.2f}")