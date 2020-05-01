## Comment to Explain Function

def read_CSV_datafile(filename):
    X = []
    y = []
    with open(filename,'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            X.append(row[0])
            if benchmark_dataset == 'AskUbuntu':
                y.append(intent_dict[row[1]])
            elif benchmark_dataset == 'Chatbot':
                y.append(intent_dict[row[1]])
            else:
                y.append(intent_dict[row[1]])
    return X,y
