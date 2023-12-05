import numpy as np

class Classifier:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}

    def train(self,X,Y):
        sample_count, feature_count = X.shape
        pseudo_count_value = 1e-10  
        # print(f" ther are  {feature_count} features and {sample_count} samples")
        self.classes = set(Y)
        num_classes = len(self.classes)
        # print(f" ther are  {num_classes}")

        # Calculate class probabilities - harccoded since only two classes
        self.class_probabilities = {0:0.5,1:0.5}

        # Calculate feature probabilities given each class
        #  all classes
        for class_ in self.classes:
            X_class_ = []
            for x in range(len(Y)):
                if Y[x] == class_:
                    X_class_.append(X[x])
            X_class_ = np.array(X_class_)

            # Calculate the probabilities of each feature 
            self.feature_probabilities[class_] = {}
            for feature_index in range(feature_count):
                #  in one sweep - calculate mean and std
                feature_values = X_class_[:, feature_index]
                # calculate mean and standard deviation
                mean,std = np.mean(feature_values), np.std(feature_values) + pseudo_count_value
                self.feature_probabilities[class_][feature_index] = (mean, std)

    def predict(self,test_array):
        predictions = []
        for test in test_array:
            probs = {}
            for class_ in self.classes:
                # originally  - prior
                probs[class_] = np.log(self.class_probabilities[class_])

                # Calculate the log probability 
                log_probability_of_features = np.sum(-0.5 * np.log(2 * np.pi * (self.feature_probabilities[class_][x][1] ** 2)) - ((test[x] - self.feature_probabilities[class_][x][0]) ** 2) / (2 * (self.feature_probabilities[class_][x][1] ** 2)) for x in range(len(test)))
                
                probs[class_] += log_probability_of_features

            # Select the class with the highest probability
            NB_classfication = max(probs, key=probs.get)
            predictions.append(NB_classfication)

        return predictions
    
    
