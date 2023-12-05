import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}

    def train(self, X, y):
        sample_count, feature_count = X.shape
        print(f" ther are  {feature_count} features and {sample_count} samples")
        self.classes = set(y)
        num_classes = len(self.classes)
        print(f" ther are  {num_classes}")

        # Calculate class probabilities
        for class_ in self.classes:
            self.class_probabilities[class_] = np.sum(y == class_) / sample_count

        # Calculate feature probabilities given each class
        for class_ in self.classes:
            # Select samples belonging to the current class
            X_class_ = []
            for x in range(len(y)):
                if y[x] == class_:
                    X_class_.append(X[x])
            X_class_ = np.array(X_class_)

            # Calculate the probabilities of each feature given the class
            self.feature_probabilities[class_] = {}
            for feature_index in range(feature_count):
                feature_values = X_class_[:, feature_index]
                # calculate mean and standard deviation
                mean,std = np.mean(feature_values), np.std(feature_values) + 1e-10  
                self.feature_probabilities[class_][feature_index] = (mean, std)

    def predict(self, test_array):
        predictions = []
        for test in test_array:
            probs = {}
            for class_ in self.classes:
                class_prob = np.log(self.class_probabilities[class_])

                # Calculate the log probability of each feature given the class
                log_feature_probs = np.sum(
                    -0.5 * np.log(2 * np.pi * (self.feature_probabilities[class_][x][1] ** 2))
                    - ((test[x] - self.feature_probabilities[class_][x][0]) ** 2)
                    / (2 * (self.feature_probabilities[class_][x][1] ** 2))
                    for x in range(len(test))
                )

                probs[class_] = class_prob + log_feature_probs

            # Select the class with the highest probability
            NB_classfication = max(probs, key=probs.get)
            predictions.append(NB_classfication)

        return predictions
