import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('FATS_OGLE_bin.dat', sep='\t')
features = ['Amplitude', 'Std', 'Period', 'Mean', 'MaxSlope', 'Meanvariance', 'LinearTrend']
y_all = np.array(df['Class'])
X_all = np.array(df[features])


def train_test_split(features, labels, proportion):
    training_indexes = np.random.rand(len(features)) < proportion
    return (features[training_indexes], features[~training_indexes],
        labels[training_indexes], labels[~training_indexes])

X_train, X_test, y_train, y_test = train_test_split(features=X_all, labels=y_all, proportion=0.9)


fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))
for i in range(7):
    ixs = i // 2, i % 2
    axes[ixs].set_title(f'{features[i]} histogram')
    axes[ixs].hist(X_train[:, i], alpha=0.5)
    axes[ixs].legend(['All classes'])
axes[3, 1].axis('off')
plt.show()

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15,15))
for i in range(7):
    ixs = i // 2, i % 2
    axes[ixs].set_title(f'{features[i]} histogram')
    for cls in range(6):
        axes[ixs].hist(X_train[y_train == cls][:, i], alpha=0.5)
    axes[ixs].legend([f'Class {i}' for i in range(6)])
axes[3, 1].axis('off')
plt.show()


from scipy.stats import norm


class GenericNaiveBayes:
    def __init__(self, distribution=norm):
        self.distribution = distribution
        if distribution != norm:
            raise NotImplementedError('Only normal distribution supported.')

    def predict(self, X):
        label = X.pop('y')
        if label not in self.K:
            raise Exception('Unknown label')
        values = np.zeros(self.F.size)
        for key in X.keys():
            i, = np.where(self.feature_names == key)
            values[i] = X[key]
        probs = self._predict(values)
        return probs[:, label][0]

    def predict_label(self, X):
        return np.argmax(self._predict(X), axis=1)

    def score(self, X, labels):
        return np.array(np.where((labels == self.predict_label(X)) == True)).size / labels.size

    def confussion_matrix(self, X, labels):
        predictions = self.predict_label(X)
        return pd.crosstab(labels, predictions, rownames=['Real class'], colnames=['Predicted class'])

    def plot_confussion_matrix(self, X, labels):
        cfm = self.confussion_matrix(X, labels)
        cfm_density = np.array([row / row.sum() for row in np.array(cfm)])
        cfms = [cfm, cfm_density]
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        for i in range(2):
            ax[i].matshow(cfms[i], cmap='binary')
            for (k, j), z in np.ndenumerate(cfm):
                ax[i].text(j, k, z, ha='center', va='center', color='green', fontsize='large', fontweight='750')
            ax[i].set_title('Confussion Matrix' + (' (as density)' if i else ''))
            ax[i].set_ylabel('Real class')
            ax[i].set_xlabel('Predited class')
        plt.show()


class NaiveBayes(GenericNaiveBayes):
    def _prepare(self, data_frame):
        column_names = list(data_frame)
        self.feature_names = np.array([i for i in column_names if i != 'Class'])
        self.y = np.array(data_frame['Class'])
        self.X = np.array(data_frame[self.feature_names])
        self.F = np.arange(self.X.shape[1])
        self.K = np.unique(self.y)
        self.means = np.zeros((self.K.size, self.F.size))
        self.stds = np.zeros((self.K.size, self.F.size))
        self.normals = np.array([[None for f in self.F] for k in self.K])
        self.priors = np.array([self.y[self.y == k].size / self.y.size for k in self.K])

    def fit(self, data_frame):
        self._prepare(data_frame)
        for k in self.K:
            for i in self.F:
                subject = self.X[self.y == k][:, i]
                self.means[k, i] = mean = subject.mean()
                self.stds[k, i] = std = subject.std()
                self.normals[k, i] = self.distribution(mean, std)

    def _predict(self, X):
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            raise Exception(f'model.fit(df) must be called before model.predict(sample).')
        if len(X.shape) == 1:
            X = np.array([X])
        if self.F.size != X.shape[1]:
            raise Exception('You should train the model with the same number of features as the prediction')

        probs = np.zeros((X.shape[0], self.K.size))
        for k in self.K:
            probs[:, k] = np.prod(np.array([self.normals[k, i].pdf(X[:, i]) for i in self.F]), axis=0)
        return probs * self.priors / probs.sum(axis = 0)


df_train, df_test, _, _ = train_test_split(features=df, labels=df, proportion=0.95)

X_testing = np.array(df_test[[i for i in list(df_test) if i != 'Class']])
y_testing = np.array(df_test['Class'])


classifier = NaiveBayes()
classifier.fit(df_train)


classifier.confussion_matrix(X_testing, y_testing)
classifier.plot_confussion_matrix(X_testing, y_testing)

classifier.score(X_testing, y_testing)

label = 5
example = {[i for i in list(df_test) if i != 'Class'][i]: X_testing[0][i] for i in range(len(X_testing[0]))}
example['y'] = label # class to get prob of
print(example)
p = classifier.predict(example)
print(f'P(y = {label} | x) = {p}')


get_ipython().run_line_magic('reset', '-f array')


class BayesianNaiveBayes(GenericNaiveBayes):
    def _prepare(self, data_frame):
        column_names = list(data_frame)
        self.feature_names = np.array([i for i in column_names if i != 'Class'])
        self.y = np.array(data_frame['Class'])
        self.X = np.array(data_frame[self.feature_names])
        self.F = np.arange(self.X.shape[1])
        self.K = np.unique(self.y)
        self.means = np.zeros((self.K.size, self.F.size))
        self.stds = np.zeros((self.K.size, self.F.size))
        self.parameter_normals = np.array([[[None, None] for f in self.F] for k in self.K])
        self.y_priors = np.array([self.y[self.y == k].size / self.y.size for k in self.K])

    def fit(self, data_frame):
        self._prepare(data_frame)
        for k in self.K:
            for i in self.F:
                subject = self.X[self.y == k][:, i]
                self.means[k, i] = mean = subject.mean()
                self.stds[k, i] = std = subject.std()
                n = len(subject)
                self.parameter_normals[k, i, 0] = self.distribution(mean, std / np.sqrt(n))
                self.parameter_normals[k, i, 1] = self.distribution(std, std / np.sqrt(2*(n - 1)))


    def _predict(self, X):
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            raise Exception(f'model.fit(df) must be called before model.predict(sample).')
        if len(X.shape) == 1:
            X = np.array([X])
        if self.F.size != X.shape[1]:
            raise Exception('You should train the model with the same number of features as the prediction')

        probs = np.zeros((X.shape[0], self.K.size))
        I = 25
        for j in range(I):
            for k in self.K:
                probs[:, k] += np.prod(np.array([self.distribution(self.parameter_normals[k, i, 0].rvs(),
                     self.parameter_normals[k, i, 1].rvs()).pdf(X[:, i]) for i in self.F]), axis=0) / I
        return probs * self.y_priors / probs.sum(axis = 0)



df_train, df_test, _, _ = train_test_split(features=df, labels=df, proportion=0.95)

X_testing = np.array(df_test[[i for i in list(df_test) if i != 'Class']])
y_testing = np.array(df_test['Class'])

classifier = BayesianNaiveBayes()
classifier.fit(df_train)


classifier.confussion_matrix(X_testing, y_testing)
classifier.plot_confussion_matrix(X_testing, y_testing)


classifier.score(X_testing, y_testing)

clnb = NaiveBayes()
clnb.fit(df_train)
clnb.score(X_testing, y_testing)


label = 3
example = {[i for i in list(df_test) if i != 'Class'][i]: X_testing[0][i] for i in range(len(X_testing[0]))}
example['y'] = label # class to get prob of
p = classifier.predict(example)
print(f'P(y = {label} | x) = {p}')


from scipy.stats import normaltest



X_all = np.array(df[[i for i in list(df) if i != 'Class']])
y_all = np.array(df['Class'])
pvalues = np.array([[normaltest(X_all[y_all == k][:, i])[1] for i in range(7)] for k in range(6)])


pvalues.mean()


ex_mean = classifier.means[0][0]
ex_std = classifier.stds[0][0]
a, b = classifier.parameter_normals[0, 0]


points = np.linspace(11.4, 12, num=100)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
axes[0].set_title('Example of mean distribution for class 0 and feature 0')
axes[0].plot(points, a.pdf(points))
axes[0].axvline(x=ex_mean, color='red')
axes[0].legend(['Distribution', 'Data obtained value'])

points = np.linspace(3.2, 3.7, num=100)
axes[1].set_title('Example of std distribution for class 0 and feature 0')
axes[1].plot(points, b.pdf(points))
axes[1].axvline(x=ex_std, color='green')
axes[1].legend(['Distribution', 'Data obtained value'])

plt.show()
