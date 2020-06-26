from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.linear_model import SGDClassifier
import os


def train_tfidf_ngrams(path):
    model = SGDClassifier(loss='log', verbose=1, class_weight='balanced')
    train_data = pd.read_csv(path + 'train.csv', header=None)
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
    X = vectorizer.fit_transform(train_data.iloc[:, -1].values.astype('U').tolist())
    y = train_data.iloc[:, 0].values.astype('U').tolist()
    return [model.fit(X, y), vectorizer]


def predict_label_tfidf_ngrams(path, model, vectorizer):
    unlabeled_data = pd.read_csv(path + 'unlabeled.csv', header=None)
    X = vectorizer.transform(unlabeled_data.iloc[:, -1].values.astype('U').tolist())
    pred_y = pd.DataFrame(model.predict_proba(X))
    pred_y.to_csv(path + 'predicted_labels.csv', index=False, header=False)


def run(datasets, size_folders):
    for dataset in datasets:
        for size_folder in size_folders:
            model, vectorizer = train_tfidf_ngrams(dataset + size_folder)
            predict_label_tfidf_ngrams(dataset + size_folder, model, vectorizer)


def main():
    os.chdir('ngrams_tfidf/')
    datasets = ['ag_news_csv/', 'amazon_review_full_csv/', 'amazon_review_polarity_csv/', 'yahoo_answers_csv/',
                'yelp_review_full_csv/', 'yelp_review_polarity_csv/']
    size_folders = ['data_5000/', 'data_50000/']
    run(datasets, size_folders)

    datasets = ['twitter_csv/']
    size_folders = ['data_5000/']
    run(datasets, size_folders)
    os.chdir('../')


if __name__ == '__main__':
    main()