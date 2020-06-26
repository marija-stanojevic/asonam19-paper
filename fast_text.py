import fasttext
import os
import pandas as pd


def reform_train(data):
    train = []
    for index, row in data.iterrows():
        train.append('__label__' + str(row.iloc[0]) + ' ' + str(row.iloc[1]))
    return pd.DataFrame(train)


def train_fast_text(path, model_name, pretrained=None, dim=100):
    train_data = pd.read_csv(path + 'train.csv', header=None).iloc[:, [0, -1]]  # labels + texts
    reform_train(train_data).to_csv(path + 'hidden_train.csv', index=False, header=False)
    classifier = fasttext.supervised(path + 'hidden_train.csv', path + model_name, pretrained_vectors=pretrained, dim=dim)
    os.remove(path + 'hidden_train.csv')
    return classifier


def predict_label_fast_text(classifier, path, num_classes):
    unlabeled_data = pd.read_csv(path + 'unlabeled.csv', header=None).iloc[:, -1].values.astype('U').tolist()
    labels = classifier.predict_proba(unlabeled_data, k=num_classes)
    for i in range(len(labels)):
        labels[i].sort(key=lambda x: x[0])
        labels[i] = [i[1] for i in labels[i]]
    pd.DataFrame(labels).to_csv(path + 'predicted_labels.csv', index=False, header=False)


def run(datasets, size_folders, classes_num, pretrained=None, dim=100):
    for i in range(len(datasets)):
        for size_folder in size_folders:
            classifier = train_fast_text(datasets[i] + size_folder, 'fasttext_model.bin', pretrained, dim)
            predict_label_fast_text(classifier, datasets[i] + size_folder, classes_num[i])


def main():
    os.chdir('fasttext/')
    datasets = ['ag_news_csv/', 'amazon_review_full_csv/', 'amazon_review_polarity_csv/', 'yahoo_answers_csv/',
                'yelp_review_full_csv/', 'yelp_review_polarity_csv/']
    size_folders = ['data_5000/', 'data_50000/']
    classes_num = [4, 5, 2, 10, 5, 2]
    run(datasets, size_folders, classes_num, '../wiki-news-300d-1M.vec', 300)

    datasets = ['twitter_csv/']
    size_folders = ['data_5000/']
    classes_num = [2]
    run(datasets, size_folders, classes_num, '../wiki-news-300d-1M.vec', 300)
    os.chdir('../')


if __name__ == '__main__':
    main()
