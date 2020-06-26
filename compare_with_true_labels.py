import os
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


def join_true_predicted_labels(balanced):
    all_labels = pd.read_csv('true_labels.csv', header=None)
    predicted_labels = pd.read_csv('predicted_labels.csv', header=None)
    all_labels['predicted'] = predicted_labels.idxmax(axis=1).astype(dtype='int32')
    if balanced:
        all_labels['predicted'] = all_labels['predicted'] + 1
    all_labels.to_csv('true_predicted_labels.csv', index=False, header=False)


def compare_predicted_and_true_labels():
    all_labels = pd.read_csv('true_predicted_labels.csv', header=None)
    print(os.getcwd())
    print(f'Accuracy is: {accuracy_score(all_labels.iloc[:, 0], all_labels.iloc[:, 1])}')
    print(f'F1 score is: {f1_score(all_labels.iloc[:, 0], all_labels.iloc[:, 1], average="micro")}')


def run(datasets, size_folders, balanced):
    ratio_folders = ['ratio_2/', 'ratio_3/', 'ratio_4/']
    models = ['slp_fasttext/']  # 'ngrams_tfidf/', 'vdcnn/', 'slp_vdcnn/', 'old_fasttext/',
    for k in range(len(models)):
        for i in range(0, len(datasets)):
            for size_folder in size_folders:
                if k < 0:
                    os.chdir(models[k] + datasets[i] + size_folder)
                    join_true_predicted_labels(balanced)
                    compare_predicted_and_true_labels()
                    os.chdir('../../..')
                else:
                    for ratio_folder in ratio_folders:
                        os.chdir(models[k] + datasets[i] + size_folder + ratio_folder)
                        compare_predicted_and_true_labels()
                        os.chdir('../../../..')


def main():
    datasets = ['ag_news_csv/', 'amazon_review_full_csv/', 'amazon_review_polarity_csv/', 'yelp_review_polarity_csv/']
    size_folders = ['data_500/', 'data_5000/', 'data_50000/']
    run(datasets, size_folders, True)

    datasets = ['twitter_csv/']
    size_folders = ['data_500/', 'data_5000/']
    run(datasets, size_folders, False)


if __name__ == '__main__':
    main()
