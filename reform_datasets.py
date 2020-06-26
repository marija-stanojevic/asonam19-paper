import os
import pandas as pd

datasets = ['ag_news_csv/', 'amazon_review_full_csv/', 'amazon_review_polarity_csv/', 'yahoo_answers_csv/',
            'yelp_review_full_csv/', 'yelp_review_polarity_csv/']


dataset_sizes = [[5000, 1700], [50000, 17000]]


def split(data, train_no, cv_no):
    train_data = data[0:train_no]
    cv_data = data[train_no :train_no + cv_no]
    unlabeled_data = data[train_no + cv_no:]
    true_labels = unlabeled_data.iloc[:, 0]
    unlabeled_data = unlabeled_data.iloc[:, 1:]
    os.mkdir(f'data_{train_no}')
    os.chdir(f'data_{train_no}')
    for ratio in [2, 3, 4]:
        os.mkdir(f'ratio_{ratio}')
        os.chdir(f'ratio_{ratio}')
        train_data.to_csv(f'train.csv', index=False, header=False)
        cv_data.to_csv(f'cv.csv', index=False, header=False)
        true_labels.to_csv(f'true_labels.csv', index=False, header=False)
        unlabeled_data.to_csv(f'unlabeled.csv', index=False, header=False)
        os.chdir('../')
    os.chdir('../')


def main():
    for dataset in datasets:
        os.chdir(dataset)
        train_data = pd.read_csv('train.csv', header=None)
        test_data = pd.read_csv('test.csv', header=None)
        data = pd.concat([train_data, test_data])
        data = data.sample(frac=1).reset_index(drop=True)
        for size in dataset_sizes:
            split(data, size[0], size[1])
        os.chdir('../')

    os.chdir('twitter_csv/')
    data = pd.read_csv('train.csv', header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    split(data, dataset_sizes[0][0], dataset_sizes[0][1])
    os.chdir('../')


if __name__ == '__main__':
    main()
