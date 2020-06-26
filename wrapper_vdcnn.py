import numpy as np
import pandas as pd
import os
import train_test_vdcnn as tt
import tensorflow as tf


def wrapper(ratio, path, balanced, class_nums):
    model_file = 'vdcnn_model.txt'
    unlabeled_file = 'unlabeled.csv'
    labels_file = 'predicted_labels.csv'
    train_file = 'train.csv'
    true_labels_file = 'true_labels.csv'
    all_labels_file = 'true_predicted_labels.csv'
    runMore = True

    while runMore:
        if balanced:
            tt.train_vdcnn(path, path + model_file)
            tt.predict_label_vdcnn(path + model_file, path + unlabeled_file, path + labels_file, class_nums)
        else:
            tt.train_vdcnn(path, path + model_file, classes_weights=tf.constant([13.0, 1.0]))
            tt.predict_label_vdcnn(path + model_file, path + unlabeled_file, path + labels_file, class_nums,
                                   classes_weights=tf.constant([13.0, 1.0]))
        labels = pd.read_csv(path + labels_file, header=None)
        os.remove(path + labels_file)
        true_labels = pd.read_csv(path + true_labels_file, header=None)
        unlabeled_data = pd.read_csv(path + unlabeled_file, header=None)

        max_class = labels.idxmax(axis=1)
        temp = np.sort(labels.values)[:, -2:]
        labels['max'] = labels.max(axis=1)
        labels['second_max'] = np.transpose(temp[:, 0])
        if not balanced:
            labels['max_class'] = max_class
        else:
            labels['max_class'] = max_class + 1
        labels['calc_ratio'] = labels['max'] / labels['second_max']
        good_labels = labels.drop(labels[labels['calc_ratio'] < ratio].index)  # keep just significant class
        new_unlabeled_data = unlabeled_data.drop(unlabeled_data.index[good_labels.index])
        new_true_labels = true_labels.drop(true_labels.index[good_labels.index])

        if len(new_unlabeled_data) < 10 or len(good_labels) < 10:   # stop if unlabeled or predicted are too small
            runMore = False
            new_unlabeled_data = pd.DataFrame([])
            new_true_labels = pd.DataFrame([])
        else:
            labels = good_labels
            true_labels = true_labels.drop(true_labels.index[new_true_labels.index])
            unlabeled_data = unlabeled_data.drop(unlabeled_data.index[new_unlabeled_data.index])

        new_train_data = pd.concat([labels['max_class'], unlabeled_data], axis=1)
        new_train_data.to_csv(path + train_file, mode='a', index=False, header=False)
        new_all_labels = pd.concat([true_labels, labels['max_class']], axis=1)
        new_all_labels.to_csv(path + all_labels_file, mode='a', index=False, header=False)

        new_unlabeled_data.to_csv(path + unlabeled_file, index=False, header=False)
        new_true_labels.to_csv(path + true_labels_file, index=False, header=False)

        print(f"Number of quality predictions is: {len(labels)}\n")
    os.remove(path + true_labels_file)


def run(datasets, size_folders, balanced, classes_num):
    ratio_folders = ['ratio_2/', 'ratio_3/', 'ratio_4/']
    ratios = [2, 3, 4]
    for i in range(0, len(datasets)):
        for size_folder in size_folders:
            for j in range(len(ratio_folders)):
                wrapper(ratios[j], 'slp_vdcnn/' + datasets[i] + size_folder + ratio_folders[j], balanced,
                        classes_num[i])


def main():
    classes_num = [4, 5, 2, 10, 5, 2]
    datasets = ['ag_news_csv/', 'amazon_review_full_csv/', 'amazon_review_polarity_csv/', 'yahoo_answers_csv/',
                'yelp_review_full_csv/', 'yelp_review_polarity_csv/']
    size_folders = ['data_5000/', 'data_50000/']
    run(datasets, size_folders, True, classes_num)

    classes_num = [2]
    datasets = ['twitter_csv/']
    size_folders = ['data_5000/']
    run(datasets, size_folders, False, classes_num)


if __name__ == '__main__':
    main()
