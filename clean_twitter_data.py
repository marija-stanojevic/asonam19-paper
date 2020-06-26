import glob, os
import pandas as pd
import csv


def get_twitter_data():
    i = 0
    for file in glob.glob("*.csv"):
        print(file)
        header = 0
        data = []
        thrown = []
        with open(file, newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            try:
                for row in reader:
                        if len(row) > 2 and row[3] == 'Link':
                            header = row[4:29]
                        elif len(row) > 1 and row[1] == 'TWITTER':
                            data.append(row[4:29])
                        else:
                            thrown.append(row)
            except:
                print(row)
        with open('data/' + str(i) + '_data.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header)
            writer.writerows(data)
            i = i + 1


def merge_labels_and_clean():
    labeled_data = pd.DataFrame()
    unlabeled_data = pd.DataFrame()
    labels = pd.read_csv('guns_organisations.csv')
    for file in glob.glob("*data.csv"):
        data = pd.read_csv(file)
        l_data = data.merge(labels, left_on='Author ID', right_on='ID on twitter')
        l_data = l_data[['Label', 'Date(ET)', 'Time(ET)', 'Followers', 'Following', 'City', 'Province/State', 'Country',
                         'Bio', 'Contents']]
        data = data[['Date(ET)', 'Time(ET)', 'Followers', 'Following', 'City', 'Province/State', 'Country', 'Bio',
                     'Contents']]
        u_data = pd.concat([data, l_data[['Date(ET)', 'Time(ET)', 'Followers', 'Following', 'City', 'Province/State',
                                          'Country', 'Bio', 'Contents']]]).drop_duplicates(keep=False)
        labeled_data = pd.concat([labeled_data, l_data])
        unlabeled_data = pd.concat([unlabeled_data, u_data])

    labeled_data = labeled_data.sample(frac=1).reset_index(drop=True)
    train_data = labeled_data[0:(int)(len(labeled_data) * 0.6)]
    test_data = labeled_data[(int)(len(labeled_data) * 0.6):(int)(len(labeled_data) * 0.8)]
    cv_data = labeled_data[(int)(len(labeled_data) * 0.8):len(labeled_data)]
    train_data.to_csv('train.csv', index=False, header=False)
    test_data.to_csv('test.csv', index=False, header=False)
    cv_data.to_csv('cv.csv', index=False, header=False)
    unlabeled_data.to_csv('unlabeled.csv', index=False, header=False)


def main():
    os.chdir("twitter_guns_advocacy/data_5000/ratio_2")
    # get_twitter_data()
    merge_labels_and_clean()


if __name__ == '__main__':
    main()
