import os
import datetime
import tensorflow as tf
from data_helper import *

# State which model to use here
from vdcnn import VDCNN

# Parameters settings
# Data loading params

# Model Hyperparameters
tf.flags.DEFINE_integer("sequence_max_length", 1024, "Sequence Max Length (default: 1024)")  # 1014
tf.flags.DEFINE_string("downsampling_type", "maxpool", "Options: maxpool, k-maxpool or linear (default: 'maxpool')")
tf.flags.DEFINE_integer("depth", 9, "Depth for VDCNN, use either 9, 17, 29 or 49 (default: 9)")
tf.flags.DEFINE_boolean("use_he_uniform", True, "Initialize embedding lookup with he_uniform (default: True)")
tf.flags.DEFINE_boolean("optional_shortcut", False, "Use optional shortcut (default: False)")
tf.flags.DEFINE_float("keep_prob", 1.0, "Try 0.6 or 0.7")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 1e-2, "Starter Learning Rate (default: 1e-2)")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_boolean("enable_tensorboard", True, "Enable Tensorboard (default: True)")

FLAGS = tf.flags.FLAGS

# print("Parameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr, value))
# print("")


def train_vdcnn(dataset_path, model_file, classes_weights=tf.constant([1.0, 1.0])):
    tf.reset_default_graph()
    max_accuracy = 0
    restore = True
    results_output = dataset_path + 'vdcnn_res.txt'

    # print("Loading data...")
    data_helper = DataHelper(sequence_max_length=FLAGS.sequence_max_length)
    train_data, train_label, test_data, test_label = data_helper.load_dataset(dataset_path)
    num_batches_per_epoch = int((len(train_data) - 1) / FLAGS.batch_size) + 1
    # print("Loading data succees...")

    # ConvNet
    acc_list = [0]
    sess = tf.Session()
    cnn = VDCNN(num_classes=train_label.shape[1],
                depth=FLAGS.depth,
                sequence_max_length=FLAGS.sequence_max_length,
                downsampling_type=FLAGS.downsampling_type,
                use_he_uniform=FLAGS.use_he_uniform,
                optional_shortcut=FLAGS.optional_shortcut,
                keep_prob=FLAGS.keep_prob,
                classes_weights = classes_weights)

    # Optimizer and LR Decay
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                                   FLAGS.num_epochs * num_batches_per_epoch, 0.95, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        gradients, variables = zip(*optimizer.compute_gradients(cnn.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 7.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

    if restore:
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, os.path.join(os.getcwd(), model_file))
    else:
        sess.run(tf.global_variables_initializer())

    with open(results_output, 'a') as f:
        f.write("================================\n  New round of training \n ================================")
        f.write(str(FLAGS.optional_shortcut) + ' ' + str(FLAGS.keep_prob) + '\n')

    # Train Step and Test Step
    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {cnn.input_x: x_batch,
                     cnn.input_y: y_batch,
                     cnn.is_training: True}
        _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        with open(results_output, 'a') as f:
                f.write("{}: Step {}, Epoch {}, Loss {:g}, Acc {:g}\n".format(time_str, step, int(step//num_batches_per_epoch)+1, loss, accuracy))

    def test_step(x_batch, y_batch):
        """
        Evaluates model on a dev set
        """
        feed_dict = {cnn.input_x: x_batch,
                     cnn.input_y: y_batch,
                     cnn.is_training: False}
        loss, preds = sess.run([cnn.loss, cnn.predictions], feed_dict)
        return preds, loss


    # Generate batches
    train_batches = data_helper.batch_iter(list(zip(train_data, train_label)), FLAGS.batch_size, FLAGS.num_epochs)

    # Training loop. For each batch...
    for train_batch in train_batches:
        x_batch, y_batch = zip(*train_batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)

        # Testing loop
        if current_step % FLAGS.evaluate_every == 0:
            with open(results_output, 'a') as f:
                f.write("\nEvaluation:\n")
            i = 0
            index = 0
            sum_loss = 0
            test_batches = data_helper.batch_iter(list(zip(test_data, test_label)), FLAGS.batch_size, 1, shuffle=False)
            y_preds = np.ones(shape=len(test_label), dtype=np.int)
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                preds, test_loss = test_step(x_test_batch, y_test_batch)
                sum_loss += test_loss
                res = np.absolute(preds - np.argmax(y_test_batch, axis=1))
                y_preds[index:index + len(res)] = res
                i += 1
                index += len(res)
            time_str = datetime.datetime.now().isoformat()
            acc = np.count_nonzero(y_preds == 0) / len(y_preds)
            acc_list.append(acc)
            with open(results_output, 'a') as f:
                if acc > max_accuracy:
                    max_accuracy = acc
                    saver = tf.train.Saver(tf.global_variables())
                    saver.save(sess, model_file)
                    f.write("New best model is at step " + str(current_step) + "\n")
                f.write("{}: Evaluation Summary, Loss {:g}, Acc {:g}\n".format(time_str, sum_loss/i, acc))
                f.write("{}: Current Max Acc {:g} in Iteration {}\n".format(time_str, max(acc_list), int(acc_list.index(max(acc_list))*FLAGS.evaluate_every)))


def predict_label_vdcnn(model_file, unlabeled_file, resulting_labels, num_classes,
                        classes_weights=tf.constant([1.0, 1.0])):
    tf.reset_default_graph()
    # print("Loading data...")
    data_helper_cl = DataHelper(sequence_max_length=FLAGS.sequence_max_length)
    unlabeled_data = data_helper_cl.load_unlabeled_data(unlabeled_file)
    # print("Loading data succees...")

    # ConvNet
    sess = tf.Session()
    cnn = VDCNN(num_classes=num_classes,
                depth=FLAGS.depth,
                sequence_max_length=FLAGS.sequence_max_length,
                downsampling_type=FLAGS.downsampling_type,
                use_he_uniform=FLAGS.use_he_uniform,
                optional_shortcut=FLAGS.optional_shortcut,
                keep_prob=FLAGS.keep_prob,
                classes_weights=classes_weights)

    def prediction_step(x_batch):
        feed_dict = {cnn.input_x: x_batch,
                     cnn.is_training: False}
        softmax_output = sess.run([cnn.softmax_output], feed_dict)
        return softmax_output

    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    # Labels prediction
    unlabeled_batches = data_helper_cl.batch_iter(unlabeled_data, FLAGS.batch_size, 1, shuffle=False)
    for unlabeled_batch in unlabeled_batches:
        softmax_output = prediction_step(unlabeled_batch)
        with open(resulting_labels, 'a') as f:
            for row in softmax_output[0]:
                for item in row:
                    if item == row[-1]:
                        f.write(str(item))
                    else:
                        f.write(str(item) + ",")
                f.write("\n")


def run(datasets, size_folders, balanced, class_nums):
    model_file = 'vdcnn_model.txt'
    unlabeled_file = 'unlabeled.csv'
    labels_file = 'predicted_labels.csv'
    for dataset in datasets:
        for size_folder in size_folders:
            if balanced:
                train_vdcnn(dataset + size_folder, dataset + size_folder + model_file)
                predict_label_vdcnn(dataset + size_folder + model_file, dataset + size_folder + unlabeled_file,
                                    dataset + size_folder + labels_file, class_nums)
            else:
                train_vdcnn(dataset + size_folder, dataset + size_folder + model_file,
                            classes_weights=tf.constant([13.0, 1.0]))
                predict_label_vdcnn(dataset + size_folder + model_file, dataset + size_folder + unlabeled_file,
                                    dataset + size_folder + labels_file, class_nums,
                                    classes_weights=tf.constant([13.0, 1.0]))


def main():
    os.chdir('vdcnn/')
    classes_num = [4, 5, 2, 10, 5, 2]
    datasets = ['ag_news_csv/', 'amazon_review_full_csv/', 'amazon_review_polarity_csv/', 'yahoo_answers_csv/',
                'yelp_review_full_csv/', 'yelp_review_polarity_csv/']
    size_folders = ['data_5000/', 'data_50000/']
    run(datasets, size_folders, True, classes_num)

    classes_num = [2]
    datasets = ['twitter_csv/']
    size_folders = ['data_5000/']
    run(datasets, size_folders, False, classes_num)
    os.chdir('../')


if __name__ == '__main__':
    main()