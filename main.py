# -*- encoding: utf-8 -*-
# Created by han on 17-7-8

from models import *
from data_loader import DataLoader


def main():
    data_loader = DataLoader('./data', multi_ins=False)

    cnn_setting = CNNSetting()
    cnn_model = CNN(data_loader.wordembedding, cnn_setting)
    optimizer = cnn_model.get_optimizer()

    print cnn_model.model_name

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        cnn_s_feed_test = {cnn_model.input_words: data_loader.test_word,
                              cnn_model.input_pos1: data_loader.test_pos1,
                              cnn_model.input_pos2: data_loader.test_pos2,
                              cnn_model.input_labels: data_loader.test_y,
                              cnn_model.dropout_keep_prob: 1}
        for i in range(100):
            j = 0
            batches = data_loader.get_train_batches(batch_size=512)
            for batch in batches:
                j += 1
                cnn_s_dict = {cnn_model.input_words: batch['word'],
                              cnn_model.input_pos1: batch['pos1'],
                              cnn_model.input_pos2: batch['pos2'],
                              cnn_model.input_labels: batch['y'],
                              cnn_model.dropout_keep_prob: 0.5
                              }
                session.run(optimizer, feed_dict=cnn_s_dict)
                loss = session.run(cnn_model.model_loss, feed_dict=cnn_s_dict)
                if j % 100 == 0:
                    print i, j, loss
            test_loss = session.run(cnn_model.model_loss, feed_dict=cnn_s_feed_test)
            print i, test_loss




if __name__ == '__main__':
    main()