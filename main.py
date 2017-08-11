# -*- encoding: utf-8 -*-
# Created by han on 17-7-8

from models import *
from data_loader import DataLoader
from evaluate import *


def main():
    data_loader = DataLoader('./data', multi_ins=False)
    cnn_setting = CNNSetting()
    cnn_model = CNN(data_loader.wordembedding, cnn_setting)
    print cnn_model.model_name

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        test_data = data_loader.get_test_data()
        for i in range(100):
            j = 0
            batches = data_loader.get_train_batches(batch_size=1024)
            for batch in batches:
                j += 1
                loss = cnn_model.fit(session, batch, dropout_keep_rate=0.5)
                _, c_label = cnn_model.predict(session, batch)
                if j % 100 == 0:
                    p, r, f1 = get_p_r_f1(c_label, batch.y)
                    print i, j, loss, p, r, f1
            test_loss, test_pred = cnn_model.predict(session, test_data)
            p, r, f1 = get_p_r_f1(test_pred, test_data.y)
            print i, test_loss, p, r, f1


if __name__ == '__main__':
    main()
