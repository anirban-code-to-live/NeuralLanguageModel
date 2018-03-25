from nltk.corpus import gutenberg
import collections
import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
import numpy as np
import os
import shutil
import re


def _read_characters(words):
    char_list = []
    char_count = 0
    for i in range(len(words)):
        # print(words[i])
        temp_word = words[i] + ' '
        # print(temp_word)
        char_count += len(temp_word)
        char_list.extend(temp_word)
    return char_list, char_count


def _build_vocab(words):
    char_list, char_count = _read_characters(words)
    counter = collections.Counter(char_list)
    char_count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    chars, _ = list(zip(*char_count_pairs))
    char_to_id = dict(zip(chars, range(len(chars))))
    id_to_char = dict((i, c) for i, c in enumerate(chars))
    return char_to_id, id_to_char


def _file_to_char_ids(words, char_to_id):
    char_list, _ = _read_characters(words)
    return [char_to_id[char] for char in char_list if char in char_to_id]


def _get_lstm_cell(num_units, keep_prob):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=0.0, state_is_tuple=True)
    cell = DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


def _calculate_perplexity(cost, num_steps):
    perplexity = np.exp(cost / num_steps)
    return perplexity


class CharacterModel:
    def __init__(self):
        guten_words = gutenberg.words()
        print(len(guten_words))
        char_to_id, id_to_char = _build_vocab(guten_words[0:100])
        # print(char_to_id)
        # print(id_to_char)
        vocab_size = len(char_to_id)
        print(vocab_size)

        # Server Settings
        BATCH_SIZE = 20
        NUM_STEPS = 30
        NUM_UNITS = 650
        NUM_LAYERS = 2
        KEEP_PROB = 0.35
        MAX_GRAD_NORM = 5
        LEARNING_RATE = 0.001
        NUM_ITERATIONS = 20000

        words_gutenberg = gutenberg.words('shakespeare-hamlet.txt')
        guten_words_count = len(words_gutenberg)
        train_data_size = int(0.6 * guten_words_count)
        validation_data_size = int(0.1 * guten_words_count)
        test_data_size = int(0.1 * guten_words_count)
        # print(guten_sents_count)
        print(train_data_size)
        # print(validation_data_size)
        # print(test_data_size)

        train_data = words_gutenberg[0:train_data_size]
        validation_data = words_gutenberg[train_data_size:train_data_size + validation_data_size]
        test_data = words_gutenberg[train_data_size + validation_data_size:train_data_size + validation_data_size + test_data_size]
        # #
        # # # print(len(train_data))
        # # # print(len(validation_data))
        # # # print(len(test_data))
        #
        # Local Settings
        # BATCH_SIZE = 1
        # NUM_STEPS = 35
        # NUM_UNITS = 200
        # NUM_LAYERS = 2
        # KEEP_PROB = 0.35
        # MAX_GRAD_NORM = 1
        # LEARNING_RATE = 0.001
        # NUM_ITERATIONS = 1000
        #
        # words_gutenberg = gutenberg.words()
        # # print(len(sentences_gutenberg))
        #
        # train_data = words_gutenberg[0:10000]
        # test_data = words_gutenberg[100:150]
        # validation_data = words_gutenberg[150:200]
        # print(len(train_data))
        # # print(train_data[3])

        char_list_train, char_count_train = _read_characters(train_data)
        char_to_id_train, id_to_char_train = _build_vocab(train_data)
        vocab_size = len(char_to_id_train)
        print('Vocab size :: ' + str(vocab_size))
        # print(word_to_id_train['<eos>'])
        train_data_id = _file_to_char_ids(train_data, char_to_id_train)

        # print(char_to_id_train)
        # print(id_to_char_train)
        print(len(train_data_id))

        # Start: Code for training
        train_data_tensor = tf.convert_to_tensor(train_data_id, name="train_data_tensor", dtype=tf.int64)
        train_data_len = tf.size(train_data_tensor)
        batch_len = train_data_len // BATCH_SIZE
        data = tf.reshape(train_data_tensor[0:BATCH_SIZE*batch_len], [BATCH_SIZE, batch_len])
        epoch_size = (batch_len - 1) // NUM_STEPS
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")

        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * NUM_STEPS],
                             [BATCH_SIZE, (i + 1) * NUM_STEPS])
        x.set_shape([BATCH_SIZE, NUM_STEPS])
        y = tf.strided_slice(data, [0, i * NUM_STEPS + 1],
                             [BATCH_SIZE, (i + 1) * NUM_STEPS + 1])
        y.set_shape([BATCH_SIZE, NUM_STEPS])
        # End: code for training
        #
        # # # Start : Code for validation
        # # validation_data_id = _file_to_word_ids(validation_data, word_to_id_train)
        # #
        # # print(validation_data_id)
        # #
        # # validation_data_tensor = tf.convert_to_tensor(validation_data_id, name="train_data_tensor", dtype=tf.int64)
        # # validation_data_len = tf.size(validation_data_tensor)
        # # batch_len = validation_data_len // BATCH_SIZE
        # # data = tf.reshape(validation_data_tensor[0:BATCH_SIZE * batch_len], [BATCH_SIZE, batch_len])
        # # epoch_size = (batch_len - 1) // NUM_STEPS
        # # assertion = tf.assert_positive(
        # #     epoch_size,
        # #     message="epoch_size == 0, decrease batch_size or num_steps")
        # #
        # # with tf.control_dependencies([assertion]):
        # #     epoch_size = tf.identity(epoch_size, name="epoch_size")
        # #
        # # i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        # # x = tf.strided_slice(data, [0, i * NUM_STEPS],
        # #                      [BATCH_SIZE, (i + 1) * NUM_STEPS])
        # # x.set_shape([BATCH_SIZE, NUM_STEPS])
        # # y = tf.strided_slice(data, [0, i * NUM_STEPS + 1],
        # #                      [BATCH_SIZE, (i + 1) * NUM_STEPS + 1])
        # # y.set_shape([BATCH_SIZE, NUM_STEPS])
        # #
        # # # end: Code for validation
        #
        # Start: code for sentence generation
        # SENTENCE_LENGTH = 100
        # seed_sentence = [char_to_id_train[' ']] * SENTENCE_LENGTH
        # x = tf.placeholder(dtype=tf.int64, shape=(None, None))
        # End: Code for sentence generation
        #
        # # print(x.shape, y.shape)
        # # print(x[0])
        # # print(y[0])
        # # print(x[1])
        # # print(y[1])
        #
        embedding = tf.get_variable("embedding", [vocab_size, NUM_UNITS])
        inputs = tf.nn.embedding_lookup(embedding, x)
        enc_cell = tf.contrib.rnn.MultiRNNCell([_get_lstm_cell(NUM_UNITS, KEEP_PROB) for _ in range(NUM_LAYERS)], state_is_tuple=True)
        initial_state = enc_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
        # state = initial_state
        # outputs = []
        # with tf.variable_scope("RNN"):
        #     for time_step in range(NUM_STEPS):
        #         if time_step > 0: tf.get_variable_scope().reuse_variables()
        #         (cell_output, state) = enc_cell(inputs[:, time_step, :], state)
        #         outputs.append(cell_output)
        # output = tf.reshape(tf.concat(outputs, 1), [-1, NUM_UNITS])

        inputs = tf.unstack(inputs, num=NUM_STEPS, axis=1)
        outputs, state = tf.nn.static_rnn(enc_cell, inputs, initial_state=initial_state)
        output = tf.reshape(tf.concat(outputs, 1), [-1, NUM_UNITS])

        softmax_w = tf.get_variable(
            "softmax_w", [NUM_UNITS, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [BATCH_SIZE, NUM_STEPS, vocab_size])

        prediction = tf.nn.softmax(logits)
        char_index = tf.argmax(prediction, 2)
        correct_pred = tf.equal(tf.argmax(prediction, 2), y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            y,
            tf.ones([BATCH_SIZE, NUM_STEPS]),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        cost = tf.reduce_sum(loss)
        final_state = state

        # if not is_training:
        #     return

        _lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), MAX_GRAD_NORM)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        _train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        _new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        _lr_update = tf.assign(_lr, _new_lr)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRAD_NORM)
        #
        # # Optimization
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)

        for epoch in range(NUM_ITERATIONS):
            loss_history = []

            _, train_loss, i_, x_, accuracy_ = sess.run((update_step, cost, i, x, accuracy))
            loss_history.append(train_loss)

            # print(i_)
            # print(x_[0])
            if (epoch + 1) % 2 == 0:
                print('Epoch: ' + str(epoch + 1), ' Loss: ' + str(np.mean(loss_history)))
                print(accuracy_)

        # # # Save model
        relative_path_parent_dir = os.path.dirname(__file__)
        # print(relative_path_parent_dir)
        if os.path.isdir(os.path.join(relative_path_parent_dir, 'char_model_data')):
            shutil.rmtree(os.path.join(relative_path_parent_dir, 'char_model_data'))
        export_dir = os.path.join(relative_path_parent_dir, 'char_model_data')
        saver = tf.train.Saver()
        saver.save(sess=sess, save_path=export_dir + '/session', write_meta_graph=False)
        # #
        # sess = tf.Session()
        # tf.train.start_queue_runners(sess=sess)
        # export_dir = os.path.join(relative_path_parent_dir, 'char_model_data')
        # # tf.saved_model.loader.load(sess, ['train_model'], export_dir)
        # tf.train.Saver().restore(sess, export_dir + '/session')
        # #
        # # for epoch in range(NUM_ITERATIONS):
        # #     loss_history = []
        # #     accuracy_history = []
        # #
        # #     validation_loss, i_, x_, pred_, accuracy_ = sess.run((cost, i, x, prediction, accuracy))
        # #     loss_history.append(validation_loss)
        # #     # accuracy_history.append(acc_)
        # #
        # #     # print(i_)
        # #     # print(x_[0])
        # #     # print(pred_.shape)
        # #     print(accuracy_)
        # #     if (epoch + 1) % 2 == 0:
        # #         print('Epoch: ' + str(epoch + 1), ' Loss: ' + str(np.mean(loss_history)))
        # #         # print('Epoch: ' + str(epoch + 1), ' Loss: ' + str(np.mean(accuracy_history)))
        #
        # # Code for sentence generation
        # x_input = np.zeros((BATCH_SIZE, NUM_STEPS))
        # characters_generated = []
        # for i in range(SENTENCE_LENGTH):
        #     for t, char in enumerate(seed_sentence):
        #         if t >= SENTENCE_LENGTH - NUM_STEPS:
        #             x_input[0, t - SENTENCE_LENGTH + NUM_STEPS] = char
        #     char_index_ = sess.run(char_index, {x: x_input})
        #     predicted_char = id_to_char_train[char_index_[0, -1]]
        #     characters_generated.append(predicted_char)
        #     seed_sentence = seed_sentence[1:] + list([char_index_[0, -1]])
        #     if predicted_char == ' ' and len(characters_generated) > 50:
        #         break
        #
        # sentence_generated = ''.join(characters_generated)
        # print(sentence_generated)