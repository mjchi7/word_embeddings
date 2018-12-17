import numpy as np 
import tensorflow as tf 
import time
import os
import nltk
nltk.download('brown')
from google.colab import output
from nltk.corpus import brown
from collections import OrderedDict, Counter
import pickle
from google.colab import files, drive
import util

    
# Note: Do a normalization across all word vector during each mini batch
# https://pdfs.semanticscholar.org/77e5/76c02792d7df5b102bb81d49df4b5382e1cc.pdf

def main():

    window_size = 6
    batch_size = 16
    model_name = 'skipgram-win12'
    ckpt_dir = '/content/gdrive/My Drive/files/{0}/{0}'.format(model_name)
    ckpt_dir = ckpt_dir + '-ep{}.ckpt'
    
    text, dictionary = util.nltk2data(brown, save_dict=False, remove_punc=True)
    train_inps, train_labels, eval_inps, eval_labels = util.data_dicer(text, dictionary, window_size, batch_size, chopoff=True, train_eval_split=0.8)
    # TODO: placeholders for input and output (label)
    vocab_size = len(dictionary.keys())
    embed_dim = 512
    num_sampled = 10
    num_true = 2 * window_size
    epochs = 1
    current_epoch = 1

    x_ph = tf.placeholder(tf.int32, shape=(None, ), name='x_ph')
    target_ph = tf.placeholder(tf.int32, shape=(None, num_true), name='target_ph')

    # TODO: construct embedding_layers USING VARIABLE SCOPE!
    with tf.variable_scope('skipgram', reuse=tf.AUTO_REUSE):
        embedding_layer = tf.get_variable('embedding_layer',
            shape=(vocab_size, embed_dim),
            dtype=tf.float32,
            initializer=tf.constant_initializer(np.random.randn(vocab_size, embed_dim)))

        output_weights = tf.get_variable('output_weight',
            shape=(vocab_size, embed_dim),
            dtype=tf.float32,
            initializer=tf.constant_initializer(np.random.randn(vocab_size, embed_dim)))

        output_biases = tf.get_variable('output_bias',
            shape=(vocab_size,),
            dtype=tf.float32,
            initializer=tf.zeros_initializer())

    # TODO: map the word using tf.nn.lookup
    # center_word shape = [batch_size, embed_dim]
    center_word = tf.nn.embedding_lookup(embedding_layer, x_ph)

    # TODO: Check which mode, either training or eval
    # TODO: Training, calculate NCE loss
    train_loss = tf.nn.nce_loss(weights=output_weights,
        biases=output_biases,
        labels=target_ph,
        inputs=center_word,
        num_true=num_true,
        num_sampled=num_sampled,
        num_classes=vocab_size
        )
    train_batch_loss = tf.reduce_mean(train_loss)
    opt = tf.train.AdamOptimizer().minimize(train_batch_loss)

    # TODO: eval, calculate manually? refers to tensorflow guide.
        # output_weights.shape = [vocab_size, dim]
        # center_word.shape = [batch_size, dim]
        # matmul, output_weights

    # projections = tf.matmul(center_word, tf.transpose(output_weights)) + output_biases
    # eval_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=projections, labels=tf.squeeze(target_ph))
    # eval_batch_loss = tf.reduce_mean(eval_loss)


    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        if current_epoch > 1:
            saver.restore(sess, ckpt_dir.format(current_epoch-1))
        start_time = time.time()
        print("="*80)
        print("="*80)
        print("Start time: {}".format(time.strftime('%Y/%m/%d, %H:%M%S', time.localtime(start_time))))
        print("="*80)
        print("="*80)
        for i in range(epochs):
            epoch_start_time = time.time()
            print("Epoch: ", i+1)

            epoch_loss = []
            # Do it over batches
            # 1. stats
            batch_number = 0
            seconds_per_batch = 0
            total_batch = len(train_inps)
            for batch_inps, batch_labels in zip(train_inps, train_labels):
                batch_start_time = time.time()
                batch_number = batch_number + 1
                left_over_time = round((total_batch + 1 - batch_number) * seconds_per_batch,3)
                output.clear(output_tags=('batch_print'), wait=True)
                with output.use_tags('batch_print'):
                    print("Current batch:{}/{}\tSeconds per batch: {}s, {}s left.".format(batch_number, total_batch+1, seconds_per_batch, left_over_time))
          
                feed_dict = {x_ph: batch_inps, target_ph: batch_labels}
                _, batch_loss_v = sess.run([opt, train_batch_loss], feed_dict=feed_dict)
                epoch_loss.append(batch_loss_v)
                batch_end_time = time.time()
                seconds_per_batch = round(batch_end_time - batch_start_time, 3)
                # TODO: averaging epoch_loss
            feed_dict = {x_ph: eval_inps, target_ph: eval_labels}
            [eval_loss_v] = sess.run([train_batch_loss], feed_dict=feed_dict)

            epoch_loss = np.mean(epoch_loss)
            # TODO: print train loss, print eval loss
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print("Epochs took: {}s".format(round(epoch_duration, 3)))
            print("Train Loss: {}\tEval Loss: {}".format(round(epoch_loss,3), round(eval_loss_v, 3)))

        end_time = time.time()
        duration = end_time - start_time
        print("="*80)
        print("="*80)
        print("End time: {}".format(time.strftime('%Y/%m/%d, %H:%M%S', time.localtime(end_time))))
        print("Duration: {}s".format(round(duration,2)))
        print("="*80)
        print("="*80)
        saver.save(sess, ckpt_dir.format(current_epoch))


if __name__ == '__main__':
    usedrive=True
    
    if usedrive:
        if not os.path.exists('/content/gdrive/My Drive'):
            print('Drive not mounted. Mounting now ...')
            drive.mount('/content/gdrive')
        else:
            print('Drive mounted, skipping mounting stage.')
    tf.reset_default_graph()
    main()