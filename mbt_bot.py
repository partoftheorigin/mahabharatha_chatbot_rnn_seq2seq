import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import sys
import time

import numpy as np
import tensorflow as tf
from flask import Flask

from bot_model import BotModel
import config_parameters
import prep_data

# For API Mode: Flask
app = Flask(__name__)


def _get_random_bucket(train_buckets_scale):
    """ Random bucket for a training sample """
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])


def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    """ check for all length on variables """
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_masks), decoder_size))


def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    """ one step in training."""
    encoder_size, decoder_size = config_parameters.BUCKETS[bucket_id]
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    # input : encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # output feed: depends on whether we do a backward step or not.
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                       model.gradient_norms[bucket_id],  # gradient norm.
                       model.losses[bucket_id]]  # loss for this batch.
    else:
        output_feed = [model.losses[bucket_id]]  # loss for this batch.
        for step in range(decoder_size):  # output logits.
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.


def _get_buckets():
    """ Dataset into buckets """
    test_buckets = prep_data.gen_buckets('test_ids_encoder', 'test_ids_decoder')
    Data_buckets = prep_data.gen_buckets('train_ids_encoder', 'train_ids_decoder')
    train_bucket_sizes = [len(Data_buckets[b]) for b in range(len(config_parameters.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    # list of increasing numbers from 0 to 1 that we'll use to select a bucket.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return test_buckets, Data_buckets, train_buckets_scale


def _get_skip_step(iteration):
    """ Training Steps """
    if iteration < 100:
        return 30
    return 100


def _check_restore_parameters(sess, saver):
    """ Checkpoint restore """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config_parameters.CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing Mahabharatha-BOT")


def _eval_test_set(sess, model, test_buckets):
    """ Evaluate on the test set. """
    for bucket_id in range(len(config_parameters.BUCKETS)):
        if len(test_buckets[bucket_id]) == 0:
            print("  Test: empty bucket %d" % (bucket_id))
            continue
        start = time.time()
        encoder_inputs, decoder_inputs, decoder_masks = prep_data.get_batch(test_buckets[bucket_id],
                                                                            bucket_id,
                                                                            batch_size=config_parameters.BATCH_SIZE)
        _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs,
                                   decoder_masks, bucket_id, True)
        print('Test bucket {}: loss {}, time {}'.format(bucket_id, step_loss, time.time() - start))


def train():
    """ Train """
    test_buckets, Data_buckets, train_buckets_scale = _get_buckets()

    model = BotModel(False, config_parameters.BATCH_SIZE)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Running session')
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)

        iteration = model.global_step.eval()
        total_loss = 0
        while True:
            skip_step = _get_skip_step(iteration)
            bucket_id = _get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = prep_data.get_batch(Data_buckets[bucket_id],
                                                                                bucket_id,
                                                                                batch_size=config_parameters.BATCH_SIZE)
            start = time.time()
            _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
            total_loss += step_loss
            iteration += 1

            if iteration % skip_step == 0:
                print('Iter {}: loss {}, time {}'.format(iteration, total_loss / skip_step, time.time() - start))
                start = time.time()
                total_loss = 0
                saver.save(sess, os.path.join(config_parameters.CPT_PATH, 'chatbot'), global_step=model.global_step)
                if iteration % (10 * skip_step) == 0:
                    # Run evals on development set and print their loss
                    _eval_test_set(sess, model, test_buckets)
                    start = time.time()
                sys.stdout.flush()


def _get_user_input():
    """ User Input """
    print("--> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


def _find_right_bucket(length):
    """ Bucket Based on Length """
    return min([b for b in range(len(config_parameters.BUCKETS))
                if config_parameters.BUCKETS[b][0] >= length])


def _construct_response(output_logits, inv_dec_vocab):
    """ Response for user input.
    @output_logits: the outputs from sequence to sequence wrapper.

    Greedy decoder - outputs are just argmaxes of output_logits.
    """
    print("Output Logits: ", output_logits[0])
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    print("Output Greedy: ", outputs)
    if config_parameters.EOS_ID in outputs:
        outputs = outputs[:outputs.index(config_parameters.EOS_ID)]
    # Print out sentence corresponding to outputs.
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])


# Function for Interactive Chat
def chat():
    _, enc_vocab = prep_data.load_vocab(os.path.join(config_parameters.PROCESSED_PATH, 'vocab.encoder'))
    inv_dec_vocab, _ = prep_data.load_vocab(os.path.join(config_parameters.PROCESSED_PATH, 'vocab.decoder'))

    model = BotModel(True, batch_size=1)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        output_file = open(os.path.join(config_parameters.PROCESSED_PATH, config_parameters.OUTPUT_FILE), 'a+')
        # Decode from standard input.
        print('After Output--------------------------------')
        max_length = config_parameters.BUCKETS[-1][0]
        print('MBT-BOT: Message Limit:', max_length)
        while True:
            line = _get_user_input()
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]
            if line == '':
                break
            output_file.write('INPUT : ' + line + '\n')
            # Get token-ids for the input sentence.
            token_ids = prep_data.sentence2id(enc_vocab, str(line))
            if (len(token_ids) > max_length):
                print('Max length :', max_length)
                line = _get_user_input()
                continue
            # Which bucket does it belong to?
            bucket_id = _find_right_bucket(len(token_ids))
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_masks = prep_data.get_batch([(token_ids, [])],
                                                                                bucket_id,
                                                                                batch_size=1)
            # Get output logits for the sentence.
            _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            response = _construct_response(output_logits, inv_dec_vocab)
            print(response)
            output_file.write('RESPONSE : ' + response + '\n')
        output_file.write('++++++++++++++++++++++++++++++++++++++\n')
        output_file.close()


# Function for Flask REST API
model_run = 0
saver = None
model = None


@app.route("/get/<string:query>")
def chat_api(query):
    global model_run
    global saver
    global model
    if model_run == 0:
        model = BotModel(True, batch_size=1)
        model.build_graph()

        saver = tf.train.Saver()
        model_run = 1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)

        _, enc_vocab = prep_data.load_vocab(os.path.join(config_parameters.PROCESSED_PATH, 'vocab.enc'))
        inv_dec_vocab, _ = prep_data.load_vocab(os.path.join(config_parameters.PROCESSED_PATH, 'vocab.dec'))

        max_length = config_parameters.BUCKETS[-1][0]
        print('GOT-BOT: Message Limit:', max_length)
        while True:
            # token-ids for the input sentence.
            token_ids = prep_data.sentence2id(enc_vocab, str(query))
            if (len(token_ids) > max_length):
                print('Max length :', max_length)
                continue
            # Bucket Search
            bucket_id = _find_right_bucket(len(token_ids))
            # 1-element batch
            encoder_inputs, decoder_inputs, decoder_masks = prep_data.get_batch([(token_ids, [])],
                                                                                bucket_id,
                                                                                batch_size=1)
            # Get output logits for the sentence.
            _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            response = _construct_response(output_logits, inv_dec_vocab)
            return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', choices={'train', 'chat', 'api'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()

    prep_data.make_dir(config_parameters.CPT_PATH)

    if args.m == 'train':
        train()
    elif args.m == 'chat':
        chat()
    elif args.m == 'api':
        app.run(host='localhost', port=5003)


if __name__ == '__main__':
    main()