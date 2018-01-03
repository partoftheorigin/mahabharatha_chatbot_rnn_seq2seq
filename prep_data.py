import os
import re
import random
import config_parameters
import numpy as np
from nltk.tokenize import moses


def pre_process_subtitles(dir):
    os.mkdir('data')
    fh = open('data/clean_subs.txt', mode='w')
    for file in os.listdir(dir):
        f = open((dir + '/' + file), encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            line = line.encode('ascii', 'ignore').decode('ascii')
            if re.search('[a-zA-Z]', line):
                clean_line = line.strip("'\n'")
                fh.write(clean_line+'\n')
    return print('Subtitles Cleaned!')


def create_dataset(file):
    with open(file, 'r') as f:
        text_lines = f.read().splitlines()

    # Generate random indexes for test data for 16% data set
    test_index = random.sample(range(0, len(text_lines)), int(len(text_lines) / 6))

    # Take 2 lines each for encoder and 2 each for decoder from list for conversation link buildup
    train_enc = open('data/train_encoder', 'w')
    train_dec = open('data/train_decoder', 'w')
    test_enc = open('data/test_encoder', 'w')
    test_dec = open('data/test_decoder', 'w')

    # To take alternate values in encoder, decoder
    enc_dec_flag = 1

    for i in range(0, len(text_lines)-1, 2):
        if i in test_index:
            if enc_dec_flag == 0:
                test_enc.write(text_lines[i] + text_lines[i+1] + '\n')
                enc_dec_flag = 1
            else:
                test_dec.write(text_lines[i] + text_lines[i+1] + '\n')
                enc_dec_flag = 0
        else:
            if enc_dec_flag == 0:
                train_enc.write(text_lines[i] + text_lines[i+1] + '\n')
                enc_dec_flag = 1
            else:
                train_dec.write(text_lines[i] + text_lines[i+1] + '\n')
                enc_dec_flag = 0
    return print('Training/Testing data created!')


# Create Vocab for each decoder and encoder files
def create_conversation_vocab(dir):
    for file in os.listdir(dir):
        if 'train' in file:
            tokenizer = moses.MosesTokenizer()
            vocab = dict()

            f = open(dir + '/' + file, mode='r')
            for line in f.readlines():
                for token in tokenizer.tokenize(line.encode('utf-8')):
                    # Create Vocab Dictionary
                    if not token in vocab:
                        vocab[token] = 0
                    vocab[token] += 1

            sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)

            f_train = open('data/vocab.{}'.format(file[-7:]), 'w')
            f_train.write('<pad>' + '\n')
            f_train.write('<unk>' + '\n')
            f_train.write('<s>' + '\n')
            f_train.write('<\s>' + '\n')
            index = 4

            for word in sorted_vocab:
                if vocab[word] < config_parameters.THRESHOLD:
                    with open('config_parameters.py', 'a') as cf:
                        if file[-7:] == 'encoder':
                            cf.write('\n' + 'ENC_VOCAB = ' + str(index))
                        else:
                            cf.write('\n' + 'DEC_VOCAB = ' + str(index))
                    break
                f_train.write(str(word) + '\n')
                index += 1
    return print('Vocab Created!')


def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}

# Sentence as a combination of vocab ids
def sentence2id(vocab, line):
    tokenizer = moses.MosesTokenizer()
    return [vocab.get(token, vocab['<unk>']) for token in tokenizer.tokenize(line)]


def map_vocab(folder):
    os.mkdir(folder + '/mapped_vocab')
    for mode in ['encoder', 'decoder']:
        vocab_path = 'vocab.' + mode
        for data in ['train', 'test']:
            inp_f = open(folder + '/' + data + '_' + mode, 'r')
            out_f = open(folder + '/mapped_vocab/' + data + '_ids_' + mode, 'w')
            _, vocab = load_vocab(folder + '/' + vocab_path)

            lines = inp_f.read().splitlines()
            for line in lines:
                if mode == 'decoder':  # we only care about '<s>' and </s> in encoder
                    ids = [vocab['<s>']]
                else:
                    ids = []
                ids.extend(sentence2id(vocab, line))
                # ids.extend([vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)])
                if mode == 'decoder':
                    ids.append(vocab['<\s>'])
                out_f.write(' '.join(str(id_) for id_ in ids) + '\n')
    return print('Vocab Mapped!')


def gen_buckets(enc, dec): # encoder and decoder file generated from map_vocab

    encode_file = open('data/mapped_vocab/' + enc, 'r')
    decode_file = open('data/mapped_vocab/' + dec, 'r')

    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in config_parameters.BUCKETS]

    i = 0
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)

        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]

        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config_parameters.BUCKETS):

            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break

        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1

    return data_buckets


def _pad_input(input_, size):
    return input_ + [config_parameters.PAD_ID] * (size - len(input_))

def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = config_parameters.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == config_parameters.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


# if __name__ == '__main__':
    # pre_process_subtitles('dataset/mahabharat-subtitles')
    # create_dataset('data/clean_subs.txt')
    # create_conversation_vocab('data')
    # map_vocab('data')