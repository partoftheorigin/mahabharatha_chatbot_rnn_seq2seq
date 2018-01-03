"""
Creates Model Parameters
TensorFlow 1.4
"""

import time

import numpy as np
import tensorflow as tf

import config_parameters


class BotModel(object):
    def __init__(self, forward_only, batch_size):
        """forward_only: no backward pass
        """
        print('Initialize new model')
        self.fw_only = forward_only
        self.batch_size = batch_size

    def _create_placeholders(self):
        # Feeds : inputs(placeholders)
        print('Create placeholders')
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in range(config_parameters.BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in range(config_parameters.BUCKETS[-1][1] + 1)]
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(config_parameters.BUCKETS[-1][1] + 1)]

        # Targets : decoder inputs 
        self.targets = self.decoder_inputs[1:]

    def _inference(self):
        print('Inference, SampleSoftmax')

        if config_parameters.NUM_SAMPLES > 0 and config_parameters.NUM_SAMPLES < config_parameters.DEC_VOCAB:
            w = tf.get_variable('proj_w', [config_parameters.HIDDEN_SIZE, config_parameters.DEC_VOCAB])
            b = tf.get_variable('proj_b', [config_parameters.DEC_VOCAB])
            self.output_projection = (w, b)

        def sampled_loss(logits, labels):  # labels, inputs
            labels = tf.reshape(labels, [-1, 1])
            local_w_t = tf.cast(tf.transpose(w), tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inputs = tf.cast(logits, tf.float32)
            return tf.cast(
                tf.nn.sampled_softmax_loss(
                    weights=tf.transpose(w),
                    biases=local_b,
                    labels=labels,
                    inputs=local_inputs,
                    num_sampled=config_parameters.NUM_SAMPLES,
                    num_classes=config_parameters.DEC_VOCAB), tf.float32)

        self.softmax_loss_function = sampled_loss

    def _create_loss(self):
        print('Creating loss...')
        start = time.time()

        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            def single_cell():
                return tf.contrib.rnn.GRUCell(config_parameters.HIDDEN_SIZE)

            cell = single_cell()
            if config_parameters.NUM_LAYERS > 1:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(config_parameters.NUM_LAYERS)])
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols=config_parameters.ENC_VOCAB,
                num_decoder_symbols=config_parameters.DEC_VOCAB,
                embedding_size=config_parameters.HIDDEN_SIZE,
                output_projection=self.output_projection,
                feed_previous=do_decode)

        if self.fw_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets,
                self.decoder_masks,
                config_parameters.BUCKETS,
                lambda x, y: _seq2seq_f(x, y, True),
                softmax_loss_function=self.softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection:
                for bucket in range(len(config_parameters.BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output,
                                                      self.output_projection[0]) + self.output_projection[1]
                                            for output in self.outputs[bucket]]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets,
                self.decoder_masks,
                config_parameters.BUCKETS,
                lambda x, y: _seq2seq_f(x, y, False),
                softmax_loss_function=self.softmax_loss_function)
        print('Time:', time.time() - start)

    def _creat_optimizer(self):
        print('Create optimizer...')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.fw_only:
                self.optimizer = tf.train.GradientDescentOptimizer(config_parameters.LR)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                start = time.time()
                for bucket in range(len(config_parameters.BUCKETS)):
                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket],
                                                                              trainables),
                                                                 config_parameters.MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables),
                                                                         global_step=self.global_step))
                    print('Creating opt for bucket {} took {} seconds'.format(bucket, time.time() - start))
                    start = time.time()

    def _create_summary(self):
        pass

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._creat_optimizer()
        self._create_summary()
