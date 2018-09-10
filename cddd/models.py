import os
import numpy as np
import shutil
import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.contrib import seq2seq

#TODO: tanh as activation for dense layer after latent space?

class BaseModel(ABC):
    def __init__(self, mode, iterator, hparams):
        self.mode = mode
        self.iterator = iterator
        self.embedding_size = hparams.emb_size
        if mode in ["TRAIN", "EVAL","ENCODE"]:
            self.encode_vocabulary = {v: k for k, v in np.load(hparams.encode_vocabulary_file).item().items()}
            self.encode_voc_size = len(self.encode_vocabulary)
        if mode in ["TRAIN", "EVAL","DECODE"]:
            self.decode_vocabulary = {v: k for k, v in np.load(hparams.decode_vocabulary_file).item().items()}
            self.decode_vocabulary_reverse = {v: k for k, v in self.decode_vocabulary.items()}
            self.decode_voc_size = len(self.decode_vocabulary)
        self.char_embedding_size = hparams.char_embedding_size
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.save_dir = hparams.save_dir
        self.checkpoint_path = os.path.join(self.save_dir, 'model.ckpt')
        self.batch_size = hparams.batch_size
        self.rand_input_swap = hparams.rand_input_swap
        self.measures_to_log = {}
        if mode == "TRAIN":
            self.lr = hparams.lr
            self.lr_decay = hparams.lr_decay
            self.lr_decay_frequency = hparams.lr_decay_frequency
            self.lr_decay_factor = hparams.lr_decay_factor
        
        if mode == "DECODE":
            self.beam_width = hparams.beam_width
        
        if mode not in ["TRAIN", "EVAL", "ENCODE", "DECODE"]:
            raise ValueError("Choose one of following modes: TRAIN, EVAL, ENCODE, DECODE")
        
    def build_graph(self):
        if self.mode in ["TRAIN", "EVAL"]:
            with tf.name_scope("Input"):
                (self.input_seq,
                 self.shifted_target_seq,
                 self.input_len,
                 self.shifted_target_len,
                 self.target_mask,
                 encoder_emb_inp,
                 decoder_emb_inp) = self._input()
                    
            with tf.variable_scope("Encoder"):
                encoded_seq = self._encoder(encoder_emb_inp)
                
            with tf.variable_scope("Decoder"):
                logits = self._decoder(encoded_seq, decoder_emb_inp)
                self.prediction = tf.argmax(logits, axis=2, output_type=tf.int32)
                
            with tf.name_scope("Measures"):
                self.loss = self._compute_loss(logits)
                self.accuracy = self._compute_accuracy(self.prediction)
                self.measures_to_log["loss"] = self.loss
                self.measures_to_log["accuracy"] = self.accuracy
            
            if self.mode == "TRAIN":
                with tf.name_scope("Training"):
                    self._training()
                    
        if self.mode == "ENCODE":
            with tf.name_scope("Input"):
                self.input_seq = tf.placeholder(tf.int32, [None, None])
                self.input_len = tf.placeholder(tf.int32, [None])
                encoder_emb_inp = self._emb_lookup(self.input_seq)
                
            with tf.variable_scope("Encoder"):
                self.encoded_seq = self._encoder(encoder_emb_inp)
                
        if self.mode == "DECODE":
            # TODO: This will fail when decoder_embedding != encoder_embedding of trained modell
            self.decoder_embedding = tf.get_variable("char_embedding", [self.decode_voc_size, self.char_embedding_size])
            with tf.name_scope("Input"):
                self.encoded_seq = tf.placeholder(tf.float32, [None, self.embedding_size])

            with tf.variable_scope("Decoder"):
                self.output_ids = self._decoder(self.encoded_seq)

        self.saver_op = tf.train.Saver()
        
    def _input(self, with_features=False):
        with tf.device('/cpu:0'):
            if with_features:
                seq1, seq2, seq1_len, seq2_len, mol_features = self.iterator.get_next()
            else:
                seq1, seq2, seq1_len, seq2_len = self.iterator.get_next()
            if self.rand_input_swap:
                rand_val = tf.random_uniform([], dtype=tf.float32)    
                input_seq = tf.cond(tf.greater_equal(rand_val, 0.5), lambda: seq1, lambda: seq2)
                input_len = tf.cond(tf.greater_equal(rand_val, 0.5), lambda: seq1_len, lambda: seq2_len)
            else:
                input_seq = seq1
                input_len = seq1_len
            target_seq = seq2
            target_len = seq2_len
            
            shifted_target_len = tf.reshape(target_len, [tf.shape(target_len)[0]]) - 1
            shifted_target_seq = tf.slice(target_seq, [0, 1], [-1, -1])
            target_mask = tf.sequence_mask(shifted_target_len, dtype=tf.float32)
            target_mask = target_mask / tf.reduce_sum(target_mask)
            input_len = tf.reshape(input_len, [tf.shape(input_len)[0]])
            
        encoder_emb_inp, decoder_emb_inp = self._emb_lookup(input_seq, target_seq)
        if with_features:
            return input_seq, shifted_target_seq, input_len, shifted_target_len, target_mask, encoder_emb_inp, decoder_emb_inp, mol_features 
        else:
            return input_seq, shifted_target_seq, input_len, shifted_target_len, target_mask, encoder_emb_inp, decoder_emb_inp
            
    def _emb_lookup(self, input_seq, target_seq=None):
        self.encoder_embedding = tf.get_variable("char_embedding", [self.encode_voc_size, self.char_embedding_size])
        encoder_emb_inp = tf.nn.embedding_lookup(self.encoder_embedding, input_seq)
        if self.mode != "ENCODE":
            assert target_seq is not None
            if self.encode_vocabulary == self.decode_vocabulary:
                self.decoder_embedding = self.encoder_embedding
            else:
                self.decoder_embedding = tf.get_variable("char_embedding2", [self.decode_voc_size, self.char_embedding_size])
            decoder_emb_inp = tf.nn.embedding_lookup(self.decoder_embedding, target_seq)
            return encoder_emb_inp, decoder_emb_inp
        else:
            return encoder_emb_inp
        
    def _training(self):
        if self.lr_decay:
            self.lr = tf.train.exponential_decay(self.lr, 
                                                 self.global_step,
                                                 self.lr_decay_frequency,
                                                 self.lr_decay_factor,
                                                 staircase=True,)
        self.opt = tf.train.AdamOptimizer(self.lr, name='optimizer')
        grads = self.opt.compute_gradients(self.loss)
        grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
        self.train_step = self.opt.apply_gradients(grads, self.global_step)
        #self.measures_to_log["learning_rate"] = self.lr
    
    @abstractmethod
    def _encoder(self, encoder_emb_inp):
        raise NotImplementedError("Must override _encoder in child class")
    
    @abstractmethod
    def _decoder(self, encoded_seq, decoder_emb_inp=None):
        raise NotImplementedError("Must override _decoder in child class")

    def _compute_loss(self, logits):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.shifted_target_seq,
                                                                  logits=logits)
        loss = (tf.reduce_sum(crossent * self.target_mask))
        return loss
    
    def _compute_accuracy(self, prediction):
        right_predictions = tf.cast(tf.equal(prediction, self.shifted_target_seq), tf.float32)
        accuracy = (tf.reduce_sum(right_predictions * self.target_mask))
        return accuracy

    def train(self, sess):
        assert self.mode == "TRAIN"
        _, step = sess.run([self.train_step, self.global_step])
        return step
    
    def eval(self, sess):
        return sess.run(list(self.measures_to_log.values()))
    
    def idx_to_char(self, seq):
        return ''.join([self.decode_vocabulary_reverse[idx] for idx in seq
                        if idx not in [-1, self.decode_vocabulary["</s>"], self.decode_vocabulary["<s>"]]])

    def seq2emb(self, sess, input_seq, input_len):
        assert self.mode == "ENCODE"
        return sess.run(self.encoded_seq, {self.input_seq: input_seq,
                                           self.input_len: input_len})
    def emb2seq(self, sess, embedding, num_top):
        assert self.mode == "DECODE"
        output_seq = sess.run(self.output_ids, {self.encoded_seq: embedding})
        return [[self.idx_to_char(seq[:, i]) for i in range(num_top)] for seq in output_seq]
    
    def initilize(self, sess, overwrite_saves=False):
        assert self.mode == "TRAIN"
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print ('Create save file in: ', self.save_dir)
        elif overwrite_saves:
            shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)
        else:
            raise ValueError("Save directory %s already exist. Set overwrite_saves=True to overwrite it or set restore=True to restore!" %(self.save_dir))
        return sess.run(self.global_step)
    def restore(self, sess, restore_path=None):
        if restore_path is None:
            restore_path = self.checkpoint_path
        self.saver_op.restore(sess, restore_path)
        if self.mode == "TRAIN":
            step = sess.run(self.global_step)
            print("Restarting training at step %d" %(step))
            return step
    def save(self, sess):
        self.saver_op.save(sess, self.checkpoint_path)
        
class GRUSeq2Seq(BaseModel):
    def __init__(self, mode, iterator, hparams):
        super().__init__(mode, iterator, hparams)
        self.cell_size = hparams.cell_size
        self.reverse_decoding = hparams.reverse_decoding
        
    def _encoder(self, encoder_emb_inp):
        encoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]
        encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                           encoder_emb_inp,
                                                           sequence_length=self.input_len,
                                                           dtype=tf.float32,
                                                           time_major=False)
        emb = tf.layers.dense(tf.concat(encoder_state, axis=1),
                              self.embedding_size,
                              activation=tf.nn.tanh
                             )
        return embgit 

    def _decoder(self, encoded_seq, decoder_emb_inp=None):
        if self.reverse_decoding:
            self.cell_size = self.cell_size[::-1]
        decoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]
        decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell)
        decoder_cell_inital = tf.layers.dense(encoded_seq, sum(self.cell_size))
        decoder_cell_inital = tuple(tf.split(decoder_cell_inital, self.cell_size, 1))
        if self.mode != "DECODE":
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp,
                                                       sequence_length=self.shifted_target_len,
                                                       time_major=False)
            projection_layer = tf.layers.Dense(self.decode_voc_size, use_bias=False)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      helper,
                                                      decoder_cell_inital,
                                                      output_layer=projection_layer)
            outputs, output_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                         impute_finished=True,
                                                                         output_time_major=False)
            return outputs.rnn_output
        else:
            decoder_cell_inital = tf.contrib.seq2seq.tile_batch(decoder_cell_inital, self.beam_width)
            projection_layer = tf.layers.Dense(self.decode_voc_size, use_bias=False)
            start_tokens = tf.fill([tf.shape(encoded_seq)[0]], self.decode_vocabulary['<s>'])
            end_token = self.decode_vocabulary['</s>']
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=self.decoder_embedding,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=decoder_cell_inital,
                beam_width=self.beam_width,
                output_layer=projection_layer,
                length_penalty_weight=0.0)

            outputs, output_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                impute_finished=False,
                output_time_major=False,
                maximum_iterations = 1000
            )

            return outputs.predicted_ids
        
class NoisyGRUSeq2Seq(GRUSeq2Seq):
    def __init__(self, mode, iterator, hparams):
        super().__init__(mode, iterator, hparams)
        self.input_dropout = hparams.input_dropout
        self.emb_noise = hparams.emb_noise
        
    def _encoder(self, encoder_emb_inp):
        if (self.mode == "TRAIN") & (self.input_dropout > 0.0):
            max_time = tf.shape(encoder_emb_inp)[1]
            encoder_emb_inp = tf.nn.dropout(encoder_emb_inp, 1. - self.input_dropout, noise_shape=[self.batch_size, max_time, 1])
        encoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]
        encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                           encoder_emb_inp,
                                                           sequence_length=self.input_len,
                                                           dtype=tf.float32,
                                                           time_major=False)
        emb = tf.layers.dense(tf.concat(encoder_state, axis=1),
                              self.embedding_size
                             )
        if (self.mode == "TRAIN") & (self.emb_noise > 0.0):
            emb += tf.random_normal(shape=tf.shape(emb), mean=0.0, stddev=self.emb_noise, dtype=tf.float32)
        emb = tf.tanh(emb)
        return emb
    
class LSTMSeq2Seq(BaseModel):
    def __init__(self, mode, iterator, hparams):
        super().__init__(mode, iterator, hparams)
        self.cell_size = hparams.cell_size
        
    def _encoder(self, encoder_emb_inp):
        encoder_cell = [tf.nn.rnn_cell.LSTMCell(size) for size in self.cell_size]
        encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                           encoder_emb_inp,
                                                           sequence_length=self.input_len,
                                                           dtype=tf.float32,
                                                           time_major=False)
        encoder_state_c = [state.c for state in encoder_state]
        emb = tf.layers.dense(tf.concat(encoder_state_c, axis=1),
                              self.embedding_size,
                              activation=tf.nn.tanh
                             )
        return emb

    def _decoder(self, encoded_seq, decoder_emb_inp=None):
        decoder_cell = [tf.nn.rnn_cell.LSTMCell(size) for size in self.cell_size]
        decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell)
        initial_state_c_full = tf.layers.dense(encoded_seq, sum(self.cell_size))
        initial_state_c = tuple(tf.split(initial_state_c_full, self.cell_size, 1))
        initial_state_h_full = tf.zeros_like(initial_state_c_full)
        initial_state_h = tuple(tf.split(initial_state_h_full, self.cell_size, 1))
        decoder_cell_inital = tuple([tf.contrib.rnn.LSTMStateTuple(initial_state_c[i], initial_state_h[i]) for i in range(len(self.cell_size))])
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp,
                                                   sequence_length=self.shifted_target_len,
                                                   time_major=False)
        projection_layer = tf.layers.Dense(self.decode_voc_size, use_bias=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                  helper,
                                                  decoder_cell_inital,
                                                  output_layer=projection_layer)
        outputs, output_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                     impute_finished=True,
                                                                     output_time_major=False)
        return outputs.rnn_output

class Conv2GRUSeq2Seq(GRUSeq2Seq):
    def __init__(self, mode, iterator, hparams):
        super().__init__(mode, iterator, hparams)
        self.conv_hidden_size = hparams.conv_hidden_size
        self.kernel_size = hparams.kernel_size
        
    def _encoder(self, x):
        for i, size in enumerate(self.conv_hidden_size):
            x = tf.layers.conv1d(x, size, self.kernel_size[i], activation=tf.nn.relu, padding='SAME')
            if i+1 < len(self.conv_hidden_size):
                x = tf.layers.max_pooling1d(x, 3, 2, padding='SAME')
        x = tf.layers.conv1d(x, self.conv_hidden_size[-1], 1, activation=tf.nn.relu, padding='SAME')
        
        emb = tf.layers.dense(tf.reduce_mean(x, axis=1),
                              self.embedding_size,
                              activation=tf.nn.tanh
                             )
        return emb

    
    
class GRUSeq2SeqWithFeatures(GRUSeq2Seq):
    def __init__(self, mode, iterator, hparams):
        super().__init__(mode, iterator, hparams)
        self.num_features = hparams.num_features
    
    def build_graph(self):
        if self.mode in ["TRAIN", "EVAL"]:
            with tf.name_scope("Input"):
                (self.input_seq,
                 self.shifted_target_seq,
                 self.input_len,
                 self.shifted_target_len,
                 self.target_mask,
                 encoder_emb_inp,
                 decoder_emb_inp,
                 self.mol_features)  = self._input(with_features=True)
                    
            with tf.variable_scope("Encoder"):
                encoded_seq = self._encoder(encoder_emb_inp)
            with tf.variable_scope("Decoder"):
                sequence_logits = self._decoder(encoded_seq, decoder_emb_inp)
                self.sequence_prediction = tf.argmax(sequence_logits, axis=2, output_type=tf.int32)
            with tf.variable_scope("Feature_Regression"):
                feature_predictions = self._feature_regression(encoded_seq)
            with tf.name_scope("Measures"):
                self.loss_sequence, self.loss_features = self._compute_loss(sequence_logits, feature_predictions)
                self.loss = self.loss_sequence + self.loss_features
                self.accuracy = self._compute_accuracy(self.sequence_prediction)
                self.measures_to_log["loss"] = self.loss
                self.measures_to_log["accuracy"] = self.accuracy
            
            if self.mode == "TRAIN":
                with tf.name_scope("Training"):
                    self._training()
                    
        if self.mode == "ENCODE":
            with tf.name_scope("Input"):
                self.input_seq = tf.placeholder(tf.int32, [None, None])
                self.input_len = tf.placeholder(tf.int32, [None])
                encoder_emb_inp = self._emb_lookup(self.input_seq)
                
            with tf.variable_scope("Encoder"):
                self.encoded_seq = self._encoder(encoder_emb_inp)
                
        if self.mode == "DECODE":
            # TODO: This will fail when decoder_embedding != encoder_embedding of trained modell
            self.decoder_embedding = tf.get_variable("char_embedding", [self.decode_voc_size, self.char_embedding_size])
            with tf.name_scope("Input"):
                self.encoded_seq = tf.placeholder(tf.float32, [None, self.embedding_size])

            with tf.variable_scope("Decoder"):
                self.output_ids = self._decoder(self.encoded_seq)
            
        self.saver_op = tf.train.Saver()
        
    def _feature_regression(self, encoded_seq):
        x = tf.layers.dense(inputs=encoded_seq,
                            units=512,
                            activation=tf.nn.relu
                            )
        x = tf.layers.dense(inputs=x,
                            units=128,
                            activation=tf.nn.relu
                            )
        x = tf.layers.dense(inputs=x,
                            units=self.num_features,
                            activation=None
                            )
        
        return x
    
    def _compute_loss(self, sequence_logits, features_predictions):
        
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.shifted_target_seq,
                                                                  logits=sequence_logits)
        loss_sequence = (tf.reduce_sum(crossent * self.target_mask))
        loss_features = tf.losses.mean_squared_error(labels=self.mol_features,
                                                     predictions=features_predictions,
                                                    )
        return loss_sequence, loss_features

class NoisyGRUSeq2SeqWithFeatures(GRUSeq2SeqWithFeatures):
    def __init__(self, mode, iterator, hparams):
        super().__init__(mode, iterator, hparams)
        self.input_dropout = hparams.input_dropout
        self.emb_noise = hparams.emb_noise
        
    def _encoder(self, encoder_emb_inp):
        if self.mode == "TRAIN":
            max_time = tf.shape(encoder_emb_inp)[1]
            encoder_emb_inp = tf.nn.dropout(encoder_emb_inp, 1. - self.input_dropout, noise_shape=[self.batch_size, max_time, 1])
        encoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]
        encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                           encoder_emb_inp,
                                                           sequence_length=self.input_len,
                                                           dtype=tf.float32,
                                                           time_major=False)
        emb = tf.layers.dense(tf.concat(encoder_state, axis=1),
                              self.embedding_size
                             )
        if (self.emb_noise >= 0) & (self.mode == "TRAIN"):
            emb += tf.random_normal(shape=tf.shape(emb), mean=0.0, stddev=self.emb_noise, dtype=tf.float32)
        emb = tf.tanh(emb)
        return emb
        