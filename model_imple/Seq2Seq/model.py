import numpy as np
import tensorflow as tf
import time
import config


class ChatBotModel(object):
    def __init__(self, forward_only, batch_size):
        """forward_only: if set, we do not construct the backward pass in the model.
        """
        print('Initialize new model')
        self.fw_only = forward_only
        self.batch_size = batch_size
    
    def _create_placeholders(self):
        # Feeds for inputs. It's a list of placeholders
        print('Create placeholders')
        self.encoder_inputs = [] 
        self.decoder_inputs = []
        self.decoder_masks = []
        # 가장 긴 버킷 기준으로 placeholder list를 만든다
        for i in range(config.BUCKETS[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name='encoder{}'.format(i)))
        for i in range(config.BUCKETS[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name='decoder{}'.format(i)))
            self.decoder_masks.append(tf.placeholder(tf.float32, shape=[None],
                                                    name='mask{}'.format(i)))

        # Our targets are decoder inputs shifted by one (to ignore <s> symbol)
        # <s> 심볼을 무시하기 위해 decoder input+1로 잡는다?
        self.targets = [self.decoder_inputs[i + 1]
                        for i in range(len(self.decoder_inputs) - 1)]

    def _inference(self):
        print('Create inference')
        # sampled softmax를 사용할거면, output projection 과정이 필요하다
        # sampled softmax는 sample 할 수가 vocabulary size보다 작을 때만 사용
        # 그렇지 않다면 그냥 softmax를 사용하는 것이 더 효과적
        # 또한 실제 test (inference)할 때는, full softmax해야 한다
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB:
            w = tf.get_variable('proj_w', [config.HIDDEN_SIZE, config.DEC_VOCAB],dtype=tf.float32)
            b = tf.get_variable('proj_b', [config.DEC_VOCAB],dtype=tf.float32)
            self.output_projection = (w, b)

        def sampled_loss(labels, inputs):
            labels = tf.reshape(labels, [-1, 1])
           
            ## 어떤식으로 reshape 되는거지?
            # This is a faster way to train a softmax classifier over a huge number of classes
            return tf.nn.sampled_softmax_loss(tf.transpose(w), b, labels, inputs,
                                              config.NUM_SAMPLES, config.DEC_VOCAB)
        
        self.softmax_loss_function = sampled_loss
        single_cell = tf.contrib.rnn.GRUCell(config.HIDDEN_SIZE)
        self.cell = tf.contrib.rnn.MultiRNNCell([single_cell] * config.NUM_LAYERS)

    def _create_loss(self):
        print('Creating loss... \nIt might take a couple of minutes depending on how many buckets you have.')
        start = time.time()
        
        #
        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            # Embedding sequence-to-sequence model with attention
            # encoder_inputs을 [num_encoder_symbols X input_size] shape로 만든다
            # 그런 다음 RNN의 state vector로 인코딩 되고
            # 이는 유지되고 있다고 추후 attention에 사용된다
            # 그리고 decoder inputs을  [num_decoder_symbols X input_size] shape로 만들고
            # last encoder state로 초기화하여 attention decoder을 수행한다
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, self.cell,
                    num_encoder_symbols=config.ENC_VOCAB,
                    num_decoder_symbols=config.DEC_VOCAB,
                    embedding_size=config.HIDDEN_SIZE, # 토큰을 임베딩 시킬 사이즈
                    output_projection=self.output_projection, # (w,b) 혹은 None
                    feed_previous=do_decode) # True면 디코더가 "GO"심볼 사용한다. 
                                                            # forward_only일 때, True
                                                            # Train 시킬 때는, False로 둔다
                                                            # 즉 경우에 따라 다른 function을 사용하는 셈

        if self.fw_only:
            # seq2seq 모델을 bucketing을 지원하도록 생성한다
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoder_inputs, 
                                        self.decoder_inputs, 
                                        self.targets, # A list of 1D batch-sized int32 Tensors
                                        self.decoder_masks, # 디코더 마스크 weights 1 or 0
                                        config.BUCKETS, # [(n,m)]
                                        lambda x, y: _seq2seq_f(x, y, True), # seq2seq model function
                                        # encoder_inputs과 decoer_inputs을 인자로 받아 outputs과 states를
                                        # 리턴하는 function이어야 한다. 
                                        softmax_loss_function=self.softmax_loss_function)
            

            # 만약 output projection을 사용한다면, output을 디코딩하기 위해 project해야 한다
            # 버킷별 아웃풋마다 projection 진행
            # forward_only 이기 때문임..!!
            if self.output_projection:
                for bucket in range(len(config.BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output, 
                                            self.output_projection[0]) + self.output_projection[1] # (w,b)
                                            for output in self.outputs[bucket]] 
        else:
            # train 모드라면
            # do_decode를 False로 단지 생성한다
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoder_inputs, 
                                        self.decoder_inputs, 
                                        self.targets,
                                        self.decoder_masks,
                                        config.BUCKETS,
                                        lambda x, y: _seq2seq_f(x, y, False),
                                        softmax_loss_function=self.softmax_loss_function)
        print('Time:', time.time() - start)

    def _creat_optimizer(self):
        print('Create optimizer... \nIt might take a couple of minutes depending on how many buckets you have.')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.fw_only: # 즉 fw_only = False로 들어오면?
                # 옵티마이저 생성
                self.optimizer = tf.train.GradientDescentOptimizer(config.LR)
                trainables = tf.trainable_variables() # 모든 trainable=True인 variable들을 리턴한다
                # 이 역시 버킷 별로 최적화한다
                
                self.gradient_norms = []
                self.train_ops = []
                start = time.time()
                for bucket in range(len(config.BUCKETS)):
                    
                    # bucket 별로 variables의 gradients를 구한 뒤, clip 하고 (exploding 예방)
                    # norm과 clipped_grads를 최적화하여 bucket 별로 넣는다
                    # optimizer.apply() 메소드 이용..
                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket], 
                                                                 trainables),
                                                                 config.MAX_GRAD_NORM)
                    self.gradient_norms.append(norm) 
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables), 
                                                            global_step=self.global_step))
                    print('Creating opt for bucket {} took {} seconds'.format(bucket, time.time() - start))
                    start = time.time()


    def _create_summary(self):
        # summary는 따로 안 만듬
        pass

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._creat_optimizer()
        self._create_summary()