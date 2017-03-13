import os
import math
import random
import numpy as np
import tensorflow as tf

class MemN2N(object):
    def __init__(self, config, sess):
        
        # flag로부터 각종 하이퍼파라미터 초기화
        self.nwords = config.nwords
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.nhop = config.nhop
        self.edim = config.edim
        self.mem_size = config.mem_size
        self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm

        self.show = config.show
        self.is_test = config.is_test
        self.checkpoint_dir = config.checkpoint_dir

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

        self.input = tf.placeholder(tf.float32, [None, self.edim], name="input") # question
        self.time = tf.placeholder(tf.int32, [None, self.mem_size], name="time")
        self.target = tf.placeholder(tf.float32, [self.batch_size, self.nwords], name="target") # answer
        self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="context") # sentences

        self.hid = []
        self.hid.append(self.input)
        self.share_list = []
        self.share_list.append([])

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = [] # loss와 perplexity를 저장해두기 위한 log list
        self.log_perp = []

    def build_memory(self):
        self.global_step = tf.Variable(0, name="global_step")

        self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std)) # V*d : Input memory representation
        self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std)) # V*d : query representation
        self.C = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std)) # output memory representation

        # Temporal Encoding
        # memory size만큼의 index에서 읽어오는
        # m_i = sum(Ax_ij + T_A(i))
        self.T_A = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std)) 
        self.T_B = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))

        # m_i = sum A_ij * x_ij + T_A_i
        Ain_c = tf.nn.embedding_lookup(self.A, self.context) # [batch_size,mem_size] -> [batch_size,edim]
        Ain_t = tf.nn.embedding_lookup(self.T_A, self.time) # 대화셋의 i번째 time에서의 temporal context [batch_size,edim]
        Ain = tf.add(Ain_c, Ain_t) # [batch_size,edim]

        # c_i = sum B_ij * u + T_B_i
        Bin_c = tf.nn.embedding_lookup(self.B, self.context) 
        Bin_t = tf.nn.embedding_lookup(self.T_B, self.time)
        Bin = tf.add(Bin_c, Bin_t)

        for h in range(self.nhop): # nhop = 3
            # hid[-1]에 첨에 input 들어가 있음
            self.hid3dim = tf.reshape(self.hid[-1], [-1, 1, self.edim]) # [batch_size,edim] -> [batch_size, 1, edim]
            Aout = tf.matmul(self.hid3dim, Ain, adjoint_b=True) # Ain.T 
            Aout2dim = tf.reshape(Aout, [-1, self.mem_size]) # 다시 [문장수, mem_size] 로 복구
            P = tf.nn.softmax(Aout2dim) # p_i

            probs3dim = tf.reshape(P, [-1, 1, self.mem_size]) # Prob도 3차원으로 표현해서
            Bout = tf.matmul(probs3dim, Bin) # weighted sum
            Bout2dim = tf.reshape(Bout, [-1, self.edim]) # 다시 2차원으로 복구

            Cout = tf.matmul(self.hid[-1], self.C) # 최초 input을 
            Dout = tf.add(Cout, Bout2dim) # o + u ??

            self.share_list[0].append(Cout)
            
            
            # 이 아래 코드는 뭐다냐.... 
            if self.lindim == self.edim:
                self.hid.append(Dout)
            elif self.lindim == 0:
                self.hid.append(tf.nn.relu(Dout))
            else:
                F = tf.slice(Dout, [0, 0], [self.batch_size, self.lindim])
                G = tf.slice(Dout, [0, self.lindim], [self.batch_size, self.edim-self.lindim])
                K = tf.nn.relu(G)
                self.hid.append(tf.concat([F, K],1))

    def build_model(self):
        self.build_memory()

        self.W = tf.Variable(tf.random_normal([self.edim, self.nwords], stddev=self.init_std)) # d*V 매핑
        z = tf.matmul(self.hid[-1], self.W) # 마지막 hop의 output.. (o^k + u^k)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=z,labels= self.target) # loss

        self.lr = tf.Variable(self.current_lr)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        params = [self.A, self.B, self.C, self.T_A, self.T_B, self.W]
        grads_and_vars = self.opt.compute_gradients(self.loss,params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                   for gv in grads_and_vars] # List of (gradient, variable) pairs // gradient는 clip_by_norm !
        
        # clip by norm 해서 각각 gradients update

        inc = self.global_step.assign_add(1) # global step 하나 올려주고
        with tf.control_dependencies([inc]): # 반드시 [] 안에 있는 value를 먼저 실행한 후, 아래 command가 실행된다.
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)
            # 원래는 train.optimizer 안에 param으로 global_step 쓰는데 clip_by_norm 하느라 쪼개는 바람에
            # 이를 보장해주기 위해서 이렇게 하는듯

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def train(self, data):
        N = int(math.ceil(len(data) / self.batch_size)) # math.ceil : returns smallest integer not less than x.
        cost = 0

        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])

        x.fill(self.init_hid) # 초기화 
        for t in range(self.mem_size):
            time[:,t].fill(t) # [[0,1,2,3,4,...,mem_size] ... ]

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Train', max=N)

        for idx in range(N):
            if self.show: bar.next()
            target.fill(0)
            for b in range(self.batch_size):
                m = random.randrange(self.mem_size, len(data)) # 100~ x  에서 하나를 가져와서..
                target[b][data[m]] = 1 # 타겟을 랜덤으로 고르는건가?
                context[b] = data[m - self.mem_size:m] # 그 단어의 100번째 전 단어까지를 context로 사용

            _, loss, self.step = self.sess.run([self.optim,
                                                self.loss,
                                                self.global_step],
                                                feed_dict={
                                                    self.input: x,
                                                    self.time: time,
                                                    self.target: target,
                                                    self.context: context})
            cost += np.sum(loss)

        if self.show: bar.finish()
        return cost/N/self.batch_size

    def test(self, data, label='Test'):
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])

        x.fill(self.init_hid)
        for t in range(self.mem_size):
            time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar(label, max=N)

        m = self.mem_size 
        for idx in range(N):
            if self.show: bar.next()
            target.fill(0)
            for b in range(self.batch_size):
                target[b][data[m]] = 1 # 타겟을 nwords 중에 하나만 1로 하는 one-hot 
                context[b] = data[m - self.mem_size:m] # 이건 당최 뭐다냐 ㅠㅠ
                m += 1

                if m >= len(data):
                    m = self.mem_size # ???

            loss = self.sess.run([self.loss], feed_dict={self.input: x,
                                                         self.time: time,
                                                         self.target: target,
                                                         self.context: context})
            cost += np.sum(loss)

        if self.show: bar.finish()
        return cost/N/self.batch_size

    def run(self, train_data, test_data):
        if not self.is_test:
            for idx in range(self.nepoch):
                train_loss = np.sum(self.train(train_data))
                test_loss = np.sum(self.test(test_data, label='Validation'))

                # Logging
                self.log_loss.append([train_loss, test_loss])
                self.log_perp.append([math.exp(train_loss), math.exp(test_loss)])

                state = {
                    'perplexity': math.exp(train_loss),
                    'epoch': idx,
                    'learning_rate': self.current_lr,
                    'valid_perplexity': math.exp(test_loss)
                }
                print(state)

                # Learning rate annealing
                # 러닝 레이트 서서히 떨어뜨리기
                if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx-1][1] * 0.9999:
                    self.current_lr = self.current_lr / 1.5
                    self.lr.assign(self.current_lr).eval()
                if self.current_lr < 1e-5: break

                if idx % 10 == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "MemN2N.model"),
                                    global_step = self.step.astype(int))
        else:
            self.load()

            valid_loss = np.sum(self.test(train_data, label='Validation'))
            test_loss = np.sum(self.test(test_data, label='Test'))

            state = {
                'valid_perplexity': math.exp(valid_loss),
                'test_perplexity': math.exp(test_loss)
            }
            print(state)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Trest mode but no checkpoint found")

