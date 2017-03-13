import os
import pprint
import tensorflow as tf

from data import read_data
from model import MemN2N

pp = pprint.PrettyPrinter()

flags = tf.app.flags

# flag를 사용하면 메모리로 잡지 않고
# 컴파일러가 컴파일할 때, define한거 마냥 치환 가능
# + Config 마냥 사용

flags.DEFINE_integer("edim", 150, "internal state dimension [150]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 3, "number of hops [6]")
flags.DEFINE_integer("mem_size", 100, "memory size [100]")
flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_string("data_dir", "data", "data directory [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_string("data_name", "ptb", "data set name [ptb]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", False, "print progress [False]")

FLAGS = flags.FLAGS

def main(_):
    count = []
    word2idx = {}

    if not os.path.exists(FLAGS.checkpoint_dir): # 체크포인트 디렉토리가 없으면
        os.makedirs(FLAGS.checkpoint_dir) # 만든다

    train_data = read_data('%s/%s.train.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
    valid_data = read_data('%s/%s.valid.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
    test_data = read_data('%s/%s.test.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)

    idx2word = dict(zip(word2idx.values(), word2idx.keys())) # word2index 를 key,value만 바꾸면 index2word dict!
    FLAGS.nwords = len(word2idx)

    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        model = MemN2N(FLAGS, sess)
        model.build_model()

        if FLAGS.is_test:
            model.run(valid_data, test_data)
        else:
            model.run(train_data, valid_data)

if __name__ == '__main__':
    tf.app.run()
