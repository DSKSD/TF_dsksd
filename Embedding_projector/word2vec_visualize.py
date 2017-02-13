import os
import tensorflow as tf
import pickle
from tensorflow.contrib.tensorboard.plugins import projector

LOG_DIR = 'logs'
metadata ='./data/metadata.tsv'

embed = pickle.load(open('./data/embedding.p','rb'))
vocabs = pickle.load(open('./data/vocab.p','rb'))
embeds = tf.Variable(embed, name='embeds')


with open(metadata, 'w', encoding='utf-8') as metadata_file:
    metadata_file.write('Name\tPOS\n')
    for row in vocabs:
        metadata_file.write('%s\t%s\n' % (row[0],row[1]))

with tf.Session() as sess:
    saver = tf.train.Saver([embeds])

    sess.run(embeds.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'embeds.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = embeds.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
