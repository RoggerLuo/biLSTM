import tensorflow as tf
import numpy as np
from os.path import join,exists


def train(FLAGS,cross_entropy,accuracy,train_initializer,dev_initializer,keep_prob):

    global_step = tf.Variable(-1, trainable=False, name='global_step')
    # keep_prob = tf.placeholder(tf.float32, [])


    # Train
    train = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy, global_step=global_step)
    
    # Saver
    saver = tf.train.Saver()
    
    # Iterator
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    if exists('./ckpt'):
        ckpt = tf.train.get_checkpoint_state('ckpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore from', ckpt.model_checkpoint_path)


    # Global step
    gstep = 0
    
    # Summaries
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(join(FLAGS.summaries_dir, 'train'),
                                   sess.graph)
    
    
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    
    for epoch in range(FLAGS.epoch_num):
        tf.train.global_step(sess, global_step_tensor=global_step)
        # Train
        sess.run(train_initializer)
        for step in range(int(FLAGS.train_steps)):
            smrs, loss, acc, gstep, _ = sess.run([summaries, cross_entropy, accuracy, global_step, train],
                                                 feed_dict={keep_prob: FLAGS.keep_prob})
            # Print log
            if step % FLAGS.steps_per_print == 0:
                print('Global Step', gstep, 'Step', step, 'Train Loss', loss, 'Accuracy', acc)
            
            # Summaries for tensorboard
            if gstep % FLAGS.steps_per_summary == 0:
                writer.add_summary(smrs, gstep)
                print('Write summaries to', FLAGS.summaries_dir)
        
        if epoch % FLAGS.epochs_per_dev == 0:
            # Dev
            sess.run(dev_initializer)
            for step in range(int(FLAGS.dev_steps)):
                if step % FLAGS.steps_per_print == 0:
                    print('Dev Accuracy', sess.run(accuracy, feed_dict={keep_prob: 1}), 'Step', step)
        
        # Save model
        if epoch % FLAGS.epochs_per_save == 0:
            saver.save(sess, FLAGS.checkpoint_dir) # , global_step=gstep # 记录global_step的话，就不会覆盖，然后文件就超大，媒每一步的都会记下来


