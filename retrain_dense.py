import pickle

import tensorflow as tf

import config
import dnnlib
from dnnlib import tflib

# Initialize dnnlib and TensorFlow.
ctx = dnnlib.RunContext(submit_config, train)
tflib.init_tf(tf_config)

url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    G, D, Gs = pickle.load(f)

#### LOAD IN DATASET HERE


with tf.name_scope('Inputs'), tf.device('/cpu:0'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // submit_config.num_gpus
        Gs_beta         = 0.5 ** tf.div(tf.cast(minibatch_in, tf.float32), G_smoothing_kimg * 1000.0) if G_smoothing_kimg > 0.0 else 0.0

    D_opt = tflib.Optimizer(name='TrainD', learning_rate=lrate_in, **D_opt_args)
    for gpu in range(submit_config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')
            lod_assign_ops = [tf.assign(G_gpu.find_var('lod'), lod_in), tf.assign(D_gpu.find_var('lod'), lod_in)]
            reals, labels = training_set.get_minibatch_tf()
            reals = process_reals(reals, lod_in, mirror_augment, training_set.dynamic_range, drange_net)
             with tf.name_scope('D_loss'), tf.control_dependencies(lod_assign_ops):
                D_loss = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set, minibatch_size=minibatch_split, reals=reals, labels=labels, **D_loss_args)
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)
    D_train_op = D_opt.apply_updates()

print('Training...\n')
    ctx.update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = ctx.get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    while cur_nimg < total_kimg * 1000:
        if ctx.should_stop(): break

        # Choose training parameters and configure training ops.
        sched = training_schedule(cur_nimg=cur_nimg, training_set=training_set, num_gpus=submit_config.num_gpus, **sched_args)
        training_set.configure(sched.minibatch // submit_config.num_gpus, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
              D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        for _mb_repeat in range(minibatch_repeats):
            for _D_repeat in range(D_repeats):
                tflib.run([D_train_op, Gs_update_op], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})
                cur_nimg += sched.minibatch




#
#
# error = tf.square(y - D)
# optimizer = tf.train.AdamOptimizer()
# train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "4x4")
# train_op = optimizer.minimize(error,var_list = train_vars)
#
# with tf.Session() as session:
#     session.run(D)
#     session.run(train_op,feed_dict={x:x_value, y:y_value})
optimizer = tf.train.AdamOptimizer()