import multiprocessing

import tensorflow as tf
import random 
import timeit
import time

WORLD_DELAY = 4 
AGENT_DELAY = 1 
DEBUG = False

class WorldModel(object):

    def __init__(self):
        pass

    def update(self):
        time.sleep(WORLD_DELAY)
        return random.random()

    def generator(self):
        while True:
            time.sleep(WORLD_DELAY)
            yield random.random()


class WorldModelRunner(multiprocessing.Process):
    call_count = 0

    def __init__(self, result_queue, worker_id):
        multiprocessing.Process.__init__(self)
        self.result_queue = result_queue
        self.worker_id = worker_id

    def run(self):
        while True:
            if DEBUG:
                WorldModelRunner.call_count += 1 
                print self.worker_id + 1, WorldModelRunner.call_count
            time.sleep(WORLD_DELAY)
            self.result_queue.put(random.random())

class Agent(object):

    def __init__(self):
        pass

    def action(self):
        time.sleep(AGENT_DELAY)
        return random.random()

class WrappedQueue():

    def __init__(self, queue):
        self.queue = queue

    def generator(self):
        while True:
            yield self.queue.get()

def dataset_train_multiprocess(): 
    # This function doesn't work
    sess = tf.Session()
    queue = multiprocessing.Queue()
    queue_wrapper = WrappedQueue(queue)

    agent = Agent()
    pool = [WorldModelRunner(queue, idx) for idx in range(multiprocessing.cpu_count())]

    for worker in pool:
        worker.start()
    
    dataset = tf.data.Dataset.from_generator(queue_wrapper.generator, tf.float64, tf.TensorShape([]))
    dataset = dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    world_state = tf.cast(next_element, tf.float64)
    agent_action = tf.py_func(agent.action, [], tf.float64)
    out = world_state - agent_action

    for i in range(30):
        out_p, world_state_p, agent_action_p = sess.run([out, world_state, agent_action])
        if DEBUG:
            print out_p, world_state_p, agent_action_p
      
    for worker in pool:
        worker.terminate()


def dataset_train(): 
    model = WorldModel()
    agent = Agent()

    dataset = tf.data.Dataset.from_generator(model.generator, tf.float64, tf.TensorShape([]))
    dataset = dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    world_state = tf.cast(next_element, tf.float64)
    agent_action = tf.py_func(agent.action, [], tf.float64)
    out = world_state - agent_action

    sess = tf.Session()

    for i in range(30):
        out_p, world_state_p, agent_action_p = sess.run([out, world_state, agent_action])
        if DEBUG:
            print out_p, world_state_p, agent_action_p

def feed_dict_update():
    model = WorldModel()
    agent = Agent()

    placeholder = tf.placeholder(tf.float64, shape=())
    feed_dict = {placeholder: model.update()} 

    sess = tf.Session()
    world_state = placeholder
    agent_action = tf.py_func(agent.action, [], tf.float64)
    out = world_state - agent_action

    for i in range(30):
        feed_dict = {placeholder: model.update()} 
        out_p, world_state_p, agent_action_p = sess.run([out, world_state, agent_action], feed_dict=feed_dict)
        if DEBUG:
            print out_p, world_state_p, agent_action_p

def test_workers():
    sess = tf.Session()
    queue = multiprocessing.Queue()

    pool = [WorldModelRunner(queue, idx) for idx in range(multiprocessing.cpu_count())]

    for worker in pool:
        worker.start()
    
    for item in range(30):
        queue.get()
      
    for worker in pool:
        worker.terminate()

def main():
    print "Time to update via feed_dict mechanism: ", timeit.timeit(feed_dict_update, number=1)
    print "Time to update via Data API: ", timeit.timeit(dataset_train, number=1)
    print "Time to update via multi_process Data API: ", timeit.timeit(dataset_train_multiprocess, number=1)

if __name__ == "__main__":
    main()
