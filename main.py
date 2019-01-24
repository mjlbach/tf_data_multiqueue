import multiprocessing

import tensorflow as tf
import random 
import timeit
import time

WORLD_DELAY = 1 
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

    def __init__(self, result_queue):
        multiprocessing.Process.__init__(self)
        self.result_queue = result_queue

    def run(self):
        while True:
            time.sleep(WORLD_DELAY)
            self.result_queue.put(random.random())

class Agent(object):

    def __init__(self):
        pass

    def action(self):
        time.sleep(AGENT_DELAY)
        return random.random()

class Queue(multiprocessing.Queue):
    def generator(self):
        while True:
            yield self.get()

class WrappedQueue():
    def __init__(self, queue):
        self.queue = queue
    def generator(self):
        while True:
            yield queue.get()

def dataset_train_multiprocess(): 
    # This function doesn't work
    sess = tf.Session()
    queue = multiprocessing.Queue()
    wrapped_queue = WrappedQueue(queue)
    print wrapped_queue.next()

    pool = [WorldModelRunner(queue) for _ in range(multiprocessing.cpu_count())]

    for worker in pool:
        worker.start()

    agent = Agent()

    
    dataset = tf.data.Dataset.from_generator(queue, tf.float64, tf.TensorShape([]))
    dataset = dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    world_state = tf.cast(next_element, tf.float64)
    agent_action = tf.py_func(agent.action, [], tf.float64)
    out = world_state - agent_action
      
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

def main():
    # print "Time to update via Data API: ", timeit.timeit(dataset_train, number=1)
    # print "Time to update via feed_dict mechanism: ", timeit.timeit(feed_dict_update, number=1)
    print "Time to update via multi_process Data API: ", timeit.timeit(dataset_train_multiprocess, number=1)

if __name__ == "__main__":
    main()
