import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import logging
from losses.losses import ClassificationLoss
from metrics.metrics import ClassificationMetrics
logger = logging.getLogger("logger")

def build_trainer(model, data, config):
    if config.trainer_name == "classification":
        trainer = ClassificationTrainer(model, data, config)
    else:
        raise ValueError("'{}' is an invalid model name")

    return trainer

class ClassificationTrainer:
    def __init__(self, model, data, config):
        self.model_train = model['train']
        self.model_test = model['eval']
        self.data = data
        self.config = config

        self.loss_fn = ClassificationLoss(config)
        self.optimizer = SGD(learning_rate=self.config.learning_rate)

        self.metric_train = ClassificationMetrics(config.train_writer, name='train')
        self.metric_train_eval = ClassificationMetrics(config.train_writer, name='train_eval')
        self.metric_test = ClassificationMetrics(config.test_writer, name='test')

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    @tf.function
    def compute_grads(self, samples, targets):
        with tf.GradientTape() as tape:
            predictions = self.model_train(samples, training=True)

            ''' generate the targets and apply the corresponding loss function '''

            loss = self.loss_fn(targets, predictions)

        gradients = tape.gradient(loss, self.model_train.trainable_weights)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.config.clip_grad_norm)
        with self.config.train_writer.as_default():
            tf.summary.scalar("grad_norm", grad_norm, self.global_step)
            self.global_step.assign_add(1)

        return gradients, predictions

    @tf.function
    def apply_grads(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.model_train.trainable_weights))

    def sync_eval_model(self):
        model_weights = self.model_train.get_weights()
        ma_weights = self.model_test.get_weights()
        alpha = self.config.moving_average_coefficient
        self.model_test.set_weights([ma * alpha + w*(1-alpha) for ma, w in zip(ma_weights, model_weights)])

    @tf.function
    def train_step(self, samples, targets):
        gradients, predictions = self.compute_grads(samples, targets)
        self.apply_grads(gradients)
        return predictions

    @tf.function
    def eval_step(self, samples):

        predictions = self.model_test(samples, training=False)

        return predictions

    def train_epoch(self, epoch):
        self.metric_train.reset_states()
        self.model_train.reset_states()

        for samples, targets in self.data['train']:
            predictions = self.train_step(samples, targets)
            self.metric_train.update_state(targets, predictions)
            self.sync_eval_model()

        self.metric_train.print(epoch)
        self.metric_train.log_metrics(epoch)

    def evaluate_train(self, epoch):
        self.metric_train_eval.reset_states()
        self.model_test.reset_states()

        for samples, targets in self.data['train_eval']:
            predictions = self.eval_step(samples)
            self.metric_train_eval.update_state(targets, predictions)

        self.metric_train_eval.print(epoch)
        self.metric_train_eval.log_metrics(epoch)

    def evaluate_test(self, epoch):
        self.metric_test.reset_states()
        self.model_test.reset_states()

        for samples, targets in self.data['test']:
            predictions = self.eval_step(samples)
            self.metric_test.update_state(targets, predictions)

        self.metric_test.print(epoch)
        self.metric_test.log_metrics(epoch)

    def train(self):
        for epoch in range(self.config.num_epochs):
            self.train_epoch(epoch)

            if epoch % self.config.eval_freq == 0:
                self.evaluate_train(epoch)
                self.evaluate_test(epoch)