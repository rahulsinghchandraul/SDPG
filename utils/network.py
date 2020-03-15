import tensorflow as tf
import numpy as np
from utils.ops import dense, relu, tanh, batchnorm, softmax
from utils.l2_projection import _l2_project

class Critic:
	def __init__(self, state, action, noise, state_dims, action_dims, noise_dims, dense1_size, dense2_size, final_layer_init, num_atoms, v_min, v_max, scope='critic'):
		# state - State input to pass through the network
		# action - Action input for which the Z distribution should be predicted
		 
		self.state = state
		self.action = action
		self.noise = noise
		self.noise_dims = np.prod(noise_dims)
		self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
		self.action_dims = np.prod(action_dims)
		self.v_min = v_min
		self.v_max = v_max
		self.scope = scope   
		batch_size = 256   
		 
		with tf.variable_scope(self.scope):           
			self.dense1_mul = dense(self.state, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
								bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')  
						 
			self.dense1 = relu(self.dense1_mul, scope='dense1')
			 
			#Merge first dense layer with action and noise input to get second dense layer            
			self.dense2a = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
								bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2a')        
			 
			self.dense2a = tf.reshape(self.dense2a, [batch_size, 1 , dense2_size])
			
			self.dense2b = dense(self.action, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), \
			1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),\
			bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2b') 
			
			self.dense2b = tf.reshape(self.dense2b, [batch_size, 1 , dense2_size])
			
			self.noise = tf.reshape(self.noise, [batch_size*num_atoms , noise_dims])
			
			self.dense2c = dense(self.noise, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.noise_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.noise_dims))),
								bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.noise_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.noise_dims))), scope='dense2c') 
						   
			
			self.dense2c = tf.reshape(self.dense2c, [batch_size, num_atoms , dense2_size])
			
			self.dense2 = relu(self.dense2a + self.dense2b + self.dense2c, scope='dense2')
			
			self.dense2 = tf.reshape(self.dense2, [batch_size*num_atoms, dense2_size])
			
						  
			self.output_mul = dense(self.dense2, 1, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
									   bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output')  
			
			self.output_tanh = tanh(0.2*self.output_mul, scope='output')
			
			self.output_samples = tf.multiply(0.5, tf.multiply(self.output_tanh, (self.v_max - self.v_min)) + (self.v_max + self.v_min))
			
			self.output_samples = tf.reshape(self.output_samples, [batch_size, num_atoms])
						 
						  
			self.network_params = tf.trainable_variables(scope=self.scope)
			self.bn_params = [] # No batch norm params
			
			self.Q_val = tf.reduce_mean(self.output_samples, axis=1) # the Q value is the mean of the generated samples
		  
			self.action_grads = tf.gradients(self.output_samples/num_atoms, self.action) # gradient of mean of output Z-distribution wrt action input - used to train actor network, weighing the grads by z_values gives the mean across the output distribution
			
	def train_step(self, real_samples,  IS_weights, learn_rate, l2_lambda):
		# target_Z_dist - target Z distribution for next state
		# target_Z_atoms - atom values of target network with Bellman update applied
		 
		with tf.variable_scope(self.scope):
			with tf.variable_scope('train'):
				self.optimizer = tf.train.AdamOptimizer(learn_rate) 
							 
				real_samples = tf.stop_gradient(real_samples)
				self.real_sort = tf.sort(real_samples, axis = 1,  direction='ASCENDING' )
				self.fake_sort = tf.sort(self.output_samples, axis=1,  direction='ASCENDING' )
				
				self.real_sort_tile = tf.tile(tf.expand_dims(real_sort, axis=2), [1, 1, num_atoms])
				self.fake_sort_tile = tf.tile(tf.expand_dims(fake_sort, axis=1), [1, num_atoms, 1])
				
				self.error_loss = real_sort_tile - fake_sort_tile
				
				self.Huber_loss = tf.losses.huber_loss(real_sort_tile, fake_sort_tile, reduction = tf.losses.Reduction.NONE)
				
				self.min_tau = 1/(2*num_atoms)
				self.max_tau = (2*num_atoms+1)/(2*num_atoms)
				self.tau = tf.reshape (tf.range(self.min_tau, self.max_tau, 1/num_atoms), [1, num_atoms])
				self.inv_tau = 1.0 - self.tau 
				
				self.loss = tf.where(tf.less(self.error_loss, 0.0), self.inv_tau * self.Huber_loss, self.tau * self.Huber_loss)
				
				self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(self.loss, axis = 2), axis = 1))
				 
				train_step = self.optimizer.minimize(self.loss, var_list=self.network_params)
				  
				return train_step
		

class Actor:
	def __init__(self, state, state_dims, action_dims, action_bound_low, action_bound_high, dense1_size, dense2_size, final_layer_init, scope='actor'):
		# state - State input to pass through the network
		# action_bounds - Network will output in range [-1,1]. Multiply this by action_bound to get output within desired boundaries of action space
		 
		
		self.state = state
		self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
		self.action_dims = np.prod(action_dims)
		self.action_bound_low = action_bound_low
		self.action_bound_high = action_bound_high
		self.scope = scope
		 
		with tf.variable_scope(self.scope):
					
			self.dense1_mul = dense(self.state, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
								bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')  
						 
			self.dense1 = relu(self.dense1_mul, scope='dense1')
			 
			self.dense2_mul = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))),
								bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))), scope='dense2')        
						 
			self.dense2 = relu(self.dense2_mul, scope='dense2')
			 
			self.output_mul = dense(self.dense2, self.action_dims, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
								bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output') 
			 
			self.output_tanh = tanh(self.output_mul, scope='output')
			 
			# Scale tanh output to lower and upper action bounds
			self.output = tf.multiply(0.5, tf.multiply(self.output_tanh, (self.action_bound_high-self.action_bound_low)) + (self.action_bound_high+self.action_bound_low))
			 
			
			self.network_params = tf.trainable_variables(scope=self.scope)
			self.bn_params = [] # No batch norm params
		
		
	def train_step(self, action_grads, learn_rate, batch_size):
		# action_grads - gradient of value output wrt action from critic network
		 
		with tf.variable_scope(self.scope):
			with tf.variable_scope('train'):
				 
				self.optimizer = tf.train.AdamOptimizer(learn_rate)
				self.grads = tf.gradients(self.output, self.network_params, -action_grads)  
				self.grads_scaled = list(map(lambda x: tf.divide(x, batch_size), self.grads)) # tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradients
				 
				train_step = self.optimizer.apply_gradients(zip(self.grads_scaled, self.network_params))
				 
				return train_step
			
			
class Critic_BN:
	def __init__(self, state, action, noise, state_dims, action_dims, noise_dims, dense1_size, dense2_size, final_layer_init, num_atoms, v_min, v_max, is_training=False, scope='critic'):
		# state - State input to pass through the network
		# action - Action input for which the Z distribution should be predicted
		
		self.state = state
		self.action = action
		self.noise = noise
		self.noise_dims = np.prod(noise_dims)
		self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
		self.action_dims = np.prod(action_dims)
		self.v_min = v_min
		self.v_max = v_max
		self.num_atoms = num_atoms
		self.scope = scope   
		batch_size = 256 
		self.is_training = is_training
		self.scope = scope    

		
		with tf.variable_scope(self.scope):           
			self.dense1_mul = dense(self.state, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
								bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')  
						 
			self.dense1 = relu(self.dense1_mul, scope='dense1')
			 
			#Merge first dense layer with action and noise input to get second dense layer            
			self.dense2a = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),
								bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2a')        
			 
			self.dense2a = tf.reshape(self.dense2a, [batch_size, 1 , dense2_size])
			
			self.dense2b = dense(self.action, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), \
			1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))),\
			bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.action_dims))), scope='dense2b') 
			
			self.dense2b = tf.reshape(self.dense2b, [batch_size, 1 , dense2_size])
			
			self.noise = tf.reshape(self.noise, [batch_size*num_atoms , noise_dims])
			
			self.dense2c = dense(self.noise, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.noise_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.noise_dims))),
								bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size+self.noise_dims))), 1/tf.sqrt(tf.to_float(dense1_size+self.noise_dims))), scope='dense2c') 
						   
			
			self.dense2c = tf.reshape(self.dense2c, [batch_size, num_atoms , dense2_size])
			
			self.dense2 = relu(self.dense2a + self.dense2b + self.dense2c, scope='dense2')
			
			self.dense2 = tf.reshape(self.dense2, [batch_size*num_atoms, dense2_size])
			
						  
			self.output_mul = dense(self.dense2, 1, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
									   bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output')  
			
			#self.output_tanh = tanh(0.05*self.output_mul, scope='output')
			
			#self.output_samples = tf.multiply(0.5, tf.multiply(self.output_tanh, (self.v_max - self.v_min)) + (self.v_max + self.v_min))
			
			self.output_samples = self.output_mul
			
			self.output_samples = tf.reshape(self.output_samples, [batch_size, num_atoms])
						 
						  
			self.network_params = tf.trainable_variables(scope=self.scope)
			self.bn_params = [v for v in tf.global_variables(scope=self.scope) if 'batch_normalization/moving' in v.name]
			
			self.Q_val = tf.reduce_mean(self.output_samples, axis=1) # the Q value is the mean of the generated samples
		  
			self.action_grads = tf.gradients(self.output_samples/num_atoms, self.action) # gradient of mean of output Z-distribution wrt action input - used to train actor network, weighing the grads by z_values gives the mean across the output distribution
			
	def train_step(self, real_samples,  IS_weights, learn_rate, l2_lambda, num_atoms):
		# target_Z_dist - target Z distribution for next state
		# target_Z_atoms - atom values of target network with Bellman update applied
		self.num_atoms = num_atoms
		 
		with tf.variable_scope(self.scope):
			with tf.variable_scope('train'):
				self.optimizer = tf.train.AdamOptimizer(learn_rate) 
							 
				real_samples = tf.stop_gradient(real_samples)
				self.real_sort = tf.sort(real_samples, axis = 1,  direction='ASCENDING' )
				self.fake_sort = tf.sort(self.output_samples, axis=1,  direction='ASCENDING' )
				
				self.real_sort_tile = tf.tile(tf.expand_dims(self.real_sort, axis=2), [1, 1, num_atoms])
				self.fake_sort_tile = tf.tile(tf.expand_dims(self.fake_sort, axis=1), [1, num_atoms, 1])
				
				self.error_loss = self.real_sort_tile - self.fake_sort_tile
				
				self.Huber_loss = tf.losses.huber_loss(self.real_sort_tile, self.fake_sort_tile, reduction = tf.losses.Reduction.NONE)
				
				self.min_tau = 1/(2*num_atoms)
				self.max_tau = (2*num_atoms+1)/(2*num_atoms)
				self.tau = tf.reshape (tf.range(self.min_tau, self.max_tau, 1/num_atoms), [1, num_atoms])
				self.inv_tau = 1.0 - self.tau 
				
				self.loss = tf.where(tf.less(self.error_loss, 0.0), self.inv_tau * self.Huber_loss, self.tau * self.Huber_loss)
				
				self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(self.loss, axis = 2), axis = 1))
				
				 
				update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.scope) # Ensure batch norm moving means and variances are updated every training step
				with tf.control_dependencies(update_ops):
					train_step = self.optimizer.minimize(self.loss, var_list=self.network_params)
				 
				return train_step
		

class Actor_BN:
	def __init__(self, state, state_dims, action_dims, action_bound_low, action_bound_high, dense1_size, dense2_size, final_layer_init, is_training=False, scope='actor'):
		# state - State input to pass through the network
		# action_bounds - Network will output in range [-1,1]. Multiply this by action_bound to get output within desired boundaries of action space
		
		self.state = state
		self.state_dims = np.prod(state_dims)       #Used to calculate the fan_in of the state layer (e.g. if state_dims is (3,2) fan_in should equal 6)
		self.action_dims = np.prod(action_dims)
		self.action_bound_low = action_bound_low
		self.action_bound_high = action_bound_high
		self.is_training = is_training
		self.scope = scope
		
		with tf.variable_scope(self.scope):
		
			self.input_norm = batchnorm(self.state, self.is_training, scope='input_norm')
		   
			self.dense1_mul = dense(self.input_norm, dense1_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))),
								bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(self.state_dims))), 1/tf.sqrt(tf.to_float(self.state_dims))), scope='dense1')  
			
			self.dense1_bn = batchnorm(self.dense1_mul, self.is_training, scope='dense1')
			
			self.dense1 = relu(self.dense1_bn, scope='dense1')
			
			self.dense2_mul = dense(self.dense1, dense2_size, weight_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))),
								bias_init=tf.random_uniform_initializer((-1/tf.sqrt(tf.to_float(dense1_size))), 1/tf.sqrt(tf.to_float(dense1_size))), scope='dense2')        
			
			self.dense2_bn = batchnorm(self.dense2_mul, self.is_training, scope='dense2')
			
			self.dense2 = relu(self.dense2_bn, scope='dense2')
			
			self.output_mul = dense(self.dense2, self.action_dims, weight_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init),
								bias_init=tf.random_uniform_initializer(-1*final_layer_init, final_layer_init), scope='output') 
			
			self.output_tanh = tanh(self.output_mul, scope='output')
			
			# Scale tanh output to lower and upper action bounds
			self.output = tf.multiply(0.5, tf.multiply(self.output_tanh, (self.action_bound_high-self.action_bound_low)) + (self.action_bound_high+self.action_bound_low))
			
		   
			self.network_params = tf.trainable_variables(scope=self.scope)
			self.bn_params = [v for v in tf.global_variables(scope=self.scope) if 'batch_normalization/moving' in v.name]
		
	def train_step(self, action_grads, learn_rate, batch_size):
		# action_grads - gradient of value output wrt action from critic network
		
		with tf.variable_scope(self.scope):
			with tf.variable_scope('train'):
				
				self.optimizer = tf.train.AdamOptimizer(learn_rate)
				self.grads = tf.gradients(self.output, self.network_params, -action_grads)  
				self.grads_scaled = list(map(lambda x: tf.divide(x, batch_size), self.grads)) # tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradients
				
				update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.scope) # Ensure batch norm moving means and variances are updated every training step
				with tf.control_dependencies(update_ops):
					train_step = self.optimizer.apply_gradients(zip(self.grads_scaled, self.network_params))
				
				return train_step
	
	
	
