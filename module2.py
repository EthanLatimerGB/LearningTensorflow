import numpy
import tensorflow as tf 

rank0_string = tf.Variable("this is a string", tf.string)
rank1_tensor = tf.Variable(["car", "perosn", "street"], tf.string)
rank2_tensor = tf.Variable([["car", "van"], ["person", "child"]], tf.string)


print("Ranks of different variables")
print(tf.rank(rank0_string))
print(tf.rank(rank1_tensor))
print(tf.rank(rank2_tensor))

print(rank0_string.shape)
print(rank1_tensor.shape)
print(rank2_tensor.shape)


## Reshaping
print("Reshaping \n\n")
tensor1 = tf.ones([1,2,3])
tensor2 = tf.reshape(tensor1, [2,3,1])
tensor3 = tf.reshape(tensor2, [3, -1])

print(tensor1)
print(tensor2)
print(tensor3)

## Pracice

t = tf.zeros([5,5,5,5])
t1 = tf.reshape(t, [25, -1])
print(t1)
