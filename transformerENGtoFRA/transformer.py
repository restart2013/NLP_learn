import re
import time
import os

import keras
import nltk
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding, Dense, LayerNormalization
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import numpy as np

# 读取数据
data_path = './fra_eng/fra.txt'

input_text = []
target_text = []
input_character = set()
target_character = set()

file = open(data_path, 'r', encoding='utf-8')
lines = file.read().split('\n')
file.close()
print('-------------------------data information-------------------------')
print('dataset:', lines[:5])

# 制作训练集
for line in lines[: min(128000, len(lines)-1)]:#训练集大小128000
    input_s, target_s, no_use = line.split('\t')
    en = re.sub(r'[,!?;+-]', '.', input_s)
    en = nltk.word_tokenize(en)
    en = [ch.lower() for ch in en if ch.isalpha() or ch == '.']
    input_text.append(en)
    fra = re.sub(r'[,!?;+-]', '.', target_s)
    fra = nltk.word_tokenize(fra)
    fra = [ch.lower() for ch in fra if ch.isalpha() or ch == '.']
    fra.append('<EOS>')
    fra.insert(0, '<SOS>')
    target_text.append(fra)
for input_s in input_text:
    for char in input_s:
        if char not in input_character:
            input_character.add(char)
for target_s in target_text:
    for char in target_s:
        if char not in target_character:
            target_character.add(char)

print('input and target text:', input_text[:5], len(lines), target_text[:5])

input_character = sorted(list(input_character))
target_character = sorted(list(target_character))
print('target dict:', target_character[: 100])
num_input_tokens = len(input_character)
num_target_tokens = len(target_character)
input_seq_len = max([len(txt) for txt in input_text])
target_seq_len = max([len(txt) for txt in target_text])
print('4 numbers:', num_input_tokens, num_target_tokens, input_seq_len, target_seq_len)

input_dict = dict([(char, i+1) for i, char in enumerate(input_character)])
target_dict = dict([(char, i+1) for i, char in enumerate(target_character)])
target_dict_ = dict([(i+1, char) for i, char in enumerate(target_character)])
target_dict_[0] = '<padding>'

x_train = np.zeros((len(input_text), input_seq_len))
y_train = np.zeros((len(input_text), target_seq_len))

for i, (input_s, target_s) in enumerate(zip(input_text, target_text)):
    for t, char in enumerate(input_s):
        x_train[i, t] = input_dict[char]
    for t, char in enumerate(target_s):
        y_train[i, t] = target_dict[char]

# make batch
x_train = tf.cast(x_train, tf.int32)
y_train = tf.cast(y_train, tf.int32)
print(y_train[3800])
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(64)
print('train_db:')
for x, y in train_db.take(2):
    print(x.shape)
    print(y.shape)


# tools
# position embedding
@tf.function
def get_angles(pos, i, d_model):
    # pos.shape = (seq_len, 1)
    # i.shape = (1, d_model)
    angle_rates = 1 / tf.pow(1/10000, 2*(tf.cast(i//2, dtype=tf.float32))/tf.cast(d_model, dtype=tf.float32))
    return tf.cast(pos, dtype=tf.float32)*angle_rates
@tf.function
def position_embedding(seq_len, d_model):
    angle_rads = get_angles(tf.range(seq_len)[:, tf.newaxis], tf.range(d_model)[tf.newaxis, :], d_model)
    sin = tf.sin(angle_rads[:, 0::2])
    cos = tf.cos(angle_rads[:, 1::2])
    pos_embedding = tf.concat([sin, cos], axis=-1)
    pos_embedding = pos_embedding[tf.newaxis, ...]
    return tf.cast(pos_embedding, dtype=tf.float32)

# padding mask
def create_padding_mask(batch_data):
    # batch_data.shape = (batch_size, seq_len)
    padding_mask = tf.cast(tf.math.equal(batch_data, 0), dtype=tf.float32)
    return padding_mask[:, tf.newaxis, tf.newaxis, :]

# decoder mask
def create_decoder_mask(seq_len):
    decoder_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return decoder_mask

# attention
@tf.function
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)   # shape = (.., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    matmul_qk /= tf.math.sqrt(dk)
    if mask is not None:
        matmul_qk += (mask * -1e9)
    weight = tf.nn.softmax(matmul_qk, axis=-1)  # shape = (.., seq_len_q, seq_len_k)
    output = tf.matmul(weight, v)   # shape = (.., seq_len_q, d_model / num_heads)
    return output

# multihead attention
class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = self.d_model // self.num_heads
        self.WQ = Dense(self.d_model)
        self.WK = Dense(self.d_model)
        self.WV = Dense(self.d_model)
        self.dense = Dense(self.d_model)
    def split_heads(self, x, batch_size):
        # x.shape = (batch_size, seq_len, d_model) = (batch_size, seq_len, num_heads*depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))    # -1即自动计算剩下维度
        return tf.transpose(x, perm=(0, 2, 1, 3))
    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        q = self.WQ(q)
        k = self.WK(k)
        v = self.WV(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)    # shape = (batch_size, num_heads, seq_len, depth)
        context = scaled_dot_product_attention(q, k, v, mask)   # shape = (batch_size, num_heads, seq_len_q, depth)
        context = tf.transpose(context, (0, 2, 1, 3))
        context = tf.reshape(context, (batch_size, -1, self.d_model))
        outputs = self.dense(context)   # shape = (batch_size, seq_len, d_model)
        return outputs
'''
# 验证数据结构是否正确
multihead_try = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))
output = multihead_try(y, y, y, mask=None)
print('shape:{0}\ncontext:{1}'.format(output.shape, output.numpy()))
'''



# encoder layer
class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.multiheadattention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = Dense(d_model)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
    def call(self, x, mask):
        # x.shape = (batch_size, seq_len, d_model or embedding_size)
        attention_outputs = self.multiheadattention(x, x, x, mask)
        out1 = self.layernorm1(x+attention_outputs)
        feedforward_outputs = self.feedforward(out1)
        out2 = self.layernorm2(out1+feedforward_outputs)
        return out2
'''
# 验证数据结构是否正确
encoderlayer_try = EncoderLayer(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))
output = encoderlayer_try(y, mask=None)
print('shape:{0}\ncontext:{1}'.format(output.shape, output.numpy()))
'''

# decoder layer
class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.mha_self = MultiHeadAttention(d_model, num_heads)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.feedforward = Dense(d_model)
        self.layernorm_self = LayerNormalization(epsilon=1e-6)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
    def call(self, x, encoder_outputs, decoder_mask, padding_mask):
        # x.shape = (batch_size, seq_len_target, d_model or embedding_size) when during predict,seq_len will change according time
        selfattention_ouputs = self.mha_self(x, x, x, decoder_mask)
        out1 = self.layernorm_self(x+selfattention_ouputs)
        attention_outputs = self.mha(out1, encoder_outputs, encoder_outputs, padding_mask)
        out2 = self.layernorm1(out1+attention_outputs)
        feedforward_outputs = self.feedforward(out2)
        out3 = self.layernorm2(out2+feedforward_outputs)
        return out3
'''
# 验证数据结构是否正确
decoderlayer_try = DecoderLayer(d_model=512, num_heads=8)
y = tf.random.uniform((1, 10, 512))    # target
x = tf.random.uniform((1, 40, 512))    # encoder_output
output = decoderlayer_try(y, x, decoder_mask=None, padding_mask=None)
print('shape:{0}\ncontext:{1}'.format(output.shape, output.numpy()))
'''

# Encoder
class Encoder(Layer):
    def __init__(self, num_layers, num_input_tokens, input_seq_len, d_model, num_heads):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(num_input_tokens, self.d_model)
        self.positionembedding = position_embedding(input_seq_len, self.d_model)
        self.encoderlayer = [EncoderLayer(self.d_model, num_heads) for i in range(self.num_layers)]
    def call(self, x, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        x += self.positionembedding[:, :seq_len, :]
        for i in range(self.num_layers):
            x = self.encoderlayer[i](x, mask)
        return x
'''
# 验证数据结构是否正确
encoder_try = Encoder(3, num_input_tokens, input_seq_len, 512, 8)
x = tf.random.uniform((1, input_seq_len))
output = encoder_try(x, None)
print('shape:{0}\ncontext:{1}'.format(output.shape, output.numpy()))
'''

# Decoder
class Decoder(Layer):
    def __init__(self, num_layers, num_target_tokens, target_seq_len, d_model, num_heads):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.embedding =Embedding(num_target_tokens, self.d_model)
        self.positionembedding = position_embedding(target_seq_len, self.d_model)
        self.decoderlayer = [DecoderLayer(self.d_model, num_heads) for i in range(self.num_layers)]
    def call(self, x, encoder_outputs, decoder_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        x += self.positionembedding[:, :seq_len, :]
        for i in range(self.num_layers):
            x = self.decoderlayer[i](x, encoder_outputs, decoder_mask, padding_mask)
        return x
'''
# 验证数据结构是否正确
decoder_try = Decoder(3, num_target_tokens, target_seq_len, 512, 8)
y = tf.random.uniform((1, target_seq_len))    # target
x = tf.random.uniform((1, input_seq_len, 512))    # encoder_outputs
output = decoder_try(y, x, None, None)
print('shape:{0}\ncontext:{1}'.format(output.shape, output.numpy()))
'''

# Transformer
class Transformer(Model):
    def __init__(self, num_layers, num_input_tokens, num_target_tokens,
                 input_seq_len, target_seq_len, d_model, num_heads):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, num_input_tokens, input_seq_len, d_model, num_heads)
        self.decoder = Decoder(num_layers, num_target_tokens, target_seq_len, d_model, num_heads)
        self.dense = Dense(num_target_tokens)
    def call(self, input, target, encoder_padding_mask, decoder_mask, decoder_padding_mask):
        encoder_outputs = self.encoder(input, encoder_padding_mask)
        decoder_outputs = self.decoder(target, encoder_outputs, decoder_mask, decoder_padding_mask)
        outputs = self.dense(decoder_outputs)
        return outputs
'''
# 验证数据结构是否正确
transformer_try = Transformer(3, num_input_tokens, num_target_tokens, input_seq_len, target_seq_len, 512, 8)
y = tf.random.uniform((32, target_seq_len))    # target
x = tf.random.uniform((32, input_seq_len))    # input
output = transformer_try(x, y, None, None, None)
print('shape:{0}\ncontext:{1}'.format(output.shape, output.numpy()))
'''


# 超参数
# 模型参数
units = 64
num_heads = 2
num_layers = 2
# 模型初始化
transformer = Transformer(num_layers, num_input_tokens+1, num_target_tokens+1,
                          input_seq_len, target_seq_len, units, num_heads)
# 训练参数
epochs = 10
# 学习率
# lr = (d_model ** -0.5) * min((step ** -0.5), (step * warmup_steps ** -1.5))
class CustomizedSchedule(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomizedSchedule, self).__init__()
        self.d_model = tf.cast(d_model, dtype=tf.float32)
        self.warmup_steps = warmup_steps
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** (-1.5))
        arg3 = tf.math.rsqrt(self.d_model)
        return arg3 * tf.math.minimum(arg1, arg2)
lr = CustomizedSchedule(units)
# 优化器
optimizer1 = Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-9)
optimizer2 = Adam(learning_rate=3e-5)
optimizer = optimizer1
# 损失函数
loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)

# tools
def creat_mask(input, target):
    encoder_padding_mask = create_padding_mask(input)   # encoder中的padding_mask
    decoder_padding_mask = create_padding_mask(input)   # decoder中对encoder的padding_mask

    decoder_mask_look_ahead = create_decoder_mask(tf.shape(target)[1])
    decoder_mask_self_padding = create_padding_mask(target)
    decoder_mask = tf.maximum(decoder_mask_look_ahead, decoder_mask_self_padding)   # decoder中的padding和lookahead_mask
                                                                                    # 事实上padding是因为training中target长短不一
    return encoder_padding_mask, decoder_mask, decoder_padding_mask


# 加载模型参数
save_path = './transformer'
if not os.path.exists(save_path):
    os.mkdir(save_path)

save_path2 = './trans_cp'
if os.path.exists(save_path2+'/trans_cp.ckpt.index'):
    transformer.load_weights(save_path2+'/trans_cp.ckpt')



# 训练
train_loss = keras.metrics.Mean(name='train_loss')
train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
def train_step(input, target):
    target_input = target[:, :-1]
    target_real = target[:, 1:]
    encoder_padding_mask, decoder_mask, decoder_padding_mask = creat_mask(input, target_input)

    with tf.GradientTape() as tape:
        pred = transformer(input, target_input, encoder_padding_mask, decoder_mask, decoder_padding_mask)
        loss = loss_function(target_real, pred)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(target_real, pred)

for epoch in range(epochs):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    for batch, (inputs, target) in enumerate(train_db.take(len(x_train)//64)):
        train_step(inputs, target)
        if batch % 20 == 0:
            print('epoch:{}\tbatch:{}\tloss:{:.4f}\taccuracy:{:.4f}'.format(
                epoch+1, batch, train_loss.result(), train_accuracy.result()))
    print('------------\nepoch:{}\tloss:{:.4f}\taccuracy:{:.4f}\ttime:{:.2f}s\n------------'.format(
        epoch+1, train_loss.result(), train_accuracy.result(), time.time()-start))


# 保存模型
transformer.summary()
tf.saved_model.save(transformer, save_path)
transformer.save_weights(save_path2+'/trans_cp.ckpt')


# 模型预测
def evaluate(input_sentence):
    result = ''
    input_sentence = re.sub(r'[,!?;+-]', '.', input_sentence)
    input_sentence = nltk.word_tokenize(input_sentence)
    input_sentence = [ch.lower() for ch in input_sentence if ch.isalpha() or ch == '.']
    input_vector = np.zeros((1, input_seq_len))
    for i, char in enumerate(input_sentence):
        input_vector[0, i] = input_dict[char]
    decoder_input = np.zeros((1, 1))
    decoder_input[0, 0] = target_dict['<SOS>']

    for i in range(target_seq_len):
        encoder_padding_mask, decoder_mask, decoder_padding_mask = creat_mask(input_vector, decoder_input)
        pred = transformer(input_vector, decoder_input, encoder_padding_mask, decoder_mask, decoder_padding_mask)
        pred = pred[:, -1, :]
        pred_id = tf.argmax(pred, axis=-1).numpy()  # 此时pred_id是numpy格式,shape = (1,),要将其转换成数字int
        pred_id = pred_id[0]
        if pred_id == target_dict['<EOS>']:
            return result
        result += target_dict_[pred_id] + ' '
        decoder_input_new = np.zeros((1, 1))
        decoder_input_new[0, 0] = pred_id
        decoder_input = tf.concat([decoder_input, decoder_input_new], axis=-1)
    return result

evaluate_num = int(input('想要预测的次数:'))
for i in range(evaluate_num):
    result = evaluate(input('english sentence:'))
    print('france sentence:', result)