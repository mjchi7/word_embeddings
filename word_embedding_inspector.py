# word embedding inspectors
import numpy as np 
import tensorflow as tf 
import argparse
import pickle
import os
import pprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def retrieve_weights(ckpt_dir, variable_scope, variable_name):
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(ckpt_dir+'.meta')
		saver.restore(sess, ckpt_dir)
		graph = tf.get_default_graph()
		embedding_layer = graph.get_tensor_by_name(variable_scope+'/'+variable_name+':0')
		embedding_weights = embedding_layer.eval()
	return embedding_weights

def find_equivalent(dictionary, query_word, embedding_matrix, max_result=5):
	query_token = dictionary.get(query_word, -1)
	if query_token == -1:
		raise ValueError('Query word is not found in dictionary. Try with another word.')
	# assume embedding_matrix shape is [vocab_size, embedding_dim]
	[vocab_size, embedding_dim] = embedding_matrix.shape
	token_onehot = np.zeros((vocab_size,))
	token_onehot[query_token] = 1

	query_vector = np.dot(token_onehot, embedding_matrix)

	reverse_dictionary = {v:k for k,v in dictionary.items()}
	# go through each word embedding and calculate cosine similarity
	# get some container to store
	words = []
	scores = []

	# Normalize embedding matrix
	embedding_matrix_normalized = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
	# Normalize query vector
	# 1D
	query_vector = query_vector / np.linalg.norm(query_vector)  
	# Make query_vector into matrix for vectorized multiplication

	# Vectorized dot product.
	cos_sim = np.dot(embedding_matrix_normalized, query_vector)
	sorted_idx = np.argsort(cos_sim)
	#                                 :-1 because we don't want the top matching word: which is our query word.
	top_k_idx = sorted_idx[-max_result-1:-1][::-1]
	btm_k_idx = sorted_idx[:max_result]

	# Convert topk and btmk to list and concatenate the list
	result_idx = top_k_idx.tolist() + btm_k_idx.tolist()

	words = [reverse_dictionary[i] for i in result_idx]
	scores = [cos_sim[i] for i in result_idx]

	result = [(words[i], scores[i]) for i in range(len(result_idx))]

	return result 

def analogy(w1, w2, w3, embedding_weights):
	pass
	# it works as such
	# this function finds out what's the word for w3 that is as similar to w2 to w1.
	# eg. 'Paris' to 'France' is similar to 'London' to 'US'
	# where w1 = 'Paris', w2 = 'France', w3 = 'London'

	# TODO: Find idx of w1, w2, and w3

	# TODO: get the vectors for w1, w2, and w3

	# TODO: find the vec_diff between w1 and w2.

	# TODO: w4 = w3 - vec_diff

	# TODO: go through each word and find the cosine similarity.

def main():
	base_dir = "C:/Users/mj/python/word_embeddings/skipgram/models"
	model_dir = 'win_size6'
	ckpt_name = 'skipgram-10-win12.ckpt'
	dictionary_name = 'brown_dictionary.pickle'
	ckpt_directory = base_dir + '/' + model_dir + '/' + ckpt_name
	dictionary_directory = base_dir + '/' + model_dir + '/' + dictionary_name
	scope_name = 'skipgram'
	variable_name = 'embedding_layer'

	parser = argparse.ArgumentParser(description='Calculate the nearest word')
	parser.add_argument('query_word', help='The word you want to query', type=str)

	args = parser.parse_args()

	query_word = args.query_word

	embedding_weights = retrieve_weights(ckpt_directory, scope_name, variable_name)
	with open(dictionary_directory, 'rb') as f:
		dictionary = pickle.loads(f.read())
	result = find_equivalent(dictionary, query_word, embedding_weights, max_result=10)
	print()
	pprint.pprint(result)

if __name__ == '__main__':
	main()
