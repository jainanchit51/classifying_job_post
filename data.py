import json
from bs4 import BeautifulSoup
import unicodedata
import re
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

def remove_special_characters(text, remove_digits=False):
	pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
	text = re.sub(pattern, '', text)
	return text


with open('data/sampleJobDataWithTags.json') as json_data:
	training_data = json.load(json_data)
	print("Total Training samples",len(training_data))


	#getting unique tags
	unique_tags = []
	for d in training_data:
		for tag in d["tags"]:
			if tag not in unique_tags:
				unique_tags.append(tag)

	print("Total Unique Tags",len(unique_tags))
	
	#finding tags distributions(finding unique tags)
	distribution = {}
	for tag in unique_tags:
		for data in training_data:
			if tag in data["tags"]:
				if tag in distribution.keys():
					distribution[tag] += 1
				else:
					distribution[tag] = 1


	#data cleansing
	for data in training_data:
		#remove html tags
		data["title"] = BeautifulSoup(data["title"], "html.parser").get_text()
		data["description"] = BeautifulSoup(data["description"], "html.parser").get_text()
		#remove accented data
		data["title"] = unicodedata.normalize('NFKD', data["title"]).encode('ascii', 'ignore').decode('utf-8', 'ignore')
		data["description"] = unicodedata.normalize('NFKD', data["description"]).encode('ascii', 'ignore').decode('utf-8', 'ignore')
		#remove special characters
		data["title"]  = remove_special_characters(data["title"])
		data["description"]  = remove_special_characters(data["description"])
		#converting to lowercase
		data["title"] = data["title"].lower()
		data["description"] = data["description"].lower()

	
	splitted_training_data = []
	for data in training_data:
		for tag in data["tags"]:
			splitted_training_data.append({"title": data["title"], "description": data["description"], "tag": tag})

	training_data= splitted_training_data
	print(training_data[0])

	#converting json to list
	titles= []
	descriptions = []
	tags =[]
	for data in training_data[:samples]:
		for tag in data["tags"]:
			titles.append(data["title"])
			descriptions.append(data["description"])
			tags.append(tag)

	print(len(titles))

	# Vectorization using Google's Universal Sentence Encoder
	module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

	# Import the Universal Sentence Encoder's TF Hub module
	embed = hub.Module(module_url)
	
	# Reduce logging output.
	tf.logging.set_verbosity(tf.logging.ERROR)

	with tf.Session() as session:
		session.run([tf.global_variables_initializer(), tf.tables_initializer()])
		titles_vectors = session.run(embed(titles))
		descriptions_vectors = session.run(embed(descriptions))

   

