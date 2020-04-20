import csv
import numpy as np
import math
import random
import sys

from scipy import ndimage
from skimage.io import imread, imshow
from skimage import transform
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize


def preprocessImage(img, scaleFactor, margin, sharpen, blob):
	processed_img = img

	# plt.figure(figsize=(12, 8))
	# plt.subplot(131)
	# plt.imshow(processed_img, cmap = 'gray')

	# if the image has more than one channel, condense them
	if len(processed_img.shape) == 3:
		processed_img = np.zeros((img.shape[0], img.shape[1]))
		processed_img = img[:,:,0] / img[:,:,3]

	# sharpen the image
	if sharpen:
		blurred_img = ndimage.gaussian_filter(processed_img, 2)
		f_blurred_img = ndimage.gaussian_filter(blurred_img, 1)
		alpha = 30
		processed_img = blurred_img * (blurred_img - f_blurred_img)

	if blob:
		processed_img = ndimage.gaussian_filter(processed_img, 2)
		processed_img = processed_img > np.average(processed_img)
		#processed_img, num_objects = ndimage.label(processed_img)

	# cut off margins
	processed_img = processed_img[margin[0] : processed_img.shape[0] - margin[0], margin[1] : processed_img.shape[1] - margin[1]]

	# resize the image
	processed_img = transform.resize(processed_img, (math.floor(processed_img.shape[0] * scaleFactor), math.floor(processed_img.shape[1] * scaleFactor)), anti_aliasing=False)

	#processed_img = transform.rescale(processed_img, scale=scaleFactor)

	# plt.subplot(132)
	# plt.imshow(processed_img, cmap = 'gray')
	# plt.show()

	processed_img = normalize(processed_img)
	
	return processed_img


test_id = random.randint(100, 1000)
print()
print("Starting test id: ", test_id)

train_proportion = 0.025
eval_proportion = 0.025
scale = 0.125
margin = (50, 50)
num_clusters = 16
sharpen = False
blob = False
even_sampling = False

data = None
with open('image_labels_single_pa.csv', newline='') as csvfile:
	print()
	reader = csv.reader(csvfile)
	image_count = sum(1 for row in reader) - 1
	csvfile.seek(0)

	# Pick images to train on
	print("Starting population sampling for training images.")
	print(". . .")
	train_number = math.floor(image_count * train_proportion)
	train_files = []

	if even_sampling:
		with open('image_labels_nofinding_pa.csv', newline='') as nofinding_csvfile:
			nofinding_reader = csv.reader(nofinding_csvfile)
			nofinding_image_count = sum(1 for row in nofinding_reader) - 1
			nofinding_csvfile.seek(0)
			nofinding_train_indices = random.sample(range(1, nofinding_image_count + 2), math.floor(train_number / 3))
			for i, row in enumerate(nofinding_reader):
				if i in nofinding_train_indices:
					train_files.append(row[0])
			with open('image_labels_infiltration_pa.csv', newline='') as infiltration_csvfile:
				infiltration_reader = csv.reader(infiltration_csvfile)
				infiltration_image_count = sum(1 for row in infiltration_reader) - 1
				infiltration_csvfile.seek(0)
				infiltration_train_indices = random.sample(range(1, infiltration_image_count + 2), math.floor(train_number / 3))
				for i, row in enumerate(infiltration_reader):
					if i in infiltration_train_indices:
						train_files.append(row[0])
				with open('image_labels_effusion_pa.csv', newline='') as effusion_csvfile:
					effusion_reader = csv.reader(effusion_csvfile)
					effusion_image_count = sum(1 for row in effusion_reader) - 1
					effusion_csvfile.seek(0)
					effusion_train_indices = random.sample(range(1, effusion_image_count + 2), math.floor(train_number / 3))
					for i, row in enumerate(effusion_reader):
						if i in effusion_train_indices:
							train_files.append(row[0])

	else:
		train_indices = random.sample(range(1, image_count + 2), train_number)
		for i, row in enumerate(reader):
			if i in train_indices:
				train_files.append(row[0])

	print("Sampling complete. Training population size: ", len(train_files))
	print()


	# Preprocess images and add them to data array
	print("Starting data preprocessing and compilation.")
	print(". . .")
	for j in range(len(train_files)):

		img = imread("images/" + train_files[j])

		# plt.figure(figsize=(12, 8))
		# plt.subplot(131)
		# plt.imshow(img, cmap='gray')

		img = preprocessImage(img, scale, margin, sharpen, blob)

		# plt.subplot(132)
		# plt.imshow(img, cmap='gray')
		# plt.show()
		# break

		if data is None:
			data = np.empty((train_number, img.shape[0] * img.shape[1]))
		data[j] = img.flatten()
		sys.stdout.write("Progress: " + str(j) + " | " + str(train_number))
		sys.stdout.flush()
		sys.stdout.write('\r')
		sys.stdout.flush()
	print("Data processing and compilation complete. Data array shape: ", data.shape)
	print()


	prePCAdims = data.shape[1]
	# Perform PCA to reduce dimensionality
	print("Starting PCA.")
	print(". . .")
	pca = PCA(0.99)
	data = pca.fit_transform(data)
	print("PCA complete. New data array shape: ", data.shape)
	print()
	postPCAdims = data.shape[1]


	# Create GMM classifier based on training data
	print("Starting GMM classifier training.")
	print(". . .")
	gmm = GaussianMixture(n_components=num_clusters, covariance_type='diag', random_state=0)
	gmm.fit_predict(data)
	print("GMM classifier training complete. Weights array shape: ", gmm.weights_.shape)
	print()


	# Store data for evaluation
	eval_data = None
	eval_labels = []


	# Run classifier on full data set
	print("Running classifier on full data set.")
	print(". . .")
	prediction_data = []
	for k in range(gmm.weights_.shape[0]):
		thisdict = {}
		thisdict["num_cluster_members"] = 0
		thisdict["num_healthy"] = 0
		thisdict["num_effusion"] = 0
		thisdict["num_infiltration"] = 0
		prediction_data.append(thisdict)

	csvfile.seek(0)
	for l, row in enumerate(reader):
		if l > 0:
			pred_img = imread("images/" + row[0])
			actual_label = row[1]

			pred_img = preprocessImage(pred_img, scale, margin, sharpen, blob)

			this_data = pred_img.flatten()
			this_data = pred_img.reshape(1, -1)
			this_data2 = pca.transform(this_data)

			prediction = gmm.predict(this_data2)[0]

			# Randomly store some data points for evaluation
			if random.randint(0, int(round(1.0 / eval_proportion, 0))) <= 1:
				if eval_data is None:
					eval_data = np.empty((1, len(this_data2[0])))
					eval_data[0] = this_data2
				else:
					new_eval_data = np.empty((len(eval_data) + 1, len(this_data2[0])))
					new_eval_data[1:] = eval_data
					new_eval_data[0] = this_data2
					eval_data = new_eval_data
				eval_labels.append(prediction)

			thisdict = prediction_data[prediction]
			thisdict["num_cluster_members"] += 1
			if row[1] == 'No Finding':
				thisdict["num_healthy"] += 1
			elif row[1] == 'Effusion':
				thisdict["num_effusion"] += 1
			elif row[1] == 'Infiltration':
				thisdict["num_infiltration"] += 1
			sys.stdout.write("Progress: " + str(l) + " | " + str(image_count))
			sys.stdout.flush()
			sys.stdout.write('\r')
			sys.stdout.flush()
	print("Running classifier on full data set complete.")
	print()

	# Create a graph of the results
	clusters_healthy = []
	clusters_effusion = []
	clusters_infiltration = []

	for n in range(gmm.weights_.shape[0]):
		thisdict = prediction_data[n]
		clusters_healthy.append(thisdict["num_healthy"])
		clusters_effusion.append(thisdict["num_effusion"])
		clusters_infiltration.append(thisdict["num_infiltration"])

	N = gmm.weights_.shape[0]
	ind = np.arange(N)
	width = 0.35

	p1 = plt.bar(ind, clusters_healthy, width)
	p2 = plt.bar(ind, clusters_effusion, width, bottom=clusters_healthy)
	p3 = plt.bar(ind, clusters_infiltration, width, bottom=np.add(clusters_healthy, clusters_effusion).tolist())

	plt.xlabel('Cluster Index')
	plt.ylabel('Count')
	plt.title('GMM Results')
	plt.xticks(ind)
	plt.legend((p1[0], p2[0], p3[0]), ('No Finding (Healthy)', 'Effusion', 'Infiltration'))

	plt.savefig("results/GMM_" + str(test_id) + "_results_graph.png")

	# Output text results
	text_output = open("results/GMM_" + str(test_id) + "_results_text.txt", "w")
	text_output.write("Test ID: " + str(test_id) + "\n")
	text_output.write("\n")
	text_output.write("----- Preprocessing Stats -----\n")
	text_output.write("Image scale: " + str(scale) + "\n")
	text_output.write("Margins: " + str(margin[0]) + " " + str(margin[1]) + "\n")
	text_output.write("Sharpen: " + str(sharpen) + "\n")
	text_output.write("Blob Extraction: " + str(blob) + "\n")
	text_output.write("Number of features pre-PCA: " + str(prePCAdims) + "\n")
	text_output.write("Number of features post-PCA: " + str(postPCAdims) + "\n")
	text_output.write("\n")
	text_output.write("----- Training Stats -----\n")
	text_output.write("Full data set size: " + str(image_count) + "\n")
	text_output.write("Training data set size: " + str(len(train_files)) + "\n")
	text_output.write("Even sampling: " + str(even_sampling) + "\n")
	text_output.write("Number of clusters: " + str(gmm.weights_.shape[0]) + "\n")
	text_output.write("\n")
	text_output.write("----- Evaluation Stats -----\n")
	text_output.write("Silhouette Coefficient: " + str(silhouette_score(eval_data, eval_labels)) + "\n")
	text_output.write("Number of samples used in Silhouette calculation: " + str(len(eval_labels)) + "\n")
	text_output.write("Davies-Bouldin Index: " + str(davies_bouldin_score(eval_data, eval_labels)) + "\n")
	text_output.write("Number of samples used in Davies-Bouldin calculation: " + str(len(eval_labels)) + "\n")
	text_output.write("\n")
	text_output.write("----- Cluster Stats -----\n")
	for m in range(len(prediction_data)):
		text_output.write("Cluster " + str(m) + "\n")
		for x in prediction_data[m]:
			text_output.write(str(x) + ": " + str(prediction_data[m][x]) + "\n")
		text_output.write("\n")

