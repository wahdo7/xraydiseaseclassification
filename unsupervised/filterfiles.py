import tarfile
import csv

tar = tarfile.open('images_001.tar.gz')

allfilenames = []
with open('image_labels.csv', newline='') as csvfile:
	reader = csv.reader(csvfile)
	i = 0
	for row in reader:
		if i == 0:
			i = 1
		else:
			allfilenames.append(row[0])

targetmembers = []
for member in tar.getmembers():
	if member.name[7:] in allfilenames:
		targetmembers.append(member)

tar.extractall(members=targetmembers)

tar.close()
