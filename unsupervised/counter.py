import tarfile
import csv

tar = tarfile.open('images.tar')

labels = {}
with open('image_labels.csv', newline='') as csvfile:
	reader = csv.reader(csvfile)
	i = 0
	for row in reader:
		if i == 0:
			i = 1
		else:
			labels[row[0]] = row[1]

counts = {}
total = 0

tar_names = tar.getnames()[1:]
for tar_name in tar_names:
	currLabel = labels[tar_name[7:]]
	if currLabel in counts:
		counts[currLabel] += 1
	else:
		counts[currLabel] = 1
	total += 1

for x in counts:
	print(x, ": " , counts[x])
print("total: ", total)

tar.close()
