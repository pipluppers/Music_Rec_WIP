import csv
import random
import math

def loadCsv(fileName):
	# Read in the file in binary mode
	lines = csv.reader(open(fileName, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		# Convert everything to floating point
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

#fileName = 'data_banknote_authentication.csv'
#dataset = loadCsv(fileName)
#print('Loaded data file {0} with {1} rows').format(fileName, len(dataset))


# Split data values into training set and test set
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]


# Separate dataset by class
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]

		# last value (aka vector[-1]) is the class label
		if (vector[-1] not in separated):
			separated[vector[-1]]=[]
		separated[vector[-1]].append(vector)
	return separated



# Mean and standard deviation
def mean(nums):
	return sum(nums)/float(len(nums))

def std(nums):
	avg = mean(nums)
	variance = sum([pow(x - avg,2) for x in nums])/float(len(nums)-1)
	return math.sqrt(variance)


# Calculate the mean and standard deviation of a dataset. Then delete the
#   class value
def summarize(dataset):
	summaries = [(mean(attr), std(attr)) for attr in zip(*dataset)]
	del summaries[-1]
	return summaries

# Calculate the mean and std for the dataset grouped by class
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

# Gaussian normal equation
# P(X | C)
def x_given_y(x, mean, std):
	denom = math.sqrt(2 * math.pi * std**2)
	ee = math.pow(x - mean, 2) / (2 * std**2)
	return (1/(denom)) * math.exp(-ee)


# Returns the maximum a-posteriori P(C | X)
# Summaries are the mean and std of each class
#      e.g. summaries = {0:[(2,0.4)], 1:[(34,7.0)]}
# Calculates the class probabilites P(C = 0) and P(C = 1)
def calculateP0_P1(summaries, inputVec):
	prob = {}
	for classValue, classSummaries in summaries.iteritems():
		prob[classValue] = 1
		for i in range(len(classSummaries)):
			# Retreive mean and std
			mean, std = classSummaries[i]
			# Input Parameters
			x = inputVec[i]
			# Calculate P(X | C) * P(C)
			prob[classValue] *= x_given_y(x, mean, std)
	return prob

# P(C | X) of unknown data point
# Predict the class value given the mean and std and input point
# Return the label with the best probability of occuring
def predict(summaries, inputVec):
	prob = calculateP0_P1(summaries, inputVec)
	# Initialize the best label to nothing and the best probability
	# 	so far to -1
	bestLabel, bestProb = None, -1
	for classValue, probability in prob.iteritems():
		# if just started, assign best probability as first point
		if bestLabel is None or probability > bestProb:
			bestLabel = classValue
			bestProb = probability
	return bestLabel

# Predictions of the class labels for the rest of the data points
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		# res is one P(C|X)
		res = predict(summaries, testSet[i])
		predictions.append(res)
	return predictions

# Get percentage of correct predictions
def evaluate(testSet, predictions):
	numCorrect = 0
	for x in range(len(testSet)):
		if (testSet[x][-1] == predictions[x]):
			numCorrect += 1
	return (numCorrect / float(len(testSet))) * 100


def main():
	fileName = 'data_banknote_authentication.csv'
	splitRatio = 0.67
	dataSet = loadCsv(fileName)
	trainingSet, testSet = splitDataset(dataSet, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataSet), len(trainingSet), len(testSet))

	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = evaluate(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)

main()

