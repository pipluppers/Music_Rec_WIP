import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


# For files that already have titles
def loadData(fileName):
	return pd.read_csv(fileName)

def main():
	
	# test dataset. Real dataset is too large to actually download. 7377418 rows
	df = pd.read_csv('data_banknote_authentication.csv', delimiter= ',', names=['Variance','Skewness','Curtosis', \
		'Entropy','Target'])
	print(df.head())

	X_train = df.drop('Target')
	Y_train = df['Target']
	nb = GaussianNB()
	nb.fit(X_train, Y_train)
	acc_nb = round(nb.scores(X_train, Y_train) * 100,2)
	print(acc_nb)
	

main()
