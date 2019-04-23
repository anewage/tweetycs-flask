# coding: utf-8
#Code snippet
#Remove duplicates AND empty values from csv file
import csv
import pandas as pd
import sys


def remove(inputcsv, outputcsv):
	#Read the csv file into a dataframe
	df=pd.read_csv(inputcsv)

	#drop duplicates using the subset of columns. Retain the first occurrance and delete the subsequent copies. perform all this in place.
	df.drop_duplicates(subset=['text'],keep='first', inplace=True)

	#remove empty content
	df.dropna(axis = 0, subset = ['text'], inplace =True)

	#remove tweets that only contains rt
	df=df[~df['text'].isin(['rt'])]


	#len(df)   #to check the number of records 

	#Reset the index after removing the duplicates. Not required if no additional steps are performed which use index.
	df.reset_index(drop=True,inplace=True)

	#Write the resulting df to a csv file and ignore index
	df.to_csv(outputcsv,index=False)

if __name__ == '__main__':
	csv_file_name = sys.argv[1]
	processed_file_name = sys.argv[1][:-4] + '-unique.csv'
	remove(csv_file_name, processed_file_name)