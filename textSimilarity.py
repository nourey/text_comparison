#importing librarires
import os
import decimal 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#setting the number of digits after comma  
decimal.getcontext().prec = 2

#importing paths for text files for db and user
test_file_path = "path/to/input/file"
db_path = "path/to/db/files"

#reading the question file for comparison
dirListing = os.listdir(test_file_path)
dirListing = str(dirListing)
dirListing = dirListing[2:-2]
my_file = open(test_file_path+dirListing,"r")
my_file = my_file.read()
print(my_file)


#comparing user's question with our question's whose stored in our db 
dirListing = os.listdir(db_path)
print("Input file\n",my_file)
for items in dirListing:
        if ".txt" in items:
            db_file = open(db_path+"/"+items,"r")
            file_content = db_file.read()
        documents=[file_content,my_file]
        tfidf = TfidfVectorizer().fit_transform(documents)
        pairwise_similarity = tfidf * tfidf.T
        results=pairwise_similarity.toarray()
        print("Similarity score is: ",np.around(results[0,1],decimals=2))
        if np.around(results[0,1], decimals=2) == 1:
            print("**********Same text has been found**********")
            print("Input file\n",my_file)
            print("**********Same text has been found**********")
        print("Compared DB file\n:",file_content)

