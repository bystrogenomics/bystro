#Global Ancestry with 1k genomes

import pandas as pd
import numpy as np
import torch
import sys

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

#Load the vcf file and ancestry information file
#genofile = '1kGPfiltered/pruned500k1kGP.vcf'
#AncstFile = 'igsr_IDwAncst_subset-Copy1.txt'
#AncstFile = 'igsr_IDwAncst.txt'
genofile = open(sys.argv[1],'r')
AncstFile = open(sys.argv[2],'r')
log = open(sys.argv[3],'w')
filename = sys.argv[4]



#Read in both files
headervcf=pd.read_csv(genofile,delimiter='\t',skiprows=25)
IDswAncst = pd.read_csv(AncstFile,delimiter = ',')



#Turn genotypes into dosage format for PCA
dosagevcf = headervcf.replace('0/0', 0)
dosagevcf = dosagevcf.replace('0/1', 1)
dosagevcf = dosagevcf.replace('1/0', 1)
dosagevcf = dosagevcf.replace('1/1', 2)


#Separate the genotype dosages
IDs = headervcf.iloc[1]
ChromPos = dosagevcf.iloc[:, 0:9]
Genos = dosagevcf.iloc[:, 9:]


#Separate IDs out as a column by just pulling out one row
IDsascol = Genos.iloc[1]
#Transpose the headers into a column
IDsascol = IDsascol.T.reset_index()
#Pull out just the converted header column
IDsascol = IDsascol.iloc[:,0]



#Convert ints to floats and remove excess whitespace for converting to tensor
Genos = Genos.astype(float)
Genos.columns = Genos.columns.str.strip()



#Find the rows with missing data
missing_mask = np.isnan(Genos).any(axis=1)


#Remove the missing data
Genos = Genos[~missing_mask]


#Convert to tensor
vcfdosages_tensor = torch.tensor(Genos.values)


#Run the low rank pca
#mat2 will be the matrix with PCs by ID 
#q is number of PCs

mat1,eigenval,mat2=torch.pca_lowrank(vcfdosages_tensor, q=10, center=True, niter=5)


#Log the eigenvalues
log.write('Eigenvals:'+str(eigenval)+'\n')

#Log size of PC matrix
log.write('Matrix size:'+str(mat2.size())+'\n')


#Convert matrix with PCs back to pandas and add back IDs
px = pd.DataFrame(mat2.numpy())
IDsWpcs = pd.concat([IDsascol, px], axis = 1)
IDsWpcs = IDsWpcs.rename(columns={'index':'Sample name'})


IDsAncPCA=pd.merge(IDswAncst,IDsWpcs, how='right', on='Sample name')


IDsWoAnc = IDsWpcs[~IDsWpcs['Sample name'].isin(IDswAncst['Sample name'])]


#Log population sizes
log.write('IDs per population group:'+'\n'+str(IDsAncPCA.value_counts('Superpopulation code'))+'\n')

#IDsAncPCA

#Define X as the PCs w ancestry
X_train = IDsAncPCA.loc[:, 0:9]

#Define X_test as the sample IDs wo ancestry
X_test = IDsWoAnc.loc[:, 0:9]


#Define Y as Populations after converting to numerical with ohe
ohe = OneHotEncoder(sparse = False)
y = ohe.fit_transform(IDsAncPCA[['Superpopulation code']])[:3202]


#Keep track of categories to add back later
SuperpopCat = ohe.categories_


#Run the linear regression
reg = LinearRegression().fit(X_train.values, y)


#Check the score of the model
#reg.score(X, y)


#Predict the ancestry of the samples
predbeta = reg.predict(X_test)


#Add the names of the populations onto the prediction
df = pd.DataFrame(predbeta, columns = SuperpopCat[0].tolist())


#Add the predicted max coeff and population as a column
df['max_value'] = df.apply(max, axis=1)
df['max_col'] = df.apply(lambda x: x.idxmax(), axis=1)


#Merge the sample IDs with predicted ancestry
cols_to_keep = ['Sample name']
cols_to_keepdf = ['max_value','max_col']
PredAncestry = pd.concat([IDsWoAnc[cols_to_keep], df[cols_to_keepdf]], axis=1)

#PredAncestry

PredAncestry.to_csv(filename, index=False)

#Example: python GlobalAncestry.py 1kGPfiltered/pruned500k1kGP.vcf igsr_IDwAncst_subset-Copy1.txt log.txt PredAncst.csv