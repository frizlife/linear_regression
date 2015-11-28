import numpy as np
import pandas as pd
import statsmodels.api as sm

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

#cleaning long way
#x = loansData['Interest.Rate'][0:5].values[1]
#x = x.rstrip('%')
#x = float(x)
#x = x/100
#x = round(x, 4)
#print x

cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%'))/100, 4))
loansData['Interest.Rate'] = cleanInterestRate
#print loansData['Interest.Rate'][0:5]

cleanLoansLength = loansData['Loan.Length'][0:5].map(lambda x: int(x.rstrip(' months')))
loansData['Loan.Length'] = cleanLoansLength
#print loansData['Loan.Length'][0:5]

#cleanFICORange = loansData['FICO.Range'].map(lambda x: x.split('-'))
#cleanFICORange = cleanFICORange.map(lambda x: [int(n) for n in x])
#loansData['FICO.Range'] = cleanFICORange
#print loansData['FICO.Range'][0:5]

loansData['FICO.Score'] = [int(val.split('-')[0]) for val in loansData['FICO.Range']]

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']


#transpose data
#The dependent variable
y = np.matrix(intrate).transpose()
# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

#print intrate
#print loanamt
#print fico
x = np.column_stack([x1,x2])
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()
print f.summary()

