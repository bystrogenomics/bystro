import sys

f0= open(sys.argv[1],'r') #this will be list of IDs w/ ancestry
f1= open(sys.argv[2],'r') #this will be fam file
out = open(sys.argv[3],'w')
x=f0.readline() #skips the first line


list0=[]

dict = {}

for line in f0:
    li=line.split()
    IDval = str(li(0))
    dict[IDval] = li(2)
for line1 in f1:
    li1=line1.split()
    if li1(0) in dict:
        out.write(line1+'\t'+dict[li1(0)]+'\n')