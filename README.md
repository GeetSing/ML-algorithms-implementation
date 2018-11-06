# ML-algorithms-implementation

IDA 2016 Challenge (event description: http://ida2016.blogs.dsv.su.se/?page_id=1387)

IDA 2016 Challenge
-------------------
++ About the data

+ Introduction

The dataset consists of data collected from heavy Scania 
trucks in everyday usage. The system in focus is the 
Air Pressure system (APS) which generates pressurised 
air that are utilized in various functions in a truck, 
such as braking and gear changes. The datasets' 
positive class consists of component failures 
for a specific component of the APS system. 
The negative class consists of trucks with failures 
for components not related to the APS. The data consists 
of a subset of all available data, selected by experts.     


+ Attributes

The attribute names of the data have been anonymized for 
proprietary reasons. It consists of both single numerical 
counters and histograms consisting of bins with different 
conditions. Typically the histograms have open-ended 
conditions at each end. For example if we measuring 
the ambient temperature "T" then the histogram could 
be defined with 4 bins where: 

bin 1 collect values for temperature T < -20
bin 2 collect values for temperature T >= -20 and T < 0     
bin 3 collect values for temperature T >= 0 and T < 20  
bin 4 collect values for temperature T > 20 

 |  b1  |  b2  |  b3  |  b4  |   
 ----------------------------- 
       -20     0      20

The attributes are as follows: class, then 
anonymized operational data. The operational data have 
an identifier and a bin id, like "Identifier_Bin".
In total there are 171 attributes, of which 7 are 
histogram variabels. Missing values are denoted by "na".

+ Data characteristics

The data comes in two files, one training set which 
contain the class labels and a test set which has an 
additional id number for each example and "na" as 
class label. Both of these sets are sampled from the 
same source with stratification regarding the class, 
hence they should contain roughly the same distribution 
of faulty and non-faulty components.  

The training set contains 60000 examples in total in which 
59000 belong to the negative class and 1000 positive class. 
The test set contains 16000 examples.


++ Participating

+ Challenge
 
The challenge task is to come up with a good prediction model 
for judging whether or not a vehicle faces imminent failure 
of the specific component or not.
