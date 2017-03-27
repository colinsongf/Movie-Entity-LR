# Movie-Entity-LR

##### Team member : Yue Chang(yc2966@columbia.edu), Jun Guo(jg3555@columbia.edu), Jieyu Yao(jy2806@columbia.edu)

#### 1. Describe your entity resolution technique, as well as its precision, recall, and F1 score?
##### This algorithm is using logistical regression by three parameters: During time, Directors, Stars. The aim is to identify the different Ids in two files "amazon.csv" and "rotten_tomatoes.csv". The precision is 95% 

#### 2. What were the most important features that powered your technique?
##### The weight of During time is 4.4. The weight of Director is 9.4. The weight of Stars is 0.6. So, the most important feature is "Director"

#### 3. How did you avoid pairwise comparison of all movies across both datasets?
##### No pairwise comparison is ever needed in the process, as both training data and testing data are already given in pairs. For training process, we can directly extract each pair, process, and use matrix operations to perform the machine learning algorithm to find weights; for the testing process, we only need to extract the information of each entry and decide based on a single score. In short, all "comparisons" are point-to-point, and the time complexity (ignoring the data entry and matrix operation) is O(l) instead of O(n,m), with "l" as the length of training/testing list, and n,m, meaning the length of each movie list, respectively.
