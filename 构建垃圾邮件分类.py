#1.1了解数据集
import pandas as pd
df=pd.read_table("smsspamcollection/SMSSpamCollection",
                 sep="\t",
                 header=None,
                 names=["label","sms_message"])
#output printing out first 5 columns
df.head()

#1.2数据预处理
#大概了解数据集的结构之后，现在将标签转换为二元变量，0表示"ham"(即非垃圾信息)，1表示"spam",这样计算比较方便计算
#因为scikit-learn处理输入的方式，scikit-learn只处理数字值，因此如果标签值保留为字符串，scikit-learn会自己进行转换
#（字符创标签将转型为未知浮点值）
#如果标签保留为字符串，模型依然能够做出预测，但是稍后计算效果指标（例如计算精确度和召回率分数）的时候可能会遇到问题，
#因此，为了避免少受出现意外的陷阱，最好将分类值转换为整数，再传入模型中。
#说明
#使用映射方法将"标签"列中的数值转换为数字值，如下所示：{"ham":0,"spam":1}这样会将"ham"值映射为0，将“spam”值映射为1
#此外，为了知道我们正在处理的数据集有多大，使用"shape"输出行数和列数
df["label"]=df.label.map({"ham":0,"spam":1})
print(df.shape)
df.head()
df["label"]=df.label.map({"ham":0,"spam":1})
print(df.shape)
df.head()
#2.1Bag of words
#数据集中有大量文本数据（5572）大多数机器学习算法都要求传入的输入是数字数据，而电子邮件、信息通常都是文本
#现在要介绍Bag of Words这个概念，它用来表示要处理的问题具有“大量单词”或很多文本数据。BOW的基本概念是拿出一段文本，
#计算该文本中单词的出现频率，注意：BOW平等的对待每一个单词，单词的出现顺序并不重要
#将文档集合转换成矩阵，每个文档是一行，每个单词（令牌）是一列，对应的（行，列）值是每个单词或令牌在此文档中出现的频率
#例如假设有四个如下所示的文档
#["Hello,how are you !",
  "Win money,win from home.",
  "Call me now",
  "Hello,Call you tomorrow?"]
  #目标是将这组文本转换为频率分布矩阵，如下所示
  #   are  call  from  hello  home  how  me  money  now  tomorrow  win  you
  #0  1    0     0     1      0     1    0   0      0    0         0    1
  #1  0    0     1     0      1     0    0   1      0    0         2    0
  #2  0    1     0     0      0     0    1   0      1    0         0    0
  #3  0    1     0     1      0     0    0   0      0    1         0    1
  #从上文中可以看出，文档在行中进行了编号，每个单词是一个列名称，相应的值是该单词在文档中出现的频率
  #如何使用一小组文档进行转换
  #要处理这一步，将使用skelarns count vectorizer 该方法作用如下
 #它会令牌化字符串（将字符串划分为单个单词）并未每个令牌设定一个整型ID
 #它会计算每个令牌出现的次数
 #请注意
 #Countvectorizer方法会自动将所有令牌化单词转换为小写形式，避免区分“He”和“he”等单词，为此，它会使用参数lowercase，该参数默认为True
 #它还会忽略所有标点符号，避免区分后面有标点的单词（例如“hello!”）和前后没有标点的同一单词（例如“hello”）。为此，他会使用参数token_patten
 #该参数使用默认正则表达式选择具有2个或多个字母数字字符的令牌
 #要注意的第三个参数是stop_words.停用词是指某个语言中最常用的字词，包括"am"/"an"/"and"/"the"等，通过将此参数值设为english
 #CountVectorizer将自动忽略（输入文本中）出现在scikit-learn中的内置英语停用词列表中的所有单词，这非常有用，
 #因为当我们尝试查找表明是垃圾内容的某些单词时，停用词汇使结论出现偏差
 #2.2从头实现Bag of Words
 在深入了解处理繁重工作的scikit-learn的Bag of Words(BoW)库之前，首先自己实现该步骤，以便了解该库的背后原理
 #第一步，将所有字符串转换成小写形式
 #假设有一个文档集合
 documents=["Hello,how are you!",
            "Win money,win from home.",]
            "Call me now.",
            "Hello,Call hello you tomorrow?"
 from sklearn.feature_ectraction.text import CountVectorizer
 vectorizer=CountVectorizer
 X=vectorizer.fit_transform(documents)
 print(vectorizer.get_feature_names())
 print(X.toarray())
 from sklearn.feature_extraction.text import CountVectorizer
 Vectorizer=CountVectorizer()
 X=Vectorizer.fit_transform(documents)
 print(Vectorizer.get_feature_names())
 print(X>toarray())
 #说明
 #将文档集合中的所有字符串转换成小写形式，将他们保存到焦作"lower_case_documents"的列表中，可以使用loer()方法
 #在Python中将字符串转换成小写形式
 documents=["Hello,how are you !",
            "Win money,win from home.",
            "Call me now.",
            "Hllow,Call hello you tomorrow!"]
lower_case_documents=[]
for i in documents:
     lower_case_documents.append(i.lower())
 print(lower_case_documents)
 lower_case_documents=[]
 for i in documents:
     lower_case_documents.append(i.lower())
print(lower_case_documents)
# 第二部，删除所有标点符号
#删除文档集合中的字符串中的所有标点。将他们保存在焦作"sams_punctuation_documents"的列表中
sans_punctuation_documents=[]
import string
for i in lower_case_documents:
    sana_punctuation_documents.append(i.translate(str.maketrans("","",string.punctuation)))
print(sans_punctuation_documents)
sans_punctuation_documents=[]
import string
for i in lower_case_documents:
    sans_punctuation_documtments.append(i.translate(str.maketrans("","",string.punctuation)))
#第三部，令牌华
#令牌华文档集合中的句子是指使用分隔符将句子拆分成单个单词。分隔符指定了我们将使用哪个字符来表示单词的开始和结束为止
#（例如，可以使用一个空格作为文档结合的单词分隔符）
#说明使用split()方法令牌华sans_punctuation_documents中存储的字符串，并将最终文档集合存储在叫做preprocessed_documents的列表中
preprocessed_documents=[]
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(" "))
print(preprocessed_documents)
preprocessed_documents=[]
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(" "))
print(preprocessed_documents)
#第四步计算频率
#获得所需格式的文档集合，现在可以数出每个单词在文档集合的每个文档中出现的次数了。为此，将使用Python  collections库中的Counter方法
#Counter会数出列表中每项的出现次数，并返回一个字典，键是被数的项目，相应的值是该项目在列表中的计数
#说明：使用Counter()方法和作为输入的preprocessed_documents创建一个字典，键是每个文档中的每个单词，
#相应的值是该单词的出现频率，将每个Counter字典当做项目另存到一个叫做frequency_list的列表中
frequency_list=[]
import pprint
from collections import Counter
for i in preprocessed_documents:
    frequency_count=Counter(i)
    frequency_list.append(frequency_count)
pprint.pprint(frequency_list)
frequency_list=[]
import collecions import Counter
for i in preprocessed_documents:
    frequency_count=Counter(i)
    frequency_list.append(frequency_count)
    #2.3在scikit-learn中实现Bag of Words
    '''
    Here we will look to create a frequency matrix on a smaller document set to make sure we understand how 
    the documentter matrix generation happends.We hae created a sample document set "documents"
    documents=["Hello,how are you!",
               "Win money,win from home.",
               "Call me now.",
               "Hello,Call hello you tomorrow?"]
#说明导入sklearn.feature_extraction.text.Countectorizer方法并创建一个实例，命名为'count_vector
from sklearn.feature_extraction.text import CountVectorizer
counter_ector=CountVectorizer()
from sklearn.feature_extraction.text import CountVectorizer
count_extor=CountVectorizer()
#在2.2，从头实现了可以首先清晰数据的额CountVectorizer方法。清理过程包括将所有数据转换为小写形式，并删除所有标点符号。
#CountVectorizer（）具有某些可以帮助我们完成这些步骤的参数，这些参数包括
#lowercase=True
#lowercase参数默认值为True,它会将所有文本都转换为小写形式
#token_pattern=(?u)\\b\\w\\w+\\b
#token_pattern 参数具有默认正则表达式(?u)\\b\\w\\w+\\b，它会忽略所有标点符号并将他们当做分隔符，并将长度大于等于2的字母数字字符串当做令牌或单词
#stop_words参数如果设为“english”，将文档集合中删除scikit-learn中定义的英语停用词列表匹配的所有单词。考虑到我们的额数据集规模不大，
#并且处理的是信息，并不是电子邮件这样的更庞大文本来源，一次将不设置此参数值
#通过如下所示数出count_Vector 对象，查看该对象的所有参数值：
'''
practice node:
print the count_Vector object which is an instance of CountVectorizer（）
'''
print(count_Vector)
#使用fit()将你的文档数据集与CountVectorizer对象拟合，并使用get_feature_names()方法获得被归类为特征的单词列表
count_vector.fit(documents)
count_vector.get_feature_names()
ecount_ectro.fit(documents)
count_ectro.get_feature_names()
#get_feature_names()方法会返回此数据集的特征名称，即组成documents词汇表的单词集合
#说明创建一个矩阵，行是4个文档中每个文档的行，列是每个单词，对应的值（行，列）是该单词（在列中）在特定文档（在行中）
#中出现的频率，为此，你可以使用transform()方法并传入文档数据集作为参数。transform（）方法会返回一个numpy
#整数矩阵，你可以使用toarray()将其转换为数组，称之为doc_array
doc_array=count_vector.transform(documents).toarray()
doc_array
doc_array=count_ector.transform(documents).toarray()
doc_array
#现在，对于单词在文档中的出现频率，已经获得了整洁的文档表示形式。为了方便理解，下一步会将此数组转换为数据帧，并相应地列命名
#说明将获得并加载到doc_array中的数组转换为数据帧，并将列名设为单词名称（之前使用get_feature_names()计算了名称）。
#将该数据帧命名为frequency_matrix
frequency_matrix=pd.DataFrame(doc_array,columns=count_vector.get_feature_names())
frequecny_matrix
frequency_matrix=pd.DataFrame(doc_array,columns=count_extor.get_feature_names())
frequency_matrix
#第3.1训练集和测试集
#现在回到数据集并继续分析工作，第一步是将数据集拆分为训练集和测试集，一边稍后测试模型
#说明通过在sklearn中使用train_test_split方法，将数据集拆分为训练集和测试集，使用一下变量拆分数据
#X_train 是sms_message列的训练数据
#y_train是label列的训练数据
#X_test是sms_message列的测试数据
#y_test是label列的测试数据。数出每个训练数据和测试数据的行数
'''
NOTE:sklearn.cross_validation import train_test_split
'''
X_train,X_test,y_train,y_test=train_test_split(df["sms_message"],
                                               df["label"],
                                               random_state=1)
print("Number of rows in the total set:{}".format(df.shape[0]))
print("Number of rows in the training set:{}".format(X_train.shape[0]))
print("Number of rows in the test set:{}".format(X_test.shape[0]))
from skearn.cross_alidation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df["sms_message"],
                                               df["label"],
                                               random_state=1)
print("Number of rows in the total set:{}".format(df.shape[0]))
print("Number of rows in the training set:{}".format(X_trian.shape[0]))
print("Number of rows in the test set:{}".format(X_test.shpae[0]))
#3.2对数据集应用Bag of Words流程
#数据已经拆分了，下个目标是按照第二部：Bag of words 中的步骤操作，并将数据转换为期望的矩阵格式。为此，将像之前一样使用CountVectorizer()
#首先，需要对CountVectorizer()拟合训练数据X_train并返回矩阵
#其次，需要转换测试数据（X_test）以返回矩阵
#注意X_train是数据集中sms_message列的训练数据，我们将使用此数据训练模型
#X_test是sms_message列的测试数据，将使用该数据（转换为矩阵后）进行预测，然后再后面的步骤中这些预测与y_test进行比较
#Instantiate the CountVectorizer method
count_vector = CountVectorizer()
#Fit the training data and then return the matrix
training_data=count_vector.fit_transform(X_train)
#trainsform testing data and return the matrix.Note we are not fitting the testing data into the CountVectorizer()
testing_data=count_vector.transform(X_test)
count_vector=CountVectorizer()
training_data=count_vector.fit_trainsform(X_train)
testing_data=count_vector.transform(X_test)
第4.1步，从头实现byies定理

