#!/usr/bin/env python
# coding: utf-8

# # Train a text classifier using Amazon SageMaker BlazingText built-in algorithm
# 
# ### Introduction
# 
# In this lab you will use SageMaker BlazingText built-in algorithm to predict the sentiment for each customer review. BlazingText is a variant of FastText which is based on word2vec. For more information on BlazingText, see the documentation here:  https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html
# 
# ### Table of Contents
# 
# - [1. Prepare dataset](#c1w4-1.)
#   - [1.1. Load the dataset](#c1w4-1.1.)
#   - [1.2. Transform the dataset](#c1w4-1.2.)
#     - [Exercise 1](#c1w4-ex-1)
#   - [1.3. Split the dataset into train and validation sets](#c1w4-1.3.)
#   - [1.4. Upload the `train` and `validation` datasets to S3 bucket](#c1w4-1.4.)
# - [2. Train the model](#c1w4-2.)
#   - [Exercise 2](#c1w4-ex-2)
#   - [Exercise 3](#c1w4-ex-3)
#   - [Exercise 4](#c1w4-ex-4)
#   - [Exercise 5](#c1w4-ex-5)
#   - [Exercise 6](#c1w4-ex-6)
#   - [Exercise 7](#c1w4-ex-7)
# - [3. Deploy the model](#c1w4-3.)
# - [4. Test the model](#c1w4-4.)
# 
# Let's install and import required modules.

# In[1]:


# please ignore warning messages during the installation
get_ipython().system('pip install --disable-pip-version-check -q sagemaker==2.35.0')
get_ipython().system('pip install --disable-pip-version-check -q nltk==3.5')


# In[2]:


import boto3
import sagemaker
import pandas as pd
import json

sess   = sagemaker.Session()
bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# <a name='c1w4-1.'></a>
# # 1. Prepare dataset
# Let's adapt the dataset into a format that BlazingText understands. The BlazingText format is as follows:
# 
# ```
# __label__<label> "<features>"
# ```
# 
# Here are some examples:
# ```
# __label__-1 "this is bad"
# __label__0 "this is ok"
# __label__1 "this is great"
# ```
# 
# Sentiment is one of three classes: negative (-1), neutral (0), or positive (1).  BlazingText requires that `__label__` is prepended to each sentiment value.
# 
# You will tokenize the `review_body` with the Natural Language Toolkit (`nltk`) for the model training. `nltk` documentation can be found [here](https://www.nltk.org/). You will also use `nltk` later in this lab to tokenize reviews to use as inputs to the deployed model.

# <a name='c1w4-1.1.'></a>
# ### 1.1. Load the dataset

# Upload the dataset into the Pandas dataframe:

# In[4]:


get_ipython().system("aws s3 cp 's3://dlai-practical-data-science/data/balanced/womens_clothing_ecommerce_reviews_balanced.csv' ./")


# In[5]:


path = './womens_clothing_ecommerce_reviews_balanced.csv'

df = pd.read_csv(path, delimiter=',')
df.head()


# <a name='c1w4-1.2.'></a>
# ### 1.2. Transform the dataset
# Now you will prepend `__label__` to each sentiment value and tokenize the review body using `nltk` module. Let's import the module and download the tokenizer:

# In[6]:


import nltk
nltk.download('punkt')


# To split a sentence into tokens you can use `word_tokenize` method. It will separate words, punctuation, and apply some stemming. Have a look at the example:

# In[7]:


sentence = "I'm not a fan of this product!"

tokens = nltk.word_tokenize(sentence)
print(tokens)


# The output of word tokenization can be converted into a string separated by spaces and saved in the dataframe. The transformed sentences are prepared then for better text understending by the model. 
# 
# Let's define a `prepare_data` function which you will apply later to transform both training and validation datasets. 

# <a name='c1w4-ex-1'></a>
# ### Exercise 1
# 
# Apply the tokenizer to each of the reviews in the `review_body` column of the dataframe `df`.

# In[8]:


def tokenize(review):
    # delete commas and quotation marks, apply tokenization and join back into a string separating by spaces
    return ' '.join([str(token) for token in nltk.word_tokenize(str(review).replace(',', '').replace('"', '').lower())])
    
def prepare_data(df):
    df['sentiment'] = df['sentiment'].map(lambda sentiment : '__label__{}'.format(str(sentiment).replace('__label__', '')))
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    df['review_body'] = df['review_body'].map(lambda review : tokenize(review)) # Replace all None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    return df


# Test the prepared function and examine the result.

# In[9]:


# create a sample dataframe
df_example = pd.DataFrame({
    'sentiment':[-1, 0, 1], 
    'review_body':[
        "I do like this product!", 
        "this product is ok", 
        "I don't like this product!"]
})

# test the prepare_data function
print(prepare_data(df_example))

# Expected output:
#      sentiment                   review_body
# 0  __label__-1      i do like this product !
# 1   __label__0            this product is ok
# 2   __label__1  i do n't like this product !


# Apply the `prepare_data` function to the dataset. 

# In[10]:


df_blazingtext = df[['sentiment', 'review_body']].reset_index(drop=True)
df_blazingtext = prepare_data(df_blazingtext)
df_blazingtext.head()


# <a name='c1w4-1.3.'></a>
# ### 1.3. Split the dataset into train and validation sets
# Split and visualize a pie chart of the train (90%) and validation (10%) sets. You can do the split using the `sklearn` model function.

# In[11]:


from sklearn.model_selection import train_test_split

# Split all data into 90% train and 10% holdout
df_train, df_validation = train_test_split(df_blazingtext, 
                                           test_size=0.10,
                                           stratify=df_blazingtext['sentiment'])

labels = ['train', 'validation']
sizes = [len(df_train.index), len(df_validation.index)]
explode = (0.1, 0)  

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle.
ax1.axis('equal')  

plt.show()


# Save the results as CSV files.

# In[12]:


blazingtext_train_path = './train.csv'
df_train[['sentiment', 'review_body']].to_csv(blazingtext_train_path, index=False, header=False, sep=' ')


# In[13]:


blazingtext_validation_path = './validation.csv'
df_validation[['sentiment', 'review_body']].to_csv(blazingtext_validation_path, index=False, header=False, sep=' ')


# <a name='c1w4-1.4.'></a>
# ### 1.4. Upload the `train` and `validation` datasets to S3 bucket
# You will use these to train and validate your model. Let's save them to S3 bucket.

# In[14]:


train_s3_uri = sess.upload_data(bucket=bucket, key_prefix='blazingtext/data', path=blazingtext_train_path)
validation_s3_uri = sess.upload_data(bucket=bucket, key_prefix='blazingtext/data', path=blazingtext_validation_path)


# <a name='c1w4-2.'></a>
# # 2. Train the model
# 
# Setup the BlazingText estimator. For more information on Estimators, see the SageMaker Python SDK documentation here: https://sagemaker.readthedocs.io/.

# <a name='c1w4-ex-2'></a>
# ### Exercise 2
# 
# Setup the container image to use for training with the BlazingText algorithm.
# 
# **Instructions**: Use the `sagemaker.image_uris.retrieve` function with the `blazingtext` algorithm. 
# 
# ```python
# image_uri = sagemaker.image_uris.retrieve(
#     region=region,
#     framework='...' # the name of framework or algorithm
# )
# ```

# In[18]:


image_uri = sagemaker.image_uris.retrieve(
    region=region,
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    framework='blazingtext' # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
)


# <a name='c1w4-ex-3'></a>
# ### Exercise 3
# 
# Create an estimator instance passing the container image and other instance parameters.
# 
# **Instructions**: Pass the container image prepared above into the `sagemaker.estimator.Estimator` function.

# In[19]:


estimator = sagemaker.estimator.Estimator(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    image_uri=image_uri, # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    role=role, 
    instance_count=1, 
    instance_type='ml.m5.large',
    volume_size=30,
    max_run=7200,
    sagemaker_session=sess
)


# Configure the hyper-parameters for BlazingText. You are using BlazingText for a supervised classification task. For more information on the hyper-parameters, see the documentation here:  https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext-tuning.html
# 
# The hyperparameters that have the greatest impact on word2vec objective metrics are: `learning_rate` and `vector_dim`.

# In[20]:


estimator.set_hyperparameters(mode='supervised',   # supervised (text classification)
                              epochs=10,           # number of complete passes through the dataset: 5 - 15
                              learning_rate=0.01,  # step size for the  numerical optimizer: 0.005 - 0.01
                              min_count=2,         # discard words that appear less than this number: 0 - 100                              
                              vector_dim=300,      # number of dimensions in vector space: 32-300
                              word_ngrams=3)       # number of words in a word n-gram: 1 - 3


# To call the `fit` method for the created estimator instance you need to setup the input data channels. This can be organized as a dictionary
# 
# ```python
# data_channels = {
#     'train': ..., # training data
#     'validation': ... # validation data
# }
# ```
# 
# where training and validation data are the Amazon SageMaker channels for S3 input data sources.

# <a name='c1w4-ex-4'></a>
# ### Exercise 4
# 
# Create a train data channel.
# 
# **Instructions**: Pass the S3 input path for training data into the `sagemaker.inputs.TrainingInput` function.

# In[21]:


train_data = sagemaker.inputs.TrainingInput(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    train_s3_uri, # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    distribution='FullyReplicated', 
    content_type='text/plain', 
    s3_data_type='S3Prefix'
)


# <a name='c1w4-ex-5'></a>
# ### Exercise 5
# 
# Create a validation data channel.
# 
# **Instructions**: Pass the S3 input path for validation data into the `sagemaker.inputs.TrainingInput` function.

# In[22]:


validation_data = sagemaker.inputs.TrainingInput(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    validation_s3_uri, # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    distribution='FullyReplicated', 
    content_type='text/plain', 
    s3_data_type='S3Prefix'
)


# <a name='c1w4-ex-6'></a>
# ### Exercise 6
# 
# Organize the data channels defined above as a dictionary.

# In[23]:


data_channels = {
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    'train': train_s3_uri, # Replace None
    'validation': validation_s3_uri # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
}


# <a name='c1w4-ex-7'></a>
# ### Exercise 7
# 
# Start fitting the model to the dataset.
# 
# **Instructions**: Call the `fit` method of the estimator passing the configured train and validation inputs (data channels).
# 
# ```python
# estimator.fit(
#     inputs=..., # train and validation input
#     wait=False # do not wait for the job to complete before continuing
# )
# ```

# In[25]:


estimator.fit(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    inputs=data_channels, # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    wait=False
)

training_job_name = estimator.latest_training_job.name
print('Training Job Name:  {}'.format(training_job_name))


# Review the training job in the console.
# 
# **Instructions**: 
# - open the link
# - notice that you are in the section `Amazon SageMaker` -> `Training jobs`
# - check the name of the training job, its status and other available information

# In[26]:


from IPython.core.display import display, HTML

display(HTML('<b>Review <a target="blank" href="https://console.aws.amazon.com/sagemaker/home?region={}#/jobs/{}">Training job</a></b>'.format(region, training_job_name)))


# Review the Cloud Watch logs (after about 5 minutes).
# 
# **Instructions**: 
# - open the link
# - open the log stream with the name, which starts from the training job name
# - have a quick look at the log messages

# In[27]:


from IPython.core.display import display, HTML

display(HTML('<b>Review <a target="blank" href="https://console.aws.amazon.com/cloudwatch/home?region={}#logStream:group=/aws/sagemaker/TrainingJobs;prefix={};streamFilter=typeLogStreamPrefix">CloudWatch logs</a> (after about 5 minutes)</b>'.format(region, training_job_name)))


# Wait for the training job to complete.
# 
# ### _This cell will take approximately 5-10 minutes to run._

# In[28]:


get_ipython().run_cell_magic('time', '', '\nestimator.latest_training_job.wait(logs=False)')


# Review the train and validation accuracy.
# 
# _Ignore any warnings._

# In[29]:


estimator.training_job_analytics.dataframe()


# Review the trained model in the S3 bucket.
# 
# **Instructions**: 
# - open the link
# - notice that you are in the section `Amazon S3` -> `[bucket name]` -> `[training job name]` (Example: `Amazon S3` -> `sagemaker-us-east-1-82XXXXXXXXXXX` -> `blazingtext-20XX-XX-XX-XX-XX-XX-XXX`)
# - check the existence of the `model.tar.gz` file in the `output` folder

# In[30]:


from IPython.core.display import display, HTML

display(HTML('<b>Review <a target="blank" href="https://s3.console.aws.amazon.com/s3/buckets/{}/{}/output/?region={}&tab=overview">Trained model</a> in S3</b>'.format(bucket, training_job_name, region)))


# <a name='c1w4-3.'></a>
# # 3. Deploy the model
# 
# Now deploy the trained model as an Endpoint.

# ### _This cell will take approximately 5-10 minutes to run._

# In[31]:


get_ipython().run_cell_magic('time', '', "\ntext_classifier = estimator.deploy(initial_instance_count=1,\n                                   instance_type='ml.m5.large',\n                                   serializer=sagemaker.serializers.JSONSerializer(),\n                                   deserializer=sagemaker.deserializers.JSONDeserializer())\n\nprint()\nprint('Endpoint name:  {}'.format(text_classifier.endpoint_name))")


# Review the endpoint in the AWS console.
# 
# **Instructions**: 
# - open the link
# - notice that you are in the section `Amazon SageMaker` -> `Endpoints` -> `[Endpoint name]` (Example: `Amazon SageMaker` -> `Endpoints` -> `blazingtext-20XX-XX-XX-XX-XX-XX-XXX`)
# - check the status and other available information about the Endpoint

# In[32]:


from IPython.core.display import display, HTML

display(HTML('<b>Review <a target="blank" href="https://console.aws.amazon.com/sagemaker/home?region={}#/endpoints/{}">SageMaker REST Endpoint</a></b>'.format(region, text_classifier.endpoint_name)))


# <a name='c1w4-4.'></a>
# # 4. Test the model

# Import the `nltk` library to convert the raw reviews into tokens that BlazingText recognizes.

# In[33]:


import nltk
nltk.download('punkt')


# Specify sample reviews to predict the sentiment.

# In[34]:


reviews = ['This product is great!',
           'OK, but not great',
           'This is not the right product.'] 


# Tokenize the reviews and specify the payload to use when calling the REST API. 

# In[35]:


tokenized_reviews = [' '.join(nltk.word_tokenize(review)) for review in reviews]

payload = {"instances" : tokenized_reviews}
print(payload)


# Now you can predict the sentiment for each review. Call the `predict` method of the text classifier passing the tokenized sentence instances (`payload`) into the data argument.

# In[36]:


predictions = text_classifier.predict(data=payload)
for prediction in predictions:
    print('Predicted class: {}'.format(prediction['label'][0].lstrip('__label__')))


# Upload the notebook into S3 bucket for grading purposes.
# 
# **Note**: you may need to click on "Save" button before the upload.

# In[37]:


get_ipython().system('aws s3 cp ./C1_W4_Assignment.ipynb s3://$bucket/C1_W4_Assignment_Learner.ipynb')


# Please go to the main lab window and click on `Submit` button (see the `Finish the lab` section of the instructions).

# In[ ]:




