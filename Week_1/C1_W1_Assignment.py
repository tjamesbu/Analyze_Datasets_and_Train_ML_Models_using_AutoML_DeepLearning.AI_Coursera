#!/usr/bin/env python
# coding: utf-8

# # Register and visualize dataset
# 
# ### Introduction
# 
# In this lab you will ingest and transform the customer product reviews dataset. Then you will use AWS data stack services such as AWS Glue and Amazon Athena for ingesting and querying the dataset. Finally you will use AWS Data Wrangler to analyze the dataset and plot some visuals extracting insights.

# ### Table of Contents
# 
# - [1. Ingest and transform the public dataset](#c1w1-1.)
#   - [1.1. List the dataset files in the public S3 bucket](#c1w1-1.1.)
#     - [Exercise 1](#c1w1-ex-1)
#   - [1.2. Copy the data locally to the notebook](#c1w1-1.2.)
#   - [1.3. Transform the data](#c1w1-1.3.)
#   - [1.4 Write the data to a CSV file](#c1w1-1.4.)
# - [2. Register the public dataset for querying and visualizing](#c1w1-2.)
#   - [2.1. Register S3 dataset files as a table for querying](#c1w1-2.1.)
#     - [Exercise 2](#c1w1-ex-2)
#   - [2.2. Create default S3 bucket for Amazon Athena](#c1w1-2.2.)
# - [3. Visualize data](#c1w1-3.)
#   - [3.1. Preparation for data visualization](#c1w1-3.1.)
#   - [3.2. How many reviews per sentiment?](#c1w1-3.2.)
#     - [Exercise 3](#c1w1-ex-3)
#   - [3.3. Which product categories are highest rated by average sentiment?](#c1w1-3.3.)
#   - [3.4. Which product categories have the most reviews?](#c1w1-3.4.)
#     - [Exercise 4](#c1w1-ex-4)
#   - [3.5. What is the breakdown of sentiments per product category?](#c1w1-3.5.)
#   - [3.6. Analyze the distribution of review word counts](#c1w1-3.6.)

# Let's install the required modules first.

# In[1]:


# please ignore warning messages during the installation
get_ipython().system('pip install --disable-pip-version-check -q sagemaker==2.35.0')
get_ipython().system('pip install --disable-pip-version-check -q pandas==1.1.4')
get_ipython().system('pip install --disable-pip-version-check -q awswrangler==2.7.0')
get_ipython().system('pip install --disable-pip-version-check -q numpy==1.18.5')
get_ipython().system('pip install --disable-pip-version-check -q seaborn==0.11.0')
get_ipython().system('pip install --disable-pip-version-check -q matplotlib===3.3.3')


# <a name='c1w1-1.'></a>
# # 1. Ingest and transform the public dataset
# 
# The dataset [Women's Clothing Reviews](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews) has been chosen as the main dataset.
# 
# It is shared in a public Amazon S3 bucket, and is available as a comma-separated value (CSV) text format:
# 
# `s3://dlai-practical-data-science/data/raw/womens_clothing_ecommerce_reviews.csv`

# <a name='c1w1-1.1.'></a>
# ### 1.1. List the dataset files in the public S3 bucket
# 
# The [AWS Command Line Interface (CLI)](https://awscli.amazonaws.com/v2/documentation/api/latest/index.html) is a unified tool to manage your AWS services. With just one tool, you can control multiple AWS services from the command line and automate them through scripts. You will use it to list the dataset files.

# **View dataset files in CSV format**

# ```aws s3 ls [bucket_name]``` function lists all objects in the S3 bucket. Let's use it to view the reviews data files in CSV format:

# <a name='c1w1-ex-1'></a>
# ### Exercise 1
# 
# View the list of the files available in the public bucket `s3://dlai-practical-data-science/data/raw/`.
# 
# **Instructions**:
# Use `aws s3 ls [bucket_name]` function. To run the AWS CLI command from the notebook you will need to put an exclamation mark in front of it: `!aws`. You should see the data file `womens_clothing_ecommerce_reviews.csv` in the list.

# In[5]:


### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
get_ipython().system('aws s3 ls s3://dlai-practical-data-science/data/raw/ # Replace None')
### END SOLUTION - DO NOT delete this comment for grading purposes

# EXPECTED OUTPUT
# ... womens_clothing_ecommerce_reviews.csv


# <a name='c1w1-1.2.'></a>
# ### 1.2. Copy the data locally to the notebook

# ```aws s3 cp [bucket_name/file_name] [file_name]``` function copies the file from the S3 bucket into the local environment or into another S3 bucket. Let's use it to copy the file with the dataset locally.

# In[6]:


get_ipython().system('aws s3 cp s3://dlai-practical-data-science/data/raw/womens_clothing_ecommerce_reviews.csv ./womens_clothing_ecommerce_reviews.csv')


# Now use the Pandas dataframe to load and preview the data.

# In[7]:


import pandas as pd
import csv

df = pd.read_csv('./womens_clothing_ecommerce_reviews.csv',
                 index_col=0)

df.shape


# In[8]:


df


# <a name='c1w1-1.3.'></a>
# ### 1.3. Transform the data
# To simplify the task, you will transform the data into a comma-separated value (CSV) file that contains only a `review_body`, `product_category`, and `sentiment` derived from the original data.

# In[9]:


df_transformed = df.rename(columns={'Review Text': 'review_body',
                                    'Rating': 'star_rating',
                                    'Class Name': 'product_category'})
df_transformed.drop(columns=['Clothing ID', 'Age', 'Title', 'Recommended IND', 'Positive Feedback Count', 'Division Name', 'Department Name'],
                    inplace=True)

df_transformed.dropna(inplace=True)

df_transformed.shape


# Now convert the `star_rating` into the `sentiment` (positive, neutral, negative), which later on will be for the prediction.

# In[10]:


def to_sentiment(star_rating):
    if star_rating in {1, 2}: # negative
        return -1 
    if star_rating == 3:      # neutral
        return 0
    if star_rating in {4, 5}: # positive
        return 1

# transform star_rating into the sentiment
df_transformed['sentiment'] = df_transformed['star_rating'].apply(lambda star_rating: 
    to_sentiment(star_rating=star_rating) 
)

# drop the star rating column
df_transformed.drop(columns=['star_rating'],
                    inplace=True)

# remove reviews for product_categories with < 10 reviews
df_transformed = df_transformed.groupby('product_category').filter(lambda reviews : len(reviews) > 10)[['sentiment', 'review_body', 'product_category']]

df_transformed.shape


# In[11]:


# preview the results
df_transformed


# <a name='c1w1-1.4.'></a>
# ### 1.4 Write the data to a CSV file

# In[12]:


df_transformed.to_csv('./womens_clothing_ecommerce_reviews_transformed.csv', 
                      index=False)


# In[13]:


get_ipython().system('head -n 5 ./womens_clothing_ecommerce_reviews_transformed.csv')


# <a name='c1w1-2.'></a>
# # 2. Register the public dataset for querying and visualizing
# You will register the public dataset into an S3-backed database table so you can query and visualize our dataset at scale. 

# <a name='c1w1-2.1.'></a>
# ### 2.1. Register S3 dataset files as a table for querying
# Let's import required modules.
# 
# `boto3` is the AWS SDK for Python to create, configure, and manage AWS services, such as Amazon Elastic Compute Cloud (Amazon EC2) and Amazon Simple Storage Service (Amazon S3). The SDK provides an object-oriented API as well as low-level access to AWS services. 
# 
# `sagemaker` is the SageMaker Python SDK which provides several high-level abstractions for working with the Amazon SageMaker.

# In[14]:


import boto3
import sagemaker
import pandas as pd
import numpy as np

sess   = sagemaker.Session()
# S3 bucket name
bucket = sess.default_bucket()
# AWS region
region = boto3.Session().region_name

# Account ID 
sts = boto3.Session(region_name=region).client(service_name="sts", region_name=region)
account_id = sts.get_caller_identity()['Account']

print('S3 Bucket: {}'.format(bucket))
print('Region: {}'.format(region))
print('Account ID: {}'.format(account_id))


# Review the empty bucket which was created automatically for this account.
# 
# **Instructions**: 
# - open the link
# - click on the S3 bucket name `sagemaker-us-east-1-ACCOUNT`
# - check that it is empty at this stage

# In[15]:


from IPython.core.display import display, HTML

display(HTML('<b>Review <a target="top" href="https://s3.console.aws.amazon.com/s3/home?region={}#">Amazon S3 buckets</a></b>'.format(region)))


# Copy the file into the S3 bucket.

# In[16]:


get_ipython().system('aws s3 cp ./womens_clothing_ecommerce_reviews_transformed.csv s3://$bucket/data/transformed/womens_clothing_ecommerce_reviews_transformed.csv')


# Review the bucket with the file we uploaded above.
# 
# **Instructions**: 
# - open the link
# - check that the CSV file is located in the S3 bucket
# - check the location directory structure is the same as in the CLI command above
# - click on the file name and see the available information about the file (region, size, S3 URI, Amazon Resource Name (ARN))

# In[17]:


from IPython.core.display import display, HTML

display(HTML('<b>Review <a target="top" href="https://s3.console.aws.amazon.com/s3/buckets/{}?region={}&prefix=data/transformed/#">Amazon S3 buckets</a></b>'.format(bucket, region)))


# **Import AWS Data Wrangler**
# 
# [AWS Data Wrangler](https://github.com/awslabs/aws-data-wrangler) is an AWS Professional Service open source python initiative that extends the power of Pandas library to AWS connecting dataframes and AWS data related services (Amazon Redshift, AWS Glue, Amazon Athena, Amazon EMR, Amazon QuickSight, etc).
# 
# Built on top of other open-source projects like Pandas, Apache Arrow, Boto3, SQLAlchemy, Psycopg2 and PyMySQL, it offers abstracted functions to execute usual ETL tasks like load/unload data from data lakes, data warehouses and databases.

# Review the AWS Data Wrangler documentation: https://aws-data-wrangler.readthedocs.io/en/stable/

# In[18]:


import awswrangler as wr


# **Create AWS Glue Catalog database**

# The data catalog features of **AWS Glue** and the inbuilt integration to Amazon S3 simplify the process of identifying data and deriving the schema definition out of the discovered data. Using AWS Glue crawlers within your data catalog, you can traverse your data stored in Amazon S3 and build out the metadata tables that are defined in your data catalog.
# 
# Here you will use `wr.catalog.create_database` function to create a database with the name `dsoaws_deep_learning` ("dsoaws" stands for "Data Science on AWS").

# In[19]:


wr.catalog.create_database(
    name='dsoaws_deep_learning',
    exist_ok=True
)


# In[20]:


dbs = wr.catalog.get_databases()

for db in dbs:
    print("Database name: " + db['Name'])


# Review the created database in the AWS Glue Catalog.
# 
# **Instructions**:
# - open the link
# - on the left side panel notice that you are in the AWS Glue -> Data Catalog -> Databases
# - check that the database `dsoaws_deep_learning` has been created
# - click on the name of the database
# - click on the `Tables in dsoaws_deep_learning` link to see that there are no tables

# In[21]:


from IPython.core.display import display, HTML

display(HTML('<b>Review <a target="top" href="https://console.aws.amazon.com/glue/home?region={}#catalog:tab=databases">AWS Glue Databases</a></b>'.format(region)))


# **Register CSV data with AWS Glue Catalog**

# <a name='c1w1-ex-2'></a>
# ### Exercise 2
# 
# Register CSV data with AWS Glue Catalog.
# 
# **Instructions**:
# Use ```wr.catalog.create_csv_table``` function with the following parameters
# ```python
# res = wr.catalog.create_csv_table(
#     database='...', # AWS Glue Catalog database name
#     path='s3://{}/data/transformed/'.format(bucket), # S3 object path for the data
#     table='reviews', # registered table name
#     columns_types={
#         'sentiment': 'int',        
#         'review_body': 'string',
#         'product_category': 'string'      
#     },
#     mode='overwrite',
#     skip_header_line_count=1,
#     sep=','    
# )
# ```

# In[35]:


wr.catalog.create_csv_table(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    database='dsoaws_deep_learning', # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    path='s3://{}/data/transformed/'.format(bucket), 
    table="reviews",    
    columns_types={
        'sentiment': 'int',        
        'review_body': 'string',
        'product_category': 'string'      
    },
    mode='overwrite',
    skip_header_line_count=1,
    sep=','
)


# Review the registered table in the AWS Glue Catalog.
# 
# **Instructions**:
# - open the link
# - on the left side panel notice that you are in the AWS Glue -> Data Catalog -> Databases -> Tables
# - check that you can see the table `reviews` from the database `dsoaws_deep_learning` in the list
# - click on the name of the table
# - explore the available information about the table (name, database, classification, location, schema etc.)

# In[36]:


from IPython.core.display import display, HTML

display(HTML('<b>Review <a target="top" href="https://console.aws.amazon.com/glue/home?region={}#">AWS Glue Catalog</a></b>'.format(region)))


# Review the table shape:

# In[37]:


table = wr.catalog.table(database='dsoaws_deep_learning',
                         table='reviews')
table


# <a name='c1w1-2.2.'></a>
# ### 2.2. Create default S3 bucket for Amazon Athena
# 
# Amazon Athena requires this S3 bucket to store temporary query results and improve performance of subsequent queries.
# 
# The contents of this bucket are mostly binary and human-unreadable. 

# In[38]:


# S3 bucket name
wr.athena.create_athena_bucket()

# EXPECTED OUTPUT
# 's3://aws-athena-query-results-ACCOUNT-REGION/'


# <a name='c1w1-3.'></a>
# # 3. Visualize data
# 
# **Reviews dataset - column descriptions**
# 
# - `sentiment`: The review's sentiment (-1, 0, 1).
# - `product_category`: Broad product category that can be used to group reviews (in this case digital videos).
# - `review_body`: The text of the review.

# <a name='c1w1-3.1.'></a>
# ### 3.1. Preparation for data visualization
# 
# **Imports**

# In[39]:


import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# **Settings**

# Set AWS Glue database and table name.

# In[40]:


# Do not change the database and table names - they are used for grading purposes!
database_name = 'dsoaws_deep_learning'
table_name = 'reviews'


# Set seaborn parameters. You can review seaborn documentation following the [link](https://seaborn.pydata.org/index.html).

# In[41]:


sns.set_style = 'seaborn-whitegrid'

sns.set(rc={"font.style":"normal",
            "axes.facecolor":"white",
            'grid.color': '.8',
            'grid.linestyle': '-',
            "figure.facecolor":"white",
            "figure.titlesize":20,
            "text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black",
            "axes.grid":True,
            'axes.labelsize':10,
            'xtick.labelsize':10,
            'font.size':10,
            'ytick.labelsize':10})


# Helper code to display values on barplots:

# **Run SQL queries using Amazon Athena**

# **Amazon Athena** lets you query data in Amazon S3 using a standard SQL interface. It reflects the databases and tables in the AWS Glue Catalog. You can create interactive queries and perform any data manipulations required for further downstream processing.

# Standard SQL query can be saved as a string and then passed as a parameter into the Athena query. Run the following cells as an example to count the total number of reviews by sentiment. The SQL query here will take the following form:
# 
# ```sql
# SELECT column_name, COUNT(column_name) as new_column_name
# FROM table_name
# GROUP BY column_name
# ORDER BY column_name
# ```
# 
# If you are not familiar with the SQL query statements, you can review some tutorials following the [link](https://www.w3schools.com/sql/default.asp).

# <a name='c1w1-3.2.'></a>
# ### 3.2. How many reviews per sentiment?

# Set the SQL statement to find the count of sentiments:

# In[42]:


statement_count_by_sentiment = """
SELECT sentiment, COUNT(sentiment) AS count_sentiment
FROM reviews
GROUP BY sentiment
ORDER BY sentiment
"""

print(statement_count_by_sentiment)


# Query data in Amazon Athena database cluster using the prepared SQL statement:

# In[43]:


df_count_by_sentiment = wr.athena.read_sql_query(
    sql=statement_count_by_sentiment,
    database=database_name
)

print(df_count_by_sentiment)


# Preview the results of the query:

# In[44]:


df_count_by_sentiment.plot(kind='bar', x='sentiment', y='count_sentiment', rot=0)


# <a name='c1w1-ex-3'></a>
# ### Exercise 3
# 
# Use Amazon Athena query with the standard SQL statement passed as a parameter, to calculate the total number of reviews per `product_category` in the table ```reviews```.
# 
# **Instructions**: Pass the SQL statement of the form
# 
# ```sql
# SELECT category_column, COUNT(column_name) AS new_column_name
# FROM table_name
# GROUP BY category_column
# ORDER BY new_column_name DESC
# ```
# 
# as a triple quote string into the variable `statement_count_by_category`. Please use the column `sentiment` in the `COUNT` function and give it a new name `count_sentiment`.

# In[49]:


# Replace all None
### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
statement_count_by_category = """
SELECT product_category, COUNT(sentiment) AS count_sentiment
FROM reviews
GROUP BY product_category
ORDER BY count_sentiment DESC
"""
### END SOLUTION - DO NOT delete this comment for grading purposes
print(statement_count_by_category)


# Query data in Amazon Athena database passing the prepared SQL statement:

# In[50]:


get_ipython().run_cell_magic('time', '', 'df_count_by_category = wr.athena.read_sql_query(\n    sql=statement_count_by_category,\n    database=database_name\n)\n\ndf_count_by_category\n\n# EXPECTED OUTPUT\n# Dresses: 6145\n# Knits: 4626\n# Blouses: 2983\n# Sweaters: 1380\n# Pants: 1350\n# ...')


# <a name='c1w1-3.3.'></a>
# ### 3.3. Which product categories are highest rated by average sentiment?

# Set the SQL statement to find the average sentiment per product category, showing the results in the descending order:

# In[51]:


statement_avg_by_category = """
SELECT product_category, AVG(sentiment) AS avg_sentiment
FROM {} 
GROUP BY product_category 
ORDER BY avg_sentiment DESC
""".format(table_name)

print(statement_avg_by_category)


# Query data in Amazon Athena database passing the prepared SQL statement:

# In[52]:


get_ipython().run_cell_magic('time', '', 'df_avg_by_category = wr.athena.read_sql_query(\n    sql=statement_avg_by_category,\n    database=database_name\n)')


# Preview the query results in the temporary S3 bucket:  `s3://aws-athena-query-results-ACCOUNT-REGION/`
# 
# **Instructions**: 
# - open the link
# - check the name of the S3 bucket
# - briefly check the content of it

# In[53]:


from IPython.core.display import display, HTML

display(HTML('<b>Review <a target="top" href="https://s3.console.aws.amazon.com/s3/buckets/aws-athena-query-results-{}-{}?region={}">Amazon S3 buckets</a></b>'.format(account_id, region, region)))


# Preview the results of the query:

# In[54]:


df_avg_by_category


# **Visualization**

# In[55]:


def show_values_barplot(axs, space):
    def _show_on_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() + float(space)
            _y = p.get_y() + p.get_height()
            value = round(float(p.get_width()),2)
            ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_plot(ax)
    else:
        _show_on_plot(axs)


# In[56]:


# Create plot
barplot = sns.barplot(
    data = df_avg_by_category, 
    y='product_category',
    x='avg_sentiment', 
    color="b", 
    saturation=1
)

# Set the size of the figure
sns.set(rc={'figure.figsize':(15.0, 10.0)})
    
# Set title and x-axis ticks 
plt.title('Average sentiment by product category')
#plt.xticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])

# Helper code to show actual values afters bars 
show_values_barplot(barplot, 0.1)

plt.xlabel("Average sentiment")
plt.ylabel("Product category")

plt.tight_layout()
# Do not change the figure name - it is used for grading purposes!
plt.savefig('avg_sentiment_per_category.png', dpi=300)

# Show graphic
plt.show(barplot)


# In[57]:


# Upload image to S3 bucket
sess.upload_data(path='avg_sentiment_per_category.png', bucket=bucket, key_prefix="images")


# Review the bucket on the account.
# 
# **Instructions**: 
# - open the link
# - click on the S3 bucket name `sagemaker-us-east-1-ACCOUNT`
# - open the images folder
# - check the existence of the image `avg_sentiment_per_category.png`
# - if you click on the image name, you can see the information about the image file. You can also download the file with the command on the top right Object Actions -> Download / Download as
# <img src="images/download_image_file.png" width="100%">

# In[58]:


from IPython.core.display import display, HTML

display(HTML('<b>Review <a target="top" href="https://s3.console.aws.amazon.com/s3/home?region={}">Amazon S3 buckets</a></b>'.format(region)))


# <a name='c1w1-3.4.'></a>
# ### 3.4. Which product categories have the most reviews?
# 
# Set the SQL statement to find the count of sentiment per product category, showing the results in the descending order:

# In[59]:


statement_count_by_category_desc = """
SELECT product_category, COUNT(*) AS count_reviews 
FROM {}
GROUP BY product_category 
ORDER BY count_reviews DESC
""".format(table_name)

print(statement_count_by_category_desc)


# Query data in Amazon Athena database passing the prepared SQL statement:

# In[60]:


get_ipython().run_cell_magic('time', '', 'df_count_by_category_desc = wr.athena.read_sql_query(\n    sql=statement_count_by_category_desc,\n    database=database_name\n)')


# Store maximum number of sentiment for the visualization plot:

# In[61]:


max_sentiment = df_count_by_category_desc['count_reviews'].max()
print('Highest number of reviews (in a single category): {}'.format(max_sentiment))


# **Visualization**

# <a name='c1w1-ex-4'></a>
# ### Exercise 4
# 
# Use `barplot` function to plot number of reviews per product category.
# 
# **Instructions**: Use the `barplot` chart example in the previous section, passing the newly defined dataframe `df_count_by_category_desc` with the count of reviews. Here, please put the `product_category` column into the `y` argument.

# In[64]:


# Create seaborn barplot
barplot = sns.barplot(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    data=df_avg_by_category, # Replace None
    y='product_category', # Replace None
    x='avg_sentiment', # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    color="b",
    saturation=1
)

# Set the size of the figure
sns.set(rc={'figure.figsize':(15.0, 10.0)})
    
# Set title
plt.title("Number of reviews per product category")
plt.xlabel("Number of reviews")
plt.ylabel("Product category")

plt.tight_layout()

# Do not change the figure name - it is used for grading purposes!
plt.savefig('num_reviews_per_category.png', dpi=300)

# Show the barplot
plt.show(barplot)


# In[65]:


# Upload image to S3 bucket
sess.upload_data(path='num_reviews_per_category.png', bucket=bucket, key_prefix="images")


# <a name='c1w1-3.5.'></a>
# ### 3.5. What is the breakdown of sentiments per product category?

# Set the SQL statement to find the count of sentiment per product category and sentiment:

# In[66]:


statement_count_by_category_and_sentiment = """
SELECT product_category,
         sentiment,
         COUNT(*) AS count_reviews
FROM {}
GROUP BY  product_category, sentiment
ORDER BY  product_category ASC, sentiment DESC, count_reviews
""".format(table_name)

print(statement_count_by_category_and_sentiment)


# Query data in Amazon Athena database passing the prepared SQL statement:

# In[67]:


get_ipython().run_cell_magic('time', '', 'df_count_by_category_and_sentiment = wr.athena.read_sql_query(\n    sql=statement_count_by_category_and_sentiment,\n    database=database_name\n)')


# Prepare for stacked percentage horizontal bar plot showing proportion of sentiments per product category.

# In[68]:


# Create grouped dataframes by category and by sentiment
grouped_category = df_count_by_category_and_sentiment.groupby('product_category')
grouped_star = df_count_by_category_and_sentiment.groupby('sentiment')

# Create sum of sentiments per star sentiment
df_sum = df_count_by_category_and_sentiment.groupby(['sentiment']).sum()

# Calculate total number of sentiments
total = df_sum['count_reviews'].sum()
print('Total number of reviews: {}'.format(total))


# Create dictionary of product categories and array of star rating distribution per category.

# In[69]:


distribution = {}
count_reviews_per_star = []
i=0

for category, sentiments in grouped_category:
    count_reviews_per_star = []
    for star in sentiments['sentiment']:
        count_reviews_per_star.append(sentiments.at[i, 'count_reviews'])
        i=i+1;
    distribution[category] = count_reviews_per_star


# Build array per star across all categories.

# In[70]:


distribution


# In[71]:


df_distribution_pct = pd.DataFrame(distribution).transpose().apply(
    lambda num_sentiments: num_sentiments/sum(num_sentiments)*100, axis=1
)
df_distribution_pct.columns=['1', '0', '-1']
df_distribution_pct


# **Visualization**
# 
# Plot the distributions of sentiments per product category.

# In[72]:


categories = df_distribution_pct.index

# Plot bars
plt.figure(figsize=(10,5))

df_distribution_pct.plot(kind="barh", 
                         stacked=True, 
                         edgecolor='white',
                         width=1.0,
                         color=['green', 
                                'orange', 
                                'blue'])

plt.title("Distribution of reviews per sentiment per category", 
          fontsize='16')

plt.legend(bbox_to_anchor=(1.04,1), 
           loc="upper left",
           labels=['Positive', 
                   'Neutral', 
                   'Negative'])

plt.xlabel("% Breakdown of sentiments", fontsize='14')
plt.gca().invert_yaxis()
plt.tight_layout()

# Do not change the figure name - it is used for grading purposes!
plt.savefig('distribution_sentiment_per_category.png', dpi=300)
plt.show()


# In[73]:


# Upload image to S3 bucket
sess.upload_data(path='distribution_sentiment_per_category.png', bucket=bucket, key_prefix="images")


# <a name='c1w1-3.6.'></a>
# ### 3.6. Analyze the distribution of review word counts

# Set the SQL statement to count the number of the words in each of the reviews:

# In[74]:


statement_num_words = """
    SELECT CARDINALITY(SPLIT(review_body, ' ')) as num_words
    FROM {}
""".format(table_name)

print(statement_num_words)


# Query data in Amazon Athena database passing the SQL statement:

# In[75]:


get_ipython().run_cell_magic('time', '', 'df_num_words = wr.athena.read_sql_query(\n    sql=statement_num_words,\n    database=database_name\n)')


# Print out and analyse some descriptive statistics: 

# In[76]:


summary = df_num_words["num_words"].describe(percentiles=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
summary


# Plot the distribution of the words number per review:

# In[77]:


df_num_words["num_words"].plot.hist(xticks=[0, 16, 32, 64, 128, 256], bins=100, range=[0, 256]).axvline(
    x=summary["100%"], c="red"
)

plt.xlabel("Words number", fontsize='14')
plt.ylabel("Frequency", fontsize='14')
plt.savefig('distribution_num_words_per_review.png', dpi=300)
plt.show()


# In[78]:


# Upload image to S3 bucket
sess.upload_data(path='distribution_num_words_per_review.png', bucket=bucket, key_prefix="images")


# Upload the notebook into S3 bucket for grading purposes.
# 
# **Note**: you may need to click on "Save" button before the upload.

# In[79]:


get_ipython().system('aws s3 cp ./C1_W1_Assignment.ipynb s3://$bucket/C1_W1_Assignment_Learner.ipynb')


# Please go to the main lab window and click on `Submit` button (see the `Finish the lab` section of the instructions).

# In[ ]:




