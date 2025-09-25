# Databricks notebook source
# MAGIC %run "./Ecommerce Dataset Schema and Dataframes"

# COMMAND ----------

# MAGIC %md
# MAGIC ### **ORDER REVIEWS DF**

# COMMAND ----------

from pyspark.sql.functions import col, when, count

# COMMAND ----------

order_reviews_cleaned_df = order_reviews_df

# COMMAND ----------

null_count = order_reviews_cleaned_df.select(
    count(when(col('review_score').isNull(), True)).alias('null_count')
).collect()[0]['null_count']

display(null_count)

# COMMAND ----------

display(order_reviews_cleaned_df)

# COMMAND ----------

order_reviews_cleaned_df = order_reviews_cleaned_df.fillna({'review_score': -1})

# COMMAND ----------

null_count = order_reviews_cleaned_df.select(
    count(
       when(
            col('review_comment_title').isNull() & col('review_comment_message').isNull(), True)
       ).alias('null_count')
    ).collect()[0]['null_count']

display(null_count)

# COMMAND ----------

order_reviews_cleaned_df = order_reviews_cleaned_df.withColumn(
    'both_null',
    when(
        col('review_comment_title').isNull() & col('review_comment_message').isNull(), 'No Comment'
).otherwise('Has Comment')
)

# COMMAND ----------

order_reviews_cleaned_df = order_reviews_cleaned_df.withColumn(
    'review_comment_title',
       when(
            col('both_null') == 'No Comment', 'No Review'
       ).otherwise(col('review_comment_title'))
    ).withColumn(
        'review_comment_message',
       when(
            col('both_null') == 'No Comment', 'No Review'
       ).otherwise(col('review_comment_message'))
    )

# COMMAND ----------

order_reviews_cleaned_df = order_reviews_cleaned_df.fillna({'review_comment_title': 'Review Given', 'review_comment_message': 'Review Given'})

# COMMAND ----------

order_reviews_cleaned_df = order_reviews_cleaned_df.fillna({'review_creation_date': '1900-01-01', 'review_answer_timestamp': '1900-01-01'})

# COMMAND ----------

display(order_reviews_cleaned_df.filter(col("order_id").isNull()))

# COMMAND ----------

display(order_reviews_cleaned_df.filter(col("review_id").isNull()))

# COMMAND ----------

order_reviews_cleaned_df = order_reviews_cleaned_df.dropna(subset=["review_id"])

# COMMAND ----------

order_reviews_cleaned_df = order_reviews_cleaned_df.dropna(subset=["order_id"])

# COMMAND ----------

order_reviews_cleaned_df = order_reviews_cleaned_df.withColumnRenamed("order_id", "review_order_id") \
    .withColumnRenamed("both_null", "has_review")

# COMMAND ----------

invalid_rows_df = order_reviews_cleaned_df.filter(col("review_order_id").rlike(r"^\d{4}-\d{2}-\d{2}.*"))
display(invalid_rows_df)

# COMMAND ----------

order_reviews_cleaned_df = order_reviews_cleaned_df.filter(~col("review_order_id").rlike(r"^\d{4}-\d{2}-\d{2}.*"))

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS olist.bronze.order_reviews;

# COMMAND ----------

order_reviews_cleaned_df.write.mode("overwrite").parquet("/Volumes/olist/bronze/order_reviews")

# COMMAND ----------

# MAGIC %md
# MAGIC ### **ORDERS DF**

# COMMAND ----------

orders_cleaned_df = orders_df

# COMMAND ----------

## Future Values Will Come Nulls In This Columns Then, Currently Dont Have

orders_cleaned_df = orders_cleaned_df.na.drop(subset=['order_id','customer_id','order_status'])

# COMMAND ----------

orders_cleaned_df.filter(col("order_delivered_customer_date").isNull()) \
    .groupBy("order_status") \
        .count() \
            .show()

# COMMAND ----------

# Optionally filter them out if needed
display(orders_cleaned_df.filter(((col("order_status") == "invoiced") & col("order_delivered_customer_date").isNull())))

# COMMAND ----------

null_count = orders_cleaned_df.filter(
        (col('order_approved_at').isNull()) & (col('order_status') == 'canceled')).count()
display(null_count)

# COMMAND ----------

orders_cleaned_df = orders_cleaned_df.withColumn('order_approved_at',
            when(
                (col('order_status') == 'delivered') & (col('order_approved_at').isNull()),
                col('order_purchase_timestamp')
        ).otherwise(col('order_approved_at'))
)

# COMMAND ----------

orders_cleaned_df = orders_cleaned_df.fillna({'order_approved_at': '1900-01-01 00:00:00'})

# COMMAND ----------

orders_cleaned_df = orders_cleaned_df.withColumnRenamed("customer_id", "order_customer_id")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS olist.bronze.orders;

# COMMAND ----------

orders_cleaned_df.write.mode("overwrite").parquet("/Volumes/olist/bronze/orders")

# COMMAND ----------

# MAGIC %md
# MAGIC ### **PRODUCTS DF**

# COMMAND ----------

products_cleaned_df = products_df

# COMMAND ----------

products_cleaned_df = products_cleaned_df.fillna({'product_category_name': 'Unknown', 'product_name_lenght': 0, 'product_description_lenght': 0, 'product_photos_qty': 0, 'product_weight_g': 0, 'product_length_cm': 0, 'product_height_cm': 0, 'product_width_cm': 0})

# COMMAND ----------

products_cleaned_df.groupBy('product_weight_g').count().orderBy('count', ascending=False).show()

# COMMAND ----------

quantiles = products_cleaned_df.approxQuantile('product_weight_g', [0.25, 0.5, 0.75], 0.0)
q1, q2, q3 = quantiles
print(f"Q1: {q1}, Median (Q2): {q2}, Q3: {q3}")

# COMMAND ----------

products_cleaned_df = products_cleaned_df.withColumn(
    'product_size',
    when(col('product_weight_g') <= q1, 'Small')
    .when((col('product_weight_g') > q1) & (col('product_weight_g') <= q2), 'Medium')
    .when((col('product_weight_g') > q2) & (col('product_weight_g') <= q3), 'Large')
    .otherwise('Extra Large')
)

# COMMAND ----------

products_cleaned_df = products_cleaned_df.na.drop(subset=["product_category_name"]) 

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS olist.bronze.products;

# COMMAND ----------

products_cleaned_df.write.mode("overwrite").parquet("/Volumes/olist/bronze/products")

# COMMAND ----------

# MAGIC %md
# MAGIC ### **ORDER PAYMENTS DF**

# COMMAND ----------

order_payments_cleaned_df = order_payments_df

# COMMAND ----------

#Alternative To Imputer

# Calculate mean of the column
mean_value = order_payments_cleaned_df.selectExpr("avg(payment_value) as mean_val").first()["mean_val"]

# Fill nulls with the mean value and create a new column
from pyspark.sql.functions import when, col

order_payments_cleaned_df = order_payments_cleaned_df.withColumn(
    "payment_value_imputed",
    when(col("payment_value").isNull(), mean_value).otherwise(col("payment_value"))
)

#from pyspark.ml.feature import Imputer
# Create the Imputer
#imputer = Imputer().setInputCols(['payment_value']).setOutputCols(['payment_value_imputed']).setStrategy('mean')
# Apply the imputer to the entire DataFrame (fit and transform in one step)
#order_payments_cleaned_df = imputer.fit(order_payments_cleaned_df).transform(order_payments_cleaned_df)

# COMMAND ----------

order_payments_cleaned_df = order_payments_cleaned_df.withColumn('payment_type',
         when(col('payment_type')=='boleto','Bank Transfer')
        .when(col('payment_type')=='credit_card','Credit Card')
        .when(col('payment_type')=='debit_card','Debit Card')
        .when(col('payment_type')=='voucher', 'Voucher')
        .otherwise('other')
)

# COMMAND ----------

order_payments_cleaned_df = order_payments_cleaned_df.withColumnRenamed("order_id", "payment_order_id")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS olist.bronze.order_payments;

# COMMAND ----------

order_payments_cleaned_df.write.mode("overwrite").parquet("/Volumes/olist/bronze/order_payments")

# COMMAND ----------

# MAGIC %md
# MAGIC ### **ORDER ITEMS DF**

# COMMAND ----------

order_items_cleaned_df = order_items_df

# COMMAND ----------

order_items_cleaned_df = order_items_cleaned_df.withColumnRenamed("order_id", "item_order_id") \
    .withColumnRenamed("order_item_id", "item_order_item_id") \
        .withColumnRenamed("product_id", "item_product_id") \
            .withColumnRenamed("seller_id", "item_seller_id")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS olist.bronze.order_items;

# COMMAND ----------

order_items_cleaned_df.write.mode("overwrite").parquet("/Volumes/olist/bronze/order_items")
