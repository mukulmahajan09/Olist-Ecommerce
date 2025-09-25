# Databricks notebook source
# MAGIC %run "./Ecommerce Dataset Schema and Dataframes"

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.window import Window

# COMMAND ----------

orders_df = spark.read.parquet("/Volumes/olist/bronze/Orders")

# COMMAND ----------

order_items_df = spark.read.parquet("/Volumes/olist/bronze/Order_Items")

# COMMAND ----------

orders_df = orders_df.repartition(16, "order_id")#.cache()
order_items_df = order_items_df.repartition(16, "item_order_id")#.cache()

# COMMAND ----------

#orders_df.count()
#order_items_df.count()

# COMMAND ----------

products_df = spark.read.parquet('/Volumes/olist/bronze/Products')

# COMMAND ----------

order_payments_df = spark.read.parquet('/Volumes/olist/bronze/Order_Payments')

# COMMAND ----------

order_payments_df = order_payments_df.groupBy("payment_order_id") \
.agg(
    max("payment_sequential").alias("max_payment_sequence"),
    collect_set("payment_type").alias("payment_type"),
    sum("payment_installments").alias("payment_installments"),
    count("*").alias("payment_split_count"),  # number of rows per order
    sum("payment_value").alias("payment_value"),
    sum("payment_value_imputed").alias("payment_value_imputed"),
)

# COMMAND ----------

#order_payments_df.cache()
#order_payments_df.count()

# COMMAND ----------

order_reviews_df = spark.read.parquet('/Volumes/olist/bronze/Order_Reviews')

# COMMAND ----------

order_reviews_df = order_reviews_df.groupBy("review_order_id") \
.agg(
    collect_list("review_score").alias("review_scores"),
    collect_list("review_comment_title").alias("review_titles"),
    collect_list("review_comment_message").alias("review_message"),
    first("review_creation_date").alias("review_creation_date"),
    first("review_answer_timestamp").alias("review_answer_timestamp")
)

# COMMAND ----------

#order_reviews_df.cache()
#order_reviews_df.count()

# COMMAND ----------

geolocation_df = geolocation_df.dropDuplicates(["geolocation_lat", "geolocation_lng"])

# COMMAND ----------

orders_items = orders_df.join(order_items_df, on=orders_df.order_id == order_items_df.item_order_id, how='inner')
#orders_items.cache()

# COMMAND ----------

from pyspark.sql.functions import broadcast

orders_items_products = orders_items.join(broadcast(products_df), on=orders_items.item_product_id == products_df.product_id, how='left')

# COMMAND ----------

orders_items_products_sellers = orders_items_products.join(broadcast(sellers_df), on=orders_items_products.item_seller_id == sellers_df.seller_id, how='left')

# COMMAND ----------

orders_items_products_sellers_payments = orders_items_products_sellers.join(order_payments_df, on=orders_items_products_sellers.order_id == order_payments_df.payment_order_id, how='left')

# COMMAND ----------

orders_items_products_sellers_payments_customers = orders_items_products_sellers_payments.join(broadcast(customers_df), on=orders_items_products_sellers_payments.order_customer_id == customers_df.customer_id, how='left')

# COMMAND ----------

orders_items_products_sellers_payments_customers_reviews = orders_items_products_sellers_payments_customers.join(order_reviews_df, on=orders_items_products_sellers_payments_customers.order_id == order_reviews_df.review_order_id, how='left')

# COMMAND ----------

orders_items_products_sellers_payments_customers_reviews_geolocation = orders_items_products_sellers_payments_customers_reviews.join(broadcast(geolocation_df), on=orders_items_products_sellers_payments_customers_reviews.customer_zip_code_prefix == geolocation_df.geolocation_zip_code_prefix, how='left')

# COMMAND ----------

final_df = orders_items_products_sellers_payments_customers_reviews_geolocation.join(broadcast(products_name_translation_df), on='product_category_name', how='left')

# COMMAND ----------

# MAGIC %md
# MAGIC ### **ADD EXTRA COLUMNS TO FINAL DF**

# COMMAND ----------

# Total Revenue & Avg order value (AOV) Per Customer

customer_spending = final_df.groupBy('order_customer_id') \
    .agg(
        count('order_id').alias('total_orders'),
            round(sum('price'),2).alias('total_spent'),
                round(avg('price'), 2).alias('AOV')
    ) \
        .orderBy(desc('total_spent'))

# COMMAND ----------

quantiles = customer_spending.approxQuantile('AOV', [0.25, 0.75], 0.0)
low_threshold, high_threshold = quantiles[0], quantiles[1]

customer_segmented = customer_spending.withColumn('aov_segment',
    when(col('AOV') >= high_threshold, 'High')
    .when(col('AOV') < low_threshold, 'Low')
    .otherwise('Medium')
)

# COMMAND ----------

customer_segmented = customer_segmented.withColumnRenamed("order_customer_id", "order_customer_id_customer_segmented")

# COMMAND ----------

# AOV Customer Segment
final_df = final_df.join(broadcast(customer_segmented), on=final_df.order_customer_id == customer_segmented.order_customer_id_customer_segmented, how='left')

# COMMAND ----------

def add_feature_columns(final_df):

    # Delivery Status
    final_df = final_df.withColumn('is_delivered', 
                                   when(col('order_status') == 'delivered', lit(1)).otherwise(lit(0))) \
                        .withColumn('is_canceled', 
                                    when(col('order_status') == 'canceled', lit(1)).otherwise(lit(0)))

    # Total Revenue
    final_df = final_df.withColumn('order_revenue', col('price') + col('freight_value'))

    # Missing Payment
    final_df = final_df.withColumn('is_payment_missing', when(col('payment_value').isNull(), 1).otherwise(0))

    # Order Review
    final_df = final_df.withColumn('has_review', \
        when(
        (array_contains(col('review_titles'), 'No Review')) & (array_contains(col('review_message'), 'No Review')), 'No Reivew')
        .otherwise('Has Comment'))

    # Review Time Delays
    final_df = final_df.withColumn('time_from_delivery_to_review', \
        datediff(col('review_creation_date'), col('order_delivered_customer_date'))) \
        .withColumn('time_to_response', \
            datediff(col('review_answer_timestamp'), col('review_creation_date')))

    # Hourly Order Distribution
    final_df = final_df.withColumn('hour_of_day', expr('hour(order_purchase_timestamp)'))

    # WeekDay VS WeekEnd Orders
    final_df = final_df.withColumn('order_day_type', \
        when(dayofweek('order_purchase_timestamp').isin([1,7]),lit('Weekend')).otherwise(lit('Weekday')))

    # Calculate Delivery Time & Time Delays
    final_df = final_df.withColumn('actual_delivery_time', \
        when(col('order_delivered_customer_date').isNotNull(), \
        datediff('order_delivered_customer_date', 'order_purchase_timestamp'))
        .otherwise(None)
    ) \
    .withColumn('estimated_delivery_time', \
        when(col('order_estimated_delivery_date').isNotNull(),
        datediff('order_estimated_delivery_date', 'order_purchase_timestamp'))
        .otherwise(None)
    ) \
    .withColumn('delivery_delay_time', \
        when(col('actual_delivery_time').isNotNull() & col('estimated_delivery_time').isNotNull(),
        col('actual_delivery_time') - col('estimated_delivery_time'))
        .otherwise(None)
    ) \
    .withColumn('is_late', \
        when((col('delivery_delay_time').isNotNull()) & (col('delivery_delay_time') > 0), 1)
        .otherwise(0)
    )


    return final_df

# COMMAND ----------

final_df = add_feature_columns(final_df)

# COMMAND ----------

# Partition based on the order_year_month
final_df = final_df.withColumn("order_year_month", date_format("order_purchase_timestamp", "yyyy-MM"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### **CACHE THE FINAL DATAFRAME**

# COMMAND ----------

#final_df.cache()

# COMMAND ----------

#final_df.count()

# COMMAND ----------

#from pyspark import StorageLevel

#final_df = final_df.repartition(16, 'order_id').persist(StorageLevel.MEMORY_AND_DISK)
#final_df.count()  # or better: display(final_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **SAVE FINAL DF DATA TO DELTA FILE FORMAT**

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS olist.silver.final_df

# COMMAND ----------

final_df.write.format('delta') \
    .mode('overwrite') \
    .option('mergeSchema', 'true') \
    .partitionBy('order_year_month') \
    .save('/Volumes/olist/silver/final_df')

# COMMAND ----------

# MAGIC %md
# MAGIC ### **CREATE TABLE ON FINAL DATAFRAME**

# COMMAND ----------

final_df.write \
    .format('delta') \
    .mode('overwrite') \
    .option('mergeSchema', 'true') \
    .partitionBy('order_year_month') \
    .saveAsTable('olist.gold.final_df')

# COMMAND ----------

# MAGIC %md
# MAGIC ### **CREATE FACTS AND DIMENSION**

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM olist.gold.final_df;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE olist.gold.final_df;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE olist.gold.final_df;

# COMMAND ----------

final_df.printSchema()

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE AS dim_orders
# MAGIC olist.silver.final_df
# MAGIC AS
# MAGIC SELECT  FROM olist.gold.final_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### **TRANSFORMATION**

# COMMAND ----------

final_df = spark.read.format("delta").load("/Volumes/olist/silver/final_df")

# COMMAND ----------

# Finding Count Of Customer By State & City

display(customers_df.groupBy('customer_state','customer_city').count().orderBy('count', ascending=False))

# COMMAND ----------

# Order Status

display(orders_df.groupBy('order_status').count().orderBy('count', ascending=False))

# COMMAND ----------

# Payment Type

display(order_payments_df.groupBy('payment_type').count().orderBy('count', ascending=False))

# COMMAND ----------

# Top Selling Products

from pyspark.sql.functions import sum, round

display(
    order_items_df.groupBy('product_id') \
    .agg(round(sum('price'), 2).alias('total_sales')) \
    .orderBy('total_sales', ascending=False)
    )

# COMMAND ----------

delivery_df.write.mode('overwrite').parquet('/Volumes/olist/bronze/Delivery')

# COMMAND ----------

 order_items_df.select('price').summary().show()

# COMMAND ----------

# Total Revenue per Seller

'''order_items_grouped_df = order_items_df.groupBy('seller_id') \
                .agg(sum('price').alias('total_revenue_per_seller')) \
                .orderBy('total_revenue_per_seller', ascending=False)

total_revenue = order_items_grouped_df.join(sellers_df, on='seller_id', how='left')'''

t_revenue_per_seller = final_df.groupBy('seller_id') \
                    .agg(round(sum('price'),2).alias('total_revenue_per_seller')) \
                    .orderBy('total_revenue_per_seller', ascending=False)

display(t_revenue_per_seller)

# COMMAND ----------

display(t_revenue_per_seller.select('seller_id').distinct())

# COMMAND ----------

# AVG Review Per Seller

avg_rev_per_seller = final_df.groupBy('seller_id') \
                .agg(avg('review_score').alias('avg_review_per_seller')) \
                .orderBy('avg_review_per_seller', ascending=False)

display(avg_rev_per_seller)

# COMMAND ----------

# Most Sold Products (Top 10)

top_10_product = final_df.filter(final_df.order_status == 'delivered') \
                .groupBy('product_id', 'product_category_name_english') \
                .agg(round(sum('price'),2).alias('top_10_most_sold_product')) \
                .orderBy('top_10_most_sold_product', ascending=False) \
                .limit(10)

display(top_10_product)

# COMMAND ----------

# Total Order Per Customer

t_orders_per_cus = final_df.filter(final_df.order_status == 'delivered') \
                .groupBy('customer_id') \
                .agg(count('order_id').alias('total_order_per_customer')) \
                .orderBy(desc('total_order_per_customer')) \
                .limit(10)

display(t_orders_per_cus)

# COMMAND ----------

# Top Customer Per Spending

top_10_product = final_df.filter(final_df.order_status == 'delivered') \
                .groupBy('customer_id') \
                .agg(round(sum('price'),2).alias('top_customer_per_spending')) \
                .orderBy(desc('top_customer_per_spending')) \
                .limit(10)

display(top_10_product)

# COMMAND ----------

# Rank Per Seller Based On Revenue

window_spec = Window.partitionBy('seller_id').orderBy(desc('price'))

rnk_per_seller = final_df.withColumn('rank', rank().over(window_spec)).filter(col('rank')<=5)
den_per_seller = final_df.withColumn('dense_rank', dense_rank().over(window_spec)).filter(col('dense_rank')<=5)

# COMMAND ----------

display(den_per_seller.select('seller_id', 'price', 'dense_rank'))

# COMMAND ----------

# Seller Performance Metrics (Revenue, Avg Reveiw, Order Count)

seller_per_mt = final_df.groupBy('seller_id') \
  .agg(
    count('order_id').alias('total_orders'),
    round(sum('price'), 2).alias('total_revenue'),
    round(avg('review_score'), 2).alias('avg_review_score'),
    round(stddev('price'), 2).alias('price_variability')
  ) \
  .orderBy(desc('total_revenue'))

# COMMAND ----------

display(seller_per_mt)

# COMMAND ----------

# Product Popularity Metrics

product_per_mt = final_df.groupBy('product_id', 'product_category_name_english') \
  .agg(
    count('order_id').alias('total_sales'),
    round(sum('price'), 2).alias('total_revenue'),
    round(avg('price'), 2).alias('avg_price'),
    round(stddev('price'), 2).alias('price_volatility'),
    collect_set('seller_id').alias('unique_seller'),
    round(avg('review_score'), 2).alias('avg_review_score'),
    count('review_id').alias('total_reviewes')
  ) \
  .orderBy(desc('total_sales'))

# COMMAND ----------

display(product_per_mt)

# COMMAND ----------

# Customer Retention Ananlysis

cust_ren = final_df.groupBy('customer_id') \
  .agg(
    first('order_purchase_timestamp').alias('first_order_date'),
    last('order_purchase_timestamp').alias('last_order_date'),
    count('order_id').alias('total_orders'),
    round(sum('price'), 2).alias('total_revenue'),
    round(avg('price'), 2).alias('avg_price')
  ) \
  .orderBy(desc('total_orders'))

# COMMAND ----------

display(cust_ren) 

# COMMAND ----------

# Order Volume By Customer State

total_orders_by_state = final_df.filter(final_df.order_status == 'delivered') \
    .groupBy('geolocation_state') \
    .agg(count('order_id').alias('order_volume'),
         round(sum('price'), 2).alias('total_spending')) \
    .orderBy(desc('order_volume'))

# COMMAND ----------

display(total_orders_by_state)

# COMMAND ----------

print("Partitions:", orders_df.rdd.getNumPartitions())

# COMMAND ----------

top_10_product.explain(True)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM delta.`/FileStore/Transformation/Final_DF` order by order_year_month desc;

# COMMAND ----------

spark.sql('''

CREATE TABLE IF NOT EXISTS prod_olist.ecommerce_olist
USING DELTA
LOCATION '/Volumes/olist/silver/Final_DF'

''')

# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS prod_olist")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from prod_olist.ecommerce_olist;

# COMMAND ----------

# MAGIC %sql
# MAGIC select order_id, count(*) as count from ecommerce_olist group by order_id having count(*) > 1;

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

from pyspark.sql import DataFrame

df_names = [var for var, val in globals().items() if isinstance(val, DataFrame)]
print(f' The dataframe list: {df_names}')

# COMMAND ----------

# Example
df_registry = {}

df_registry['customers'] = customers_df
df_registry['orders'] = orders_df
df_registry['products'] = products_df

# To list all DataFrame names:
print(df_registry.keys())  # dict_keys(['customers', 'orders', 'products'])

# To access a specific DataFrame:
df_registry['customers'].show()

# COMMAND ----------

geolocation_df.storageLevel
#geolocation_df.explain(True)
