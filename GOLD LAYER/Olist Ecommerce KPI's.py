# Databricks notebook source
# MAGIC %md
# MAGIC ### **KPI'S**

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
