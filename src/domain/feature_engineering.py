from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from src.utility.environment import conf  # Injeção da Configuração Unificada

class RFMTransformer:
    """
    Domain Service: Transforma pedidos em indicadores de comportamento (RFM).
    """
    def __init__(self, orders_df: DataFrame, payments_df: DataFrame):
        self.orders = orders_df
        self.payments = payments_df

    def transform(self) -> DataFrame:
        # 1. Join
        full_df = self.orders.join(self.payments, "order_id", "inner")
        
        # 2. Referência Temporal
        max_date = full_df.select(F.max("order_purchase_timestamp")).collect()[0][0]
        
        # 3. Cálculo RFM
        rfm_df = full_df.groupBy("customer_id").agg(
            F.datediff(F.lit(max_date), F.max("order_purchase_timestamp")).alias("recency"),
            F.countDistinct("order_id").alias("frequency"),
            F.sum("payment_value").alias("monetary")
        )
        
        # 4. Regra de Churn (Usa a config centralizada)
        print(f"Aplicando regra de Churn: > {conf.churn_window_days} dias sem compra.")
        
        final_df = rfm_df.withColumn(
            "churn", 
            F.when(F.col("recency") > conf.churn_window_days, 1).otherwise(0)
        )
        
        return final_df.na.fill(0)