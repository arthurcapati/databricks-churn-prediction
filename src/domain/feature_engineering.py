featuresfrom pyspark.sql import DataFrame
from pyspark.sql import functions as F
from src.utility.environment import Environment 

class RFMTransformer:
    """
    Domain Service: Transforma pedidos em indicadores de comportamento (RFM).
    """
    def __init__(self, master_df: DataFrame):
        self.master_df = master_df

    def transform(self, snapshot_date: str) -> DataFrame:
        print(f"--- Processando Safra: {snapshot_date} ---")
        
        # ==========================================
        # 1. JANELA DE OBSERVAÇÃO (O PASSADO - X)
        # ==========================================
        # Pegamos apenas dados que aconteceram ANTES ou NA data de corte
        past_master = self.master_df.filter(F.col("order_purchase_timestamp") <= snapshot_date)
               
        # Cálculo das Features RFM (Data Engineering)
        features_df = past_master.groupBy("customer_unique_id").agg(
            F.datediff(F.lit(snapshot_date), F.max("order_purchase_timestamp")).alias("recency"),
            F.countDistinct("order_id").alias("frequency"),
            F.sum("total_order_payment").alias("monetary"),
            F.first("customer_zip_code_prefix", ignorenulls=True).alias("zip_code"),
            F.first("customer_city", ignorenulls=True).alias("city"),
            F.first("customer_state", ignorenulls=True).alias("state"),
            F.sum("total_items").alias("total_items_volume"),
            F.max("payment_method_count").alias("payment_method_count"),
            F.first("primary_payment_type", ignorenulls=True).alias("primary_payment_type"),
            F.max("max_installments").alias("max_installments"),
            F.avg("Avg_review_score").alias("avg_satisfaction"),
            F.min("Min_review_score").alias("min_review_score"),
            F.sum("Total_Review_Number").alias("total_reviews_given")
        )
        
        # ==========================================
        # 2. JANELA DE RESPOSTA (O FUTURO - y)
        # ==========================================
        # Definimos até quando vamos olhar para dizer se ele voltou ou não
        churn_window_days = Environment.churn_window_days # Ex: 90 dias
        future_limit = F.date_add(F.lit(snapshot_date), churn_window_days)
        
        # Filtramos quem comprou DEPOIS da data de corte
        future_master = self.master_df.filter(
            (F.col("order_purchase_timestamp") > snapshot_date) & 
            (F.col("order_purchase_timestamp") <= future_limit)
        )

        # Lista de clientes que RETORNARAM (Não são Churn)
        active_customers = future_master.select("customer_unique_id").distinct() \
                                        .withColumn("fez_compra_futura", F.lit(1))

        # ==========================================
        # 3. JUNÇÃO FINAL E CRIAÇÃO DO LABEL
        # ==========================================
        final_df = features_df.join(active_customers, "customer_unique_id", "left")
        
        # Lógica do Target:
        # Se "fez_compra_futura" for NULL, significa que ele NÃO apareceu no futuro -> CHURN (1)
        # Se "fez_compra_futura" for 1, significa que ele comprou -> NÃO CHURN (0)
        final_df = final_df.withColumn(
            "churn", 
            F.when(F.col("fez_compra_futura").isNull(), 1).otherwise(0)
        ).drop("fez_compra_futura")
        
        return final_df.na.fill(0) # Preenche nulos (ex: monetary) com 0 se necessário