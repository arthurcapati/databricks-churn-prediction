from pyspark.sql import SparkSession, DataFrame

class DataManager:
    """
    Responsável apenas pelo I/O de dados. 
    Não conhece regras de negócio.
    """
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def read_csv(self, path: str) -> DataFrame:
        return self.spark.read.csv(
            path,
            header=True,
            inferSchema=True,
            multiLine=True, # Essencial para o dataset de reviews
            escape='"',
            quote='"'
        )

    def read_delta(self, path: str) -> DataFrame:
        return self.spark.read.format("delta").load(path)

    def save_delta(self, df: DataFrame, path: str, mode: str = "overwrite"):
        df.write.format("delta").mode(mode).option("overwriteSchema", "true").save(path)
        print(f"-> Dados salvos em Delta: {path}")