import time
import psutil
import os
import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, Estimator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from src.utility.environment import Environment
from abc import ABC, abstractmethod
from typing import List

class ITrainer(ABC):
    def __init__(self, experiment_path: str):
        mlflow.set_experiment(experiment_path)
        self.experiment_path = experiment_path

    def _get_ram_usage(self) -> float:
        """Retorna o uso de memória do processo atual em MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    @abstractmethod
    def train(self, df: DataFrame, estimator: Estimator, run_name: str = None):
        pass

class PySparkTrainer(ITrainer):
    """
    Classe genérica para treinar qualquer classificador Spark MLlib.
    Calcula Accuracy, Precision, Recall, F1 e AUC-ROC.
    """

    def __init__(self, experiment_path: str):
        super().__init__(experiment_path)

    def train(self, df: DataFrame, estimator: Estimator, run_name: str = None, feature_cols: List[str] = None):
        """
        :param df: DataFrame com as features e coluna target 'churn'.
        :param estimator: Instância do algoritmo (ex: RandomForestClassifier, GBTClassifier).
        :param run_name: Nome para identificar a run no MLflow.
        """
        
        # Se não passar nome, usa o nome da classe do algoritmo (ex: RandomForestClassifier)
        if not run_name:
            run_name = estimator.__class__.__name__

        with mlflow.start_run(run_name=run_name) as run:
            print(f"🏁 Iniciando Run: {run_name} (ID: {run.info.run_id})")
            
            # 1. Preparação de Features
            # Nota: Em um setup avançado, o Assembler poderia vir de fora, 
            # mas para este projeto mantemos aqui por simplicidade.
            if feature_cols==None:
                feature_cols = ["recency", "frequency", "monetary"]
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            
            # 2. Split Treino/Teste
            train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
            
            # 3. Pipeline Genérico
            # O pipeline aceita qualquer estimator que tenha fit/transform
            pipeline = Pipeline(stages=[assembler, estimator])
            
            # 4. Treinamento
            print("⏳ Treinando modelo...")
            start_train = time.time()
            
            model = pipeline.fit(train_data)
            
            end_train = time.time()
            train_time_sec = end_train - start_train
            # -------------------------------------
            
            # --- [NOVO] MONITORAMENTO DE INFERÊNCIA (Benchmark) ---
            print("⚡ Benchmark de Inferência...")
            
            start_inference = time.time()
            ram_start = self._get_ram_usage()
            
            predictions = model.transform(test_data)
            total_rows = predictions.count() # AÇÃO QUE FORÇA O CÁLCULO
            
            ram_end = self._get_ram_usage()
            end_inference = time.time()
            
            # Cálculos de Performance
            inference_time_sec = end_inference - start_inference
            rows_per_second = total_rows / inference_time_sec if inference_time_sec > 0 else 0
            ram_delta_mb = ram_end - ram_start # Quanto de memória a inferência consumiu a mais
            
            print(f"   -> Tempo de Treino: {train_time_sec:.2f}s")
            print(f"   -> Tempo de Inferência ({total_rows} rows): {inference_time_sec:.4f}s")
            print(f"   -> Throughput: {rows_per_second:.0f} rows/sec")
            print(f"   -> Pico de RAM Estimado: {ram_delta_mb:.2f} MB")
            
            # ------------------------------------------------------
            
            # 6. Avaliação Multi-métrica
            print("📊 Calculando métricas...")
            
            # Binary Evaluator (Ideal para AUC em classificação binária)
            bin_eval = BinaryClassificationEvaluator(
                labelCol="churn", rawPredictionCol="rawPrediction"
            )
            auc_roc = bin_eval.setMetricName("areaUnderROC").evaluate(predictions)
            
            # Multiclass Evaluator (Para Acc, Prec, Recall, F1)
            # Usamos 'weighted' para ter uma média ponderada das classes
            multi_eval = MulticlassClassificationEvaluator(
                labelCol="churn", predictionCol="prediction"
            )
            
            accuracy = multi_eval.setMetricName("accuracy").evaluate(predictions)
            precision = multi_eval.setMetricName("weightedPrecision").evaluate(predictions)
            recall = multi_eval.setMetricName("weightedRecall").evaluate(predictions)
            f1_score = multi_eval.setMetricName("f1").evaluate(predictions)
            
            print(f"   -> Accuracy:  {accuracy:.4f}")
            print(f"   -> AUC-ROC:   {auc_roc:.4f}")
            print(f"   -> Precision: {precision:.4f}")
            print(f"   -> Recall:    {recall:.4f}")

            # --- MLFLOW LOGGING ---
            
            # Log Metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc_roc", auc_roc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1_score)

            # 2. [NOVO] Métricas de Performance (Sistema)
            mlflow.log_metric("train_time_sec", train_time_sec)
            mlflow.log_metric("inference_time_sec", inference_time_sec)
            mlflow.log_metric("throughput_rows_sec", rows_per_second)
            mlflow.log_metric("ram_usage_mb", ram_end) # Uso total do driver
            
            # Log Parameters (Extrai automaticamente do estimator)
            # Isso pega, por exemplo, 'numTrees', 'maxDepth', 'regParam'
            mlflow.log_params(estimator.extractParamMap())
            
            # Log Tags (Para facilitar busca depois)
            mlflow.set_tag("algorithm", estimator.__class__.__name__)
            
            # Log do Modelo
            input_sample = df.select(feature_cols).limit(5).toPandas() 
            prediction_sample = model.transform(df.limit(5)).toPandas()
            signature = infer_signature(input_sample, prediction_sample)

            mlflow.spark.log_model(
                spark_model=model,
                artifact_path="model", 
                signature=signature,   # <--- OBRIGATÓRIO PARA UNITY CATALOG
                input_example=input_sample # Opcional, mas boa prática
            )
            
            print(f"✅ Run finalizada com sucesso.")
            return run.info.run_id