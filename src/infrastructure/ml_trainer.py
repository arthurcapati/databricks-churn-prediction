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
from typing import List, Any, Dict
from hyperopt import fmin, tpe, Trials, STATUS_OK, SparkTrials

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

    def _fix_hyperopt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        O Hyperopt retorna inteiros como floats (ex: 5.0). 
        O Spark quebra se receber float onde espera int.
        Esta função corrige isso baseada em nomes comuns de parâmetros.
        """
        int_params = [
            'maxDepth', 'maxIter', 'numTrees', 'k', 'maxBins', 
            'numFolds', 'aggregationDepth'
        ]
        
        cleaned = {}
        for k, v in params.items():
            if k in int_params:
                cleaned[k] = int(v)
            else:
                cleaned[k] = v
        return cleaned

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
        
    def tune(self, 
            df: DataFrame, 
            estimator_cls: Any, 
            search_space: Dict, 
            feature_cols: List[str] = None, 
            max_evals: int = 10, 
            metric: str = "areaUnderROC",
            run_name: str = None):
        """
        Otimização Genérica usando Hyperopt.
        
        :param df: DataFrame completo.
        :param estimator_cls: A CLASSE do algoritmo (ex: GBTClassifier), NÃO a instância.
        :param search_space: Espaço de busca do Hyperopt.
        """
        
        class_name = estimator_cls.__name__
        if not run_name:
            run_name = f"Hyperopt_{class_name}"

        print(f"🚀 Iniciando Otimização para: {class_name}")
        
        # 1. Preparação de Dados (Feita UMA vez para performance)
        # Diferente do treino único, aqui cacheamos os vetores para não reprocessar 50x
        if feature_cols is None:
            # Fallback ou lógica para pegar colunas numéricas
            feature_cols = ["recency", "frequency", "monetary"] # Exemplo
            
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        
        # Transformamos antes do loop para ganhar velocidade
        df_vec = assembler.transform(df)
        
        # Split Treino/Validação (Fixo para comparar maçãs com maçãs)
        train_df, val_df = df_vec.randomSplit([0.8, 0.2], seed=42)
                
        # Avaliador (Focado em AUC)
        evaluator = BinaryClassificationEvaluator(
            labelCol="churn", rawPredictionCol="rawPrediction", metricName=metric
        )

        # --- FUNÇÃO OBJETIVO ---
        def objective_function(params):
            # Corrige tipos (float -> int)
            params = self._fix_hyperopt_params(params)
            
            # Cria Run Aninhada (Child Run)
            with mlflow.start_run(nested=True):
                # 1. Instanciar o modelo com os parâmetros da vez
                # Aqui está a mágica: passamos **params para o construtor da classe
                model_instance = estimator_cls(labelCol="churn", featuresCol="features", **params)
                
                # Log Params
                mlflow.log_params(params)
                mlflow.set_tag("stage", "tuning_trial")
                
                # 2. Treino e Validação
                start_time = time.time()
                try:
                    model = model_instance.fit(train_df)
                    predictions = model.transform(val_df)
                    loss = evaluator.evaluate(predictions)
                except Exception as e:
                    print(f"❌ Erro com params {params}: {str(e)}")
                    return {'loss': 999, 'status': STATUS_OK} # Penalidade alta em caso de erro
                
                duration = time.time() - start_time
                
                # Log Métricas Básicas da Trial
                mlflow.log_metric("auc_val", loss)
                mlflow.log_metric("trial_duration", duration)
                
                # O Hyperopt quer MINIMIZAR, então retornamos -AUC
                return {'loss': -loss, 'status': STATUS_OK, 'params': params}

        # --- EXECUÇÃO DO HYPEROPT ---
        trials = Trials()
        
        # Configura Experimento Pai
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("type", "hyperparameter_tuning")
            mlflow.set_tag("algorithm", class_name)
            
            best_result = fmin(
                fn=objective_function,
                space=search_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials
            )
            
            # Recupera os melhores parâmetros reais (corrigidos)
            # O fmin retorna indices para alguns tipos, então é melhor pegar do objeto trials
            best_trial = sorted(trials.results, key=lambda x: x['loss'])[0]
            best_params = best_trial['params']

            
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_auc_val", -best_trial['loss'])
            
            # Criação do modelo
            estimator = estimator_cls(labelCol="churn", featuresCol="features", **best_params)
            pipeline = Pipeline(stages=[assembler, estimator])
            model = pipeline.fit(df)
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
            
            return best_params