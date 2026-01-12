import os
import dotenv
from pathlib import Path

class EnvironmentClass(object):
    """
    Central Configuration Hub.
    Gerencia variáveis de ambiente, caminhos do sistema e constantes de negócio.
    """

    def __init__(self):
        base_path = os.getcwd()
        if 'notebooks' in base_path:
            base_path = '/'.join(base_path.split('/')[:-1])
        self.__base_path = base_path
        self.__mode = os.getenv("MODE", "development")
        self._load_dotenv()

    def _load_dotenv(self):
        """Carrega o arquivo .env apropriado."""
        env_file = ".env.production" if self.__mode == "production" else ".env"
        path = Path(self.__base_path) / env_file
        
        if path.exists():
            dotenv.load_dotenv(dotenv_path=path)
        # Se não existir, assume que as variáveis já estão no OS (ex: Pipeline CI/CD)

    def _get_var(self, key: str, default: str = None, required: bool = False) -> str:
        """Helper para recuperar variáveis, lançando erro se required=True e não existir."""
        val = os.getenv(key, default)
        if val is None and required:
            raise KeyError(f"A variável de ambiente obrigatória '{key}' não foi definida.")
        return val

    # --- 1. Propriedades de Ambiente & Infra ---
    @property
    def mode(self) -> str:
        return self.__mode

    @property
    def kaggle_username(self) -> str:
        return self._get_var("KAGGLE_USERNAME", required=True)

    @property
    def kaggle_key(self) -> str:
        return self._get_var("KAGGLE_KEY", required=True)

    # --- 2. Caminhos do Data Lake (Medallion Architecture) ---
    @property
    def root_path(self) -> str:
        """
        Define a raiz do armazenamento. 
        Permite override via env var, útil para mudar local de testes vs produção.
        """
        default_root = "."
        return self._get_var("BASE_PATH_OVERRIDE", default_root)

    @property
    def feature_store_path(self) -> str:
        return self._get_var("FEATURES_CATALOG")

    # --- 3. Constantes de Negócio (Domain Configs) ---
    @property
    def churn_window_days(self) -> int:
        """
        Regra de negócio: Dias sem comprar para considerar Churn.
        Pode ser ajustada via variável de ambiente sem deploy de código.
        """
        return int(self._get_var("CHURN_WINDOW_DAYS", 90))

    @property
    def kaggle_dataset_name(self) -> str:
        return "olistbr/brazilian-ecommerce"
    
    @property
    def KAGGLE_CONFIG_DIR(self) -> str:
     return self._get_var("KAGGLE_CONFIG_DIR")

    @property
    def DATABRICKS_CATALOG(self) -> str:
     return self._get_var("DATABRICKS_CATALOG")

    @property
    def MLFLOW_DFS_TMP(self) -> str:
     return self._get_var("MLFLOW_DFS_TMP")


# Singleton Instance
Environment = EnvironmentClass()