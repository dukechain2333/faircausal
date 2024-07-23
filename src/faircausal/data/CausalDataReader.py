import pandas as pd
import numpy as np

from faircausal.data._BuildCausalModel import build_causal_model, generate_beta_params
from faircausal.utils.Dag import is_valid_causal_dag
from faircausal.utils.Data import classify_variables, check_if_dummy


class CausalDataReader:

    def __init__(self, *args, **kwargs):

        self.sm = None
        self.auto_load_flag = False
        self.data = None
        self.data_type = None
        self.beta_dict = None
        self.causal_dag = None
        self.original_data = None
        self.original_data_type = None
        self.outcome_variable = None

        if len(args) == 3:
            self.__load_manually(*args, **kwargs)
        elif len(args) == 1:
            self.__load_auto(*args, **kwargs)
        else:
            raise ValueError("Invalid number of arguments. Expected either 1 or 3 arguments.")

    def __getitem__(self, item):
        return self.get_model()[item]

    def __load_manually(self, data: pd.DataFrame, data_type: dict, beta_dict: dict, causal_dag: dict, **kwargs):

        # Check if the input data is in the correct type
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not isinstance(data_type, dict):
            raise TypeError("data_type must be a dictionary.")
        if not isinstance(beta_dict, dict):
            raise TypeError("beta_dict must be a dictionary.")
        if not isinstance(causal_dag, dict):
            raise TypeError("causal_dag must be a dictionary.")

        # Check if the data type is valid
        if not self.__check_if_valid_data_type(data_type):
            raise ValueError("Invalid data type, expected 'discrete' or 'continuous'.")
        are_nodes_in_df, missing_node = self.__check_dag_nodes_in_dataframe(data_type, data)
        if not are_nodes_in_df:
            raise ValueError(f"Node {missing_node} is not present in the data.")
        self.data_type = data_type
        self.original_data_type = data_type

        # Check if the causal DAG is a valid DAG
        if not is_valid_causal_dag(causal_dag):
            raise ValueError("The input graph must be a Directed Acyclic Graph (DAG).")
        self.causal_dag = causal_dag

        # Check if the nodes in the causal DAG are present in the data
        are_nodes_in_df, missing_node = self.__check_dag_nodes_in_dataframe(causal_dag, data)
        if not are_nodes_in_df:
            raise ValueError(f"Node {missing_node} is not present in the data.")

        # Check if the data is one-hot encoded
        if not check_if_dummy(data, data_type):
            raise ValueError("The input data must be one-hot encoded dummy variables.")
        self.data = data
        self.original_data = data

        # Check if the number of beta coefficients is correct
        beta_count = self.__count_linear_regression_parameters(causal_dag)
        if len(beta_dict) != beta_count:
            raise ValueError(f"Invalid number of beta coefficients, expected {beta_count}.")
        self.beta_dict = beta_dict

    def __load_auto(self, data: pd.DataFrame):

        self.auto_load_flag = True

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        self.data = data
        self.original_data_type = self.data_type
        self.original_data = data

    def fit_causal_model(self, outcome_variable: str):

        if not self.auto_load_flag:
            raise AttributeError("get_causal_model() is only available when the data is auto-loaded.")

        if not isinstance(outcome_variable, str):
            raise TypeError("outcome_variable must be a string.")

        self.data = self.original_data.copy()

        if outcome_variable not in self.data.columns:
            raise ValueError(f"Outcome variable {outcome_variable} not found in data.")

        self.outcome_variable = outcome_variable

        self.data_type = classify_variables(self.data)

        self.sm, self.causal_dag, self.data, self.data_type = build_causal_model(self.data, self.data_type, self.outcome_variable)
        self.beta_dict = generate_beta_params(self.causal_dag, self.data)

    @staticmethod
    def __count_linear_regression_parameters(dag: dict):
        total_params = 0
        for node, parents in dag.items():
            if parents:
                total_params += len(parents) + 1
        return total_params

    @staticmethod
    def __check_dag_nodes_in_dataframe(dag: dict, df: pd.DataFrame):
        df_columns = df.columns.tolist()

        for node in dag:
            if node not in df_columns:
                return False, node

        return True, None

    @staticmethod
    def __check_if_valid_data_type(data_type: dict):
        for node, dtype in data_type.items():
            if dtype not in ['discrete', 'continuous']:
                return False

    def get_discrete(self):
        return {node: dtype for node, dtype in self.data_type.items() if dtype == 'discrete'}

    def get_continuous(self):
        return {node: dtype for node, dtype in self.data_type.items() if dtype == 'continuous'}

    def get_model(self):
        return {
            'data': self.data,
            'data_type': self.data_type,
            'beta_dict': self.beta_dict,
            'causal_dag': self.causal_dag,
            'original_data': self.original_data,
            'original_data_type': self.original_data_type,
            'outcome_variable': self.outcome_variable
        }

if __name__ == '__main__':
    data = pd.DataFrame({
        'A': np.random.normal(0, 1, 200),
        'B': np.random.choice(['x', 'y', 'z'], 200),
        'C': np.random.normal(5, 2, 200),
        'D': np.random.choice([0, 1], 200),
        'E': np.random.choice([0, 1, 2], 200)
    })
    data_reader = CausalDataReader(data)
    from faircausal.optimizing.ObjectFunctions import loss, negative_log_likelihood

    data_reader.fit_causal_model('B')
    print(data_reader.get_model())
    print(negative_log_likelihood(data_reader))
    print(loss(data_reader))
    data_reader.fit_causal_model('A')
    print(data_reader.get_model())
    print(negative_log_likelihood(data_reader))
    print(loss(data_reader))
