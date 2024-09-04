import warnings

import pandas as pd
from causalnex.structure.notears import from_pandas

from faircausal.data._BuildCausalModel import generate_linear_models
from faircausal.utils.Dag import has_cycle, is_connected
from faircausal.utils.Data import transform_data
from faircausal.visualization.CausalGraph import show_graph


class CausalDataReader:

    def __init__(self, *args, **kwargs):
        self.__fit_flag = False
        self.__set_outcome_variable_flag = False

        self.data = None
        self.linear_models = None
        self.causal_dag = None
        self.s_model = None
        self.original_data = None
        self.outcome_variable = None

        self.__load_auto(*args, **kwargs)

    def __getitem__(self, item):
        return self.get_model()[item]

    def __setitem__(self, key, value):
        if key == 'data':
            self.data = value
        elif key == 'linear_models':
            self.linear_models = value
        elif key == 'causal_dag':
            self.causal_dag = value
        elif key == 'original_data':
            self.original_data = value
        elif key == 'outcome_variable':
            self.outcome_variable = value
        else:
            raise ValueError("Invalid key.")

    def __load_auto(self, data: pd.DataFrame):

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        self.data = data
        self.original_data = data

    def __check_graph_validity(self):

        causal_dag = {node: list(self.s_model.successors(node)) for node in self.s_model.nodes}

        if not has_cycle(causal_dag) and is_connected(causal_dag):
            return True
        elif has_cycle(causal_dag) and not is_connected(causal_dag):
            warnings.warn("The causal graph has a cycle. You need to remove the cycle before fitting the model.")
            warnings.warn(
                "The causal graph has disconnected nodes. You need to remove the disconnected nodes before fitting the model.")
            return False
        elif has_cycle(causal_dag) and is_connected(causal_dag):
            warnings.warn("The causal graph has a cycle. You need to remove the cycle before fitting the model.")
            return False
        else:
            warnings.warn(
                "The causal graph has disconnected nodes. You need to remove the disconnected nodes before fitting the model.")
            return False

    def build_causal_graph(self, max_iter: int = 100, w_threshold: float = 0.8):
        self.data = transform_data(self.data)
        self.s_model = from_pandas(self.data, max_iter=max_iter, w_threshold=w_threshold)
        self.causal_dag = {node: list(self.s_model.successors(node)) for node in self.s_model.nodes}

        self.__check_graph_validity()

    def fit_linear_models(self):

        if not self.__check_graph_validity():
            raise RuntimeError(
                "The causal graph is not valid. You need to remove the cycle or disconnected nodes before fitting the model.")

        self.linear_models = generate_linear_models(self.causal_dag, self.data)

        if self.__set_outcome_variable_flag:
            if self.outcome_variable not in self.linear_models:
                warnings.warn(f"Outcome variable {self.outcome_variable} not found in the causal linear models.")

    def set_outcome_variable(self, outcome_variable: str):

        if not isinstance(outcome_variable, str):
            raise TypeError("outcome_variable must be a string.")

        if outcome_variable not in self.data.columns:
            raise ValueError(f"Outcome variable {outcome_variable} not found in data.")

        if self.__fit_flag:
            if outcome_variable not in self.linear_models:
                warnings.warn(f"Outcome variable {outcome_variable} not found in the causal linear models.")

        self.outcome_variable = outcome_variable
        self.__set_outcome_variable_flag = True

    def show(self, title="Causal Graph", save_path=None, figsize=(10, 7),
             node_color='lightblue', edge_color='gray', edge_width=1, arrow_size=20):
        show_graph(self.causal_dag, title=title, save_path=save_path, figsize=figsize,
                   node_color=node_color, edge_color=edge_color, edge_width=edge_width, arrow_size=arrow_size)

    def get_model(self):
        return {
            'data': self.data,
            'linear_models': self.linear_models,
            'causal_dag': self.causal_dag,
            'original_data': self.original_data,
            'outcome_variable': self.outcome_variable,
        }
