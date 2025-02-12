import warnings

import pandas as pd

from faircausal.data._BuildCausalModel import generate_linear_models
from faircausal.utils.Dag import has_cycle, is_connected
from faircausal.visualization.CausalGraph import show_graph


class CausalDataReader:

    def __init__(self, *args, **kwargs):
        self.__fit_flag = False
        self.__set_outcome_variable_flag = False

        self.data = None
        self.predicted_data = None
        self.linear_models = None
        self.causal_dag = {}
        self.original_data = None
        self.outcome_variable = None
        self.mediator = None
        self.exposure = None

        self.__load_auto(*args, **kwargs)

    def __getitem__(self, item):
        return self.get_model()[item]

    def get_model(self):
        return {
            'data': self.data,
            'linear_models': self.linear_models,
            'causal_dag': self.causal_dag,
            'original_data': self.original_data,
            'outcome_variable': self.outcome_variable,
            'mediator': self.mediator,
            'exposure': self.exposure
        }

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

        has_cycles = has_cycle(self.causal_dag)
        is_fully_connected = is_connected(self.causal_dag)

        if not has_cycles and is_fully_connected:
            return True

        if has_cycles and is_fully_connected:
            warnings.warn("The causal graph has cycles. You need to remove the cycles before fitting the model.")
            return False

        if not has_cycles and not is_fully_connected:
            warnings.warn(
                "The causal graph has disconnected components. You need to ensure full connectivity before fitting the model.")
            return False

        if has_cycles and not is_fully_connected:
            warnings.warn(
                "The causal graph has cycles and disconnected components. You need to remove the cycles and ensure full connectivity before fitting the model.")
            return False

        return False

    def fit_linear_models(self):

        if not self.__check_graph_validity():
            raise RuntimeError(
                "The causal graph is not valid. You need to remove the cycle or disconnected nodes before fitting the model.")

        self.linear_models = generate_linear_models(self.causal_dag, self.data)

        if self.__set_outcome_variable_flag:
            if self.outcome_variable not in self.linear_models:
                warnings.warn(f"Outcome variable {self.outcome_variable} not found in the causal linear models.")

    def set_causal_variables(self, exposure: str, outcome_variable: str):
        """
        Automatically sets the exposure, mediators, and outcome variable.

        :param exposure: Name of the exposure variable.
        :param outcome_variable: Name of the outcome variable.
        """
        # Validate exposure
        if not isinstance(exposure, str):
            raise TypeError("Exposure must be a string.")
        if exposure not in self.data.columns:
            raise ValueError(f"Exposure variable {exposure} not found in data.")
        if self.__fit_flag and exposure not in self.linear_models:
            warnings.warn(f"Exposure {exposure} not found in the causal linear models.")

        # Validate outcome variable
        if not isinstance(outcome_variable, str):
            raise TypeError("Outcome variable must be a string.")
        if outcome_variable not in self.data.columns:
            raise ValueError(f"Outcome variable {outcome_variable} not found in data.")
        if self.__fit_flag and outcome_variable not in self.linear_models:
            warnings.warn(f"Outcome {outcome_variable} not found in the causal linear models.")

        # Store exposure and outcome
        self.exposure = exposure
        self.outcome_variable = outcome_variable

        # Find mediators from the causal DAG
        self.mediator = self._find_mediators()

    def _find_mediators(self):
        """
        Identify mediators between the exposure and outcome using the causal DAG.

        :return: Dictionary of mediators in {M1: None, M2: M3, M3: None} format.
        """

        mediators = {}  # {M1: None, M2: M3, M3: None} structure

        # Identify paths from exposure to outcome
        paths = self._find_all_paths(self.exposure, self.outcome_variable)

        # Flatten paths into a unique list of potential mediators
        all_mediators = set(var for path in paths for var in path if var not in {self.exposure, self.outcome_variable})

        # Determine if mediators are parallel or sequential
        for path in paths:
            for i in range(len(path) - 1):
                if path[i] in all_mediators:
                    next_var = path[i + 1]
                    if next_var in all_mediators:
                        mediators[path[i]] = next_var  # Sequential mediator
                    elif path[i] not in mediators:
                        mediators[path[i]] = None  # Parallel mediator

        return mediators

    def _find_all_paths(self, start, end, path=None):
        """
        Recursively find all causal paths from start to end.

        :param start: Starting variable (exposure).
        :param end: Target variable (outcome).
        :param path: Current path (used in recursion).
        :return: List of paths (each path is a list of variables).
        """
        if path is None:
            path = []

        path = path + [start]

        if start == end:
            return [path]

        if start not in self.causal_dag:
            return []

        paths = []
        for node in self.causal_dag[start]:
            if node not in path:  # Avoid cycles
                new_paths = self._find_all_paths(node, end, path)
                paths.extend(new_paths)

        return paths

    def show(self, title="Causal Graph", save_path=None, figsize=(10, 7),
             node_color='lightblue', edge_color='gray', edge_width=1, arrow_size=20):
        show_graph(self.causal_dag, title=title, save_path=save_path, figsize=figsize,
                   node_color=node_color, edge_color=edge_color, edge_width=edge_width, arrow_size=arrow_size)
