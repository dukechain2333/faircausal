import pandas as pd

from _BuildCausalModel import build_causal_model, generate_beta_params


class CausalDataReader:
    def __init__(self, *args, **kwargs):
        if len(args) == 3:
            self.__load_manually(*args, **kwargs)
        elif len(args) == 1:
            self.__load_auto(*args, **kwargs)
        else:
            raise ValueError("Invalid number of arguments. Expected either 1 or 3 arguments.")

    def __load_manually(self, data: pd.DataFrame, beta_dict: dict, causal_dag: dict, **kwargs):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not isinstance(beta_dict, dict):
            raise TypeError("beta_dict must be a dictionary.")
        if not isinstance(causal_dag, dict):
            raise TypeError("causal_dag must be a dictionary.")

        if not self.__is_valid_causal_dag(causal_dag):
            raise ValueError("The input graph must be a Directed Acyclic Graph (DAG).")
        self.causal_dag = causal_dag

        are_nodes_in_df, missing_node = self.__check_dag_nodes_in_dataframe(causal_dag, data)
        if not are_nodes_in_df:
            raise ValueError(f"Node {missing_node} is not present in the data.")
        self.data = data

        beta_count = self.__count_linear_regression_parameters(causal_dag)
        if len(beta_dict) != beta_count:
            raise ValueError(f"Invalid number of beta coefficients, expected {beta_count}.")
        self.beta_dict = beta_dict

    def __load_auto(self, data: pd.DataFrame, **kwargs):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        self.data = data

        self.sm, self.causal_dag = build_causal_model(data)
        self.beta_dict = generate_beta_params(self.causal_dag, data)

    @staticmethod
    def __is_valid_causal_dag(dag: dict):
        for node, children in dag.items():
            for child in children:
                if child not in dag:
                    raise ValueError(f"Node {child} is not present in the graph.")

        def has_cycle(dag: dict):
            visited = set()
            stack = set()

            def visit(node):
                if node in stack:
                    return True
                if node in visited:
                    return False
                visited.add(node)
                stack.add(node)
                for child in dag.get(node, []):
                    if visit(child):
                        return True
                stack.remove(node)
                return False

            return any(visit(node) for node in dag)

        if has_cycle(dag):
            return False

        return True

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

    def get_model(self):
        return {
            'data': self.data,
            'beta_dict': self.beta_dict,
            'causal_dag': self.causal_dag
        }

    def show_model_graph(self):
        pass
