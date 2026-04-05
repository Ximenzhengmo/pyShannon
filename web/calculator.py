import ast
import itertools
import math
import operator
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probability import entropy


MAX_BASE_ASSIGNMENTS = 256
MAX_TOTAL_VARIABLES = 6


class CalculatorError(ValueError):
    pass


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCS = {
    "abs": abs,
    "max": max,
    "min": min,
    "round": round,
}


def get_exam_sample_payload():
    return {
        "base_variables": [
            {"key": "X", "latex": "X", "states": ["0", "1"]},
            {"key": "Y", "latex": "Y", "states": ["0", "1"]},
        ],
        "derived_variables": [
            {"key": "Z", "latex": "Z", "expression": "X*Y"},
        ],
        "joint_probabilities": [
            {"assignment": {"X": "0", "Y": "0"}, "probability": 0.3},
            {"assignment": {"X": "0", "Y": "1"}, "probability": 0.2},
            {"assignment": {"X": "1", "Y": "0"}, "probability": 0.4},
            {"assignment": {"X": "1", "Y": "1"}, "probability": 0.1},
        ],
        "normalize_probabilities": False,
    }


def parse_numeric_state(label):
    text = str(label).strip()
    if text == "":
        raise CalculatorError("State labels cannot be empty.")

    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return text


def format_state_value(value):
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)) and float(value).is_integer():
            return str(int(round(float(value))))
        return f"{float(value):.8g}"
    return str(value)


def parse_probability(value):
    try:
        prob = float(value)
    except (TypeError, ValueError) as exc:
        raise CalculatorError(f"Invalid probability value: {value!r}") from exc
    if prob < 0:
        raise CalculatorError("Probabilities must be non-negative.")
    return prob


def safe_eval_expression(expr, env):
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid derived-variable expression: {expr}") from exc

    def _eval(subnode):
        if isinstance(subnode, ast.Expression):
            return _eval(subnode.body)
        if isinstance(subnode, ast.Constant):
            return subnode.value
        if isinstance(subnode, ast.Name):
            if subnode.id not in env:
                raise CalculatorError(f"Unknown symbol `{subnode.id}` in expression `{expr}`.")
            return env[subnode.id]
        if isinstance(subnode, ast.BinOp):
            op_type = type(subnode.op)
            if op_type not in _BINARY_OPS:
                raise CalculatorError(f"Operator `{op_type.__name__}` is not allowed.")
            return _BINARY_OPS[op_type](_eval(subnode.left), _eval(subnode.right))
        if isinstance(subnode, ast.UnaryOp):
            op_type = type(subnode.op)
            if op_type not in _UNARY_OPS:
                raise CalculatorError(f"Unary operator `{op_type.__name__}` is not allowed.")
            return _UNARY_OPS[op_type](_eval(subnode.operand))
        if isinstance(subnode, ast.Call):
            if not isinstance(subnode.func, ast.Name) or subnode.func.id not in _ALLOWED_FUNCS:
                raise CalculatorError("Only abs, min, max, and round are allowed in expressions.")
            func = _ALLOWED_FUNCS[subnode.func.id]
            args = [_eval(arg) for arg in subnode.args]
            return func(*args)
        raise CalculatorError(f"Unsupported expression node `{type(subnode).__name__}`.")

    value = _eval(node)
    if isinstance(value, str):
        raise CalculatorError("Derived-variable expressions must evaluate to numeric values.")
    return value


def compact_join(tokens):
    if all(re.fullmatch(r"[A-Za-z]", token or "") for token in tokens):
        return "".join(tokens)
    return ",".join(tokens)


def sanitize_identifier(text):
    return re.sub(r"[^A-Za-z0-9_]", "_", text).strip("_") or "var"


def normalize_variable_specs(variable_specs):
    normalized = []
    seen_keys = set()
    for spec in variable_specs:
        key = str(spec.get("key", "")).strip()
        latex = str(spec.get("latex", key)).strip() or key
        state_labels = [str(item).strip() for item in spec.get("states", []) if str(item).strip()]

        if not key:
            raise CalculatorError("Each base variable needs a non-empty key.")
        if key in seen_keys:
            raise CalculatorError(f"Duplicate variable key `{key}`.")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise CalculatorError(f"Variable key `{key}` must be a valid identifier.")
        if len(state_labels) < 2:
            raise CalculatorError(f"Variable `{key}` needs at least two states.")
        if len(set(state_labels)) != len(state_labels):
            raise CalculatorError(f"Variable `{key}` has duplicated states.")

        seen_keys.add(key)
        normalized.append(
            {
                "key": key,
                "latex": latex,
                "state_labels": state_labels,
                "state_values": [parse_numeric_state(label) for label in state_labels],
            }
        )
    return normalized


def normalize_derived_specs(derived_specs, occupied_keys):
    normalized = []
    for spec in derived_specs:
        key = str(spec.get("key", "")).strip()
        latex = str(spec.get("latex", key)).strip() or key
        expression = str(spec.get("expression", "")).strip()

        if not key:
            raise CalculatorError("Each derived variable needs a non-empty key.")
        if key in occupied_keys:
            raise CalculatorError(f"Duplicate variable key `{key}`.")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise CalculatorError(f"Derived variable key `{key}` must be a valid identifier.")
        if not expression:
            raise CalculatorError(f"Derived variable `{key}` needs an expression.")

        occupied_keys.add(key)
        normalized.append({"key": key, "latex": latex, "expression": expression})
    return normalized


def generate_base_assignments(base_variables):
    state_lists = [var["state_labels"] for var in base_variables]
    rows = []
    for labels in itertools.product(*state_lists):
        rows.append({var["key"]: label for var, label in zip(base_variables, labels)})
    return rows


def build_probability_map(joint_probabilities, base_variables):
    base_keys = [var["key"] for var in base_variables]
    prob_map = {}
    for entry in joint_probabilities:
        assignment = entry.get("assignment", {})
        key = tuple(str(assignment.get(var_key, "")).strip() for var_key in base_keys)
        if "" in key:
            raise CalculatorError("Every probability row must provide a complete base-variable assignment.")
        prob_map[key] = parse_probability(entry.get("probability", 0.0))
    return prob_map


def build_augmented_assignments(base_variables, derived_variables, base_rows):
    derived_states = {spec["key"]: set() for spec in derived_variables}
    augmented_rows = []

    value_maps = {
        var["key"]: dict(zip(var["state_labels"], var["state_values"]))
        for var in base_variables
    }

    for base_row in base_rows:
        numeric_env = {key: value_maps[key][base_row[key]] for key in value_maps}
        derived_row = {}
        for spec in derived_variables:
            result = safe_eval_expression(spec["expression"], numeric_env)
            label = format_state_value(result)
            derived_row[spec["key"]] = label
            derived_states[spec["key"]].add(label)
            numeric_env[spec["key"]] = result
        augmented_rows.append((base_row, derived_row))

    return augmented_rows, derived_states


def sort_state_labels(labels):
    parsed = [(label, parse_numeric_state(label)) for label in labels]
    if all(not isinstance(value, str) for _, value in parsed):
        return [label for label, _ in sorted(parsed, key=lambda item: float(item[1]))]
    return sorted(labels)


def reorder_marginal(tensor, subset):
    subset = tuple(subset)
    if not subset:
        return np.asarray(tensor, dtype=np.float32)

    keep_axes = tuple(axis for axis in range(tensor.ndim) if axis in subset)
    sum_axes = tuple(axis for axis in range(tensor.ndim) if axis not in subset)
    if sum_axes:
        marginal = tensor.sum(axis=sum_axes)
    else:
        marginal = tensor

    if keep_axes == subset:
        return np.asarray(marginal, dtype=np.float32)

    permutation = [keep_axes.index(axis) for axis in subset]
    return np.transpose(marginal, axes=permutation).astype(np.float32)


def flatten(p):
    return np.asarray(p, dtype=np.float32).reshape(-1)


def sorted_union(*groups):
    axes = set()
    for group in groups:
        axes.update(group)
    return tuple(sorted(axes))


def subset_token(subset, variables, field):
    items = [variables[idx][field] for idx in subset]
    return compact_join(items)


def subset_code_name(subset, variables):
    parts = [sanitize_identifier(variables[idx]["key"].lower()) for idx in subset]
    return "p_" + "_".join(parts)


def subset_axis_repr(subset):
    if len(subset) == 1:
        return f"({subset[0]},)"
    return str(tuple(subset))


def entropy_process_latex(subset, variables):
    token = subset_token(subset, variables, "latex")
    lower = [sanitize_identifier(str(variables[idx]["key"]).lower()) for idx in subset]
    joined = ",".join(lower)
    return f"H({token})=-\\sum_{{{joined}}} p({joined})\\log_2 p({joined})"


def make_record(category, formula, latex, value, dependencies, process_latex, code, root_formula=None):
    return {
        "category": category,
        "formula": formula,
        "latex": latex,
        "value": float(value),
        "dependencies": list(dependencies),
        "process_latex": process_latex,
        "code": code,
        "root_formula": root_formula or formula,
    }


def build_code_prelude(variables, tensor_shape, nonzero_rows, state_to_index):
    lines = [
        "import numpy as np",
        "from probability import entropy",
        "",
        f"# Variable order: {[var['key'] for var in variables]}",
    ]
    for var in variables:
        lines.append(f"# {var['key']} states: {var['state_labels']}")
    lines += [
        "",
        f"tensor = np.zeros({tensor_shape}, dtype=np.float32)",
    ]

    for row in nonzero_rows:
        assignment = row["assignment"]
        index = tuple(state_to_index[var["key"]][assignment[var["key"]]] for var in variables)
        index_text = ", ".join(str(item) for item in index)
        comment = ", ".join(f"{var['key']}={assignment[var['key']]}" for var in variables)
        lines.append(f"tensor[{index_text}] = {float(row['probability']):.8g}  # {comment}")

    lines += [
        "",
        "def flatten(p):",
        "    return np.asarray(p, dtype=np.float32).reshape(-1)",
        "",
        "def reorder_marginal(tensor, subset):",
        "    subset = tuple(subset)",
        "    if not subset:",
        "        return np.asarray(tensor, dtype=np.float32)",
        "    keep_axes = tuple(axis for axis in range(tensor.ndim) if axis in subset)",
        "    sum_axes = tuple(axis for axis in range(tensor.ndim) if axis not in subset)",
        "    marginal = tensor.sum(axis=sum_axes) if sum_axes else tensor",
        "    if keep_axes == subset:",
        "        return np.asarray(marginal, dtype=np.float32)",
        "    permutation = [keep_axes.index(axis) for axis in subset]",
        "    return np.transpose(marginal, axes=permutation).astype(np.float32)",
        "",
        "results = {}",
    ]
    return "\n".join(lines)


def build_formula_records(variables, tensor):
    indices = tuple(range(len(variables)))
    formula_index = {}
    metrics = {
        "entropies": [],
        "conditional_entropies": [],
        "mutual_informations": [],
        "conditional_mutual_informations": [],
    }

    subsets = []
    for r in range(1, len(indices) + 1):
        subsets.extend(itertools.combinations(indices, r))

    for subset in subsets:
        token_text = subset_token(subset, variables, "key")
        token_latex = subset_token(subset, variables, "latex")
        formula = f"H({token_text})"
        latex = f"H({token_latex})"
        dist_name = subset_code_name(subset, variables)
        code = (
            f"{dist_name} = reorder_marginal(tensor, {subset_axis_repr(subset)})\n"
            f"results[{formula!r}] = entropy(flatten({dist_name}))"
        )
        record = make_record(
            "entropies",
            formula,
            latex,
            entropy(flatten(reorder_marginal(tensor, subset))),
            [],
            entropy_process_latex(subset, variables),
            code,
        )
        formula_index[formula] = record
        metrics["entropies"].append(record)

    for target in subsets:
        remaining = tuple(idx for idx in indices if idx not in target)
        for r in range(1, len(remaining) + 1):
            for condition in itertools.combinations(remaining, r):
                joint = sorted_union(target, condition)
                target_text = subset_token(target, variables, "key")
                condition_text = subset_token(condition, variables, "key")
                target_latex = subset_token(target, variables, "latex")
                condition_latex = subset_token(condition, variables, "latex")
                formula = f"H({target_text}|{condition_text})"
                latex = f"H({target_latex} \\mid {condition_latex})"
                dep_joint = f"H({subset_token(joint, variables, 'key')})"
                dep_condition = f"H({subset_token(condition, variables, 'key')})"
                value = formula_index[dep_joint]["value"] - formula_index[dep_condition]["value"]
                code = (
                    f"results[{formula!r}] = results[{dep_joint!r}] - results[{dep_condition!r}]"
                )
                process_latex = f"{latex} = {formula_index[dep_joint]['latex']} - {formula_index[dep_condition]['latex']}"
                record = make_record(
                    "conditional_entropies",
                    formula,
                    latex,
                    value,
                    [dep_joint, dep_condition],
                    process_latex,
                    code,
                )
                formula_index[formula] = record
                metrics["conditional_entropies"].append(record)

    for left in subsets:
        remaining = tuple(idx for idx in indices if idx not in left)
        for r in range(1, len(remaining) + 1):
            for right in itertools.combinations(remaining, r):
                if not (left < right):
                    continue
                joint = sorted_union(left, right)
                left_text = subset_token(left, variables, "key")
                right_text = subset_token(right, variables, "key")
                left_latex = subset_token(left, variables, "latex")
                right_latex = subset_token(right, variables, "latex")
                formula = f"I({left_text};{right_text})"
                latex = f"I({left_latex}; {right_latex})"
                dep_left = f"H({left_text})"
                dep_right = f"H({right_text})"
                dep_joint = f"H({subset_token(joint, variables, 'key')})"
                value = formula_index[dep_left]["value"] + formula_index[dep_right]["value"] - formula_index[dep_joint]["value"]
                code = (
                    f"results[{formula!r}] = results[{dep_left!r}] + results[{dep_right!r}] - results[{dep_joint!r}]"
                )
                process_latex = f"{latex} = {formula_index[dep_left]['latex']} + {formula_index[dep_right]['latex']} - {formula_index[dep_joint]['latex']}"
                record = make_record(
                    "mutual_informations",
                    formula,
                    latex,
                    value,
                    [dep_left, dep_right, dep_joint],
                    process_latex,
                    code,
                )
                formula_index[formula] = record
                metrics["mutual_informations"].append(record)

    for left in subsets:
        remaining_after_left = tuple(idx for idx in indices if idx not in left)
        for right_size in range(1, len(remaining_after_left) + 1):
            for right in itertools.combinations(remaining_after_left, right_size):
                if not (left < right):
                    continue
                remaining_after_pair = tuple(
                    idx for idx in indices if idx not in left and idx not in right
                )
                for cond_size in range(1, len(remaining_after_pair) + 1):
                    for condition in itertools.combinations(remaining_after_pair, cond_size):
                        left_text = subset_token(left, variables, "key")
                        right_text = subset_token(right, variables, "key")
                        condition_text = subset_token(condition, variables, "key")
                        left_latex = subset_token(left, variables, "latex")
                        right_latex = subset_token(right, variables, "latex")
                        condition_latex = subset_token(condition, variables, "latex")
                        formula = f"I({left_text};{right_text}|{condition_text})"
                        latex = f"I({left_latex}; {right_latex} \\mid {condition_latex})"
                        dep_left_cond = f"H({left_text}|{condition_text})"
                        dep_left_given_both = f"H({left_text}|{subset_token(sorted_union(right, condition), variables, 'key')})"
                        value = formula_index[dep_left_cond]["value"] - formula_index[dep_left_given_both]["value"]
                        code = (
                            f"results[{formula!r}] = results[{dep_left_cond!r}] - results[{dep_left_given_both!r}]"
                        )
                        process_latex = f"{latex} = {formula_index[dep_left_cond]['latex']} - {formula_index[dep_left_given_both]['latex']}"
                        record = make_record(
                            "conditional_mutual_informations",
                            formula,
                            latex,
                            value,
                            [dep_left_cond, dep_left_given_both],
                            process_latex,
                            code,
                        )
                        formula_index[formula] = record
                        metrics["conditional_mutual_informations"].append(record)

    formula_order = []
    for key in (
        "entropies",
        "conditional_entropies",
        "mutual_informations",
        "conditional_mutual_informations",
    ):
        formula_order.extend([record["formula"] for record in metrics[key]])

    return metrics, formula_index, formula_order


def compute_from_payload(payload):
    base_variables = normalize_variable_specs(payload.get("base_variables", []))
    if not base_variables:
        raise CalculatorError("Add at least one base variable.")

    derived_variables = normalize_derived_specs(
        payload.get("derived_variables", []),
        occupied_keys={var["key"] for var in base_variables},
    )

    if len(base_variables) + len(derived_variables) > MAX_TOTAL_VARIABLES:
        raise CalculatorError(f"At most {MAX_TOTAL_VARIABLES} total variables are supported.")

    base_rows = generate_base_assignments(base_variables)
    if len(base_rows) > MAX_BASE_ASSIGNMENTS:
        raise CalculatorError(
            f"Too many joint assignments ({len(base_rows)}). Reduce the number of variables or states."
        )

    prob_map = build_probability_map(payload.get("joint_probabilities", []), base_variables)
    normalize_probabilities = bool(payload.get("normalize_probabilities", False))

    base_keys = [var["key"] for var in base_variables]
    probabilities = []
    for row in base_rows:
        key = tuple(row[var_key] for var_key in base_keys)
        probabilities.append(prob_map.get(key, 0.0))

    total_probability = float(sum(probabilities))
    if total_probability <= 0:
        raise CalculatorError("The total probability must be greater than 0.")

    if normalize_probabilities:
        probabilities = [value / total_probability for value in probabilities]
        total_probability = 1.0
    elif not np.isclose(total_probability, 1.0, atol=1e-6):
        raise CalculatorError(
            f"The joint probabilities sum to {total_probability:.6f}, not 1. "
            "Use the normalize option or fix the input."
        )

    augmented_rows, derived_states = build_augmented_assignments(
        base_variables,
        derived_variables,
        base_rows,
    )

    all_variables = [
        {
            "key": var["key"],
            "latex": var["latex"],
            "state_labels": list(var["state_labels"]),
        }
        for var in base_variables
    ]
    for spec in derived_variables:
        all_variables.append(
            {
                "key": spec["key"],
                "latex": spec["latex"],
                "state_labels": sort_state_labels(derived_states[spec["key"]]),
            }
        )

    state_to_index = {
        var["key"]: {label: idx for idx, label in enumerate(var["state_labels"])}
        for var in all_variables
    }
    tensor_shape = tuple(len(var["state_labels"]) for var in all_variables)
    full_tensor = np.zeros(tensor_shape, dtype=np.float32)

    distribution_rows = []
    for (base_row, derived_row), probability in zip(augmented_rows, probabilities):
        full_row = dict(base_row)
        full_row.update(derived_row)
        index = tuple(state_to_index[var["key"]][full_row[var["key"]]] for var in all_variables)
        full_tensor[index] += probability
        distribution_rows.append({"assignment": full_row, "probability": probability})

    nonzero_rows = [row for row in distribution_rows if row["probability"] > 0]
    metrics, formula_index, formula_order = build_formula_records(all_variables, full_tensor)
    code_prelude = build_code_prelude(all_variables, tensor_shape, nonzero_rows, state_to_index)

    return {
        "summary": {
            "total_probability": total_probability,
            "base_assignment_count": len(base_rows),
            "variable_count": len(all_variables),
            "normalized_input": normalize_probabilities,
        },
        "variables": all_variables,
        "rows": nonzero_rows,
        "metrics": metrics,
        "formula_index": formula_index,
        "formula_order": formula_order,
        "code_prelude": code_prelude,
    }
