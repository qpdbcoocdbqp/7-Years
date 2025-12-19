import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from functools import partial
from typing import Dict, Any, Optional, List, Tuple


def judge_function(true_one: pa.Table, fits: pa.Table) -> pa.Table:
    mapper = fits.join(true_one, keys=["key"], join_type="full outer", right_suffix="_true")
    mapper = mapper.append_column(
        "is_correct", pc.equal(mapper.column("val"), mapper.column("val_true"))
        )
    mapper_id = mapper.group_by(["idx"]).aggregate([
        ("is_correct", "sum")
        ]).sort_by([("is_correct_sum", "descending")]).column("idx")[0]    
    result = mapper.filter(pc.equal(pc.field("idx"), mapper_id)).select(["key", "is_correct"])
    del mapper, mapper_id
    return result

def judge(true_dict: dict, fit_dict: dict, flat_transform: Any=None) -> pa.Table:
    if flat_transform is not None:
        fits = flat_transform([fit_dict]).to_pylist()
        true_list = flat_transform([true_dict]).to_pylist()
    else:
        fits = [fit_dict]
        true_list = [true_dict]

    fits = pa.concat_tables([
        pa.Table.from_arrays(
            [
                [i] * len(item),
                list(item.keys()),
                list(map(str, item.values()))
            ],
            names=["idx", "key", "val"]
        )
        for i, item in enumerate(fits)]
        )
    true_list = [
        pa.Table.from_arrays(
            [
                list(item.keys()),
                list(map(str, item.values()))
            ],
            names=["key", "val"]
        )
        for item in true_list
        ]
    fit_judge_function = partial(judge_function, fits=fits)
    judge_df = pa.concat_tables(list(map(fit_judge_function, true_list)))
    del fits, true_list, fit_judge_function
    return judge_df

class StructuredEvaluator:
    def evaluate_response_accuracy_with_breakdown(
        self, parsed_responses: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[Dict[str, bool]]]:
        overall_results = []
        field_breakdowns = []

        for pred_dict, true_dict in zip(parsed_responses, ground_truths):
            is_correct = self._compare_json_objects(pred_dict, true_dict)
            overall_results.append(1 if is_correct else 0)
            field_breakdown = self._calculate_field_breakdown(pred_dict, true_dict)
            field_breakdowns.append(field_breakdown)

        return np.array(overall_results), field_breakdowns

    def _compare_json_objects(
        self, pred_dict: Dict[str, Any], true_dict: Dict[str, Any]
    ) -> bool:
        return pred_dict == true_dict

    def _calculate_field_breakdown(
        self, pred_dict: Dict[str, Any], true_dict: Dict[str, Any]
    ) -> Dict[str, bool]:
        comparison_result = self._compare_fields_detailed(true_dict, pred_dict)
        return comparison_result["field_correctness"]

    def _compare_fields_detailed(
        self, ground_truth: Dict[str, Any], extracted: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if extracted is None:
            return {
                "overall_accuracy": 0.0,
                "field_accuracies": {},
                "extraction_failed": True,
                "errors": ["extract failed"],
                "exact_matches": 0,
                "total_fields": len(ground_truth),
                "field_correctness": {},
            }

        errors = []
        field_accuracies = {}
        field_correctness = {}
        exact_matches = 0
        total_fields = len(ground_truth)

        for field, gt_value in ground_truth.items():
            field_path = f"{field}"
            extracted_value = extracted.get(field)

            is_correct = gt_value == extracted_value
            field_correctness[field_path] = is_correct

            if is_correct:
                exact_matches += 1
            else:
                errors.append(
                    f"{field_path}: expected '{gt_value}', responsed '{extracted_value}'"
                )

            field_accuracies[field_path] = is_correct

        return {
            "overall_accuracy": (
                exact_matches / total_fields if total_fields > 0 else 0.0
            ),
            "field_accuracies": field_accuracies,
            "extraction_failed": False,
            "errors": errors,
            "exact_matches": exact_matches,
            "total_fields": total_fields,
            "field_correctness": field_correctness,
            "exact_match_rate": (
                exact_matches / total_fields if total_fields > 0 else 0.0
            ),
        }