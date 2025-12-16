from typing import Dict, Any, Optional, List, Tuple
import numpy as np


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