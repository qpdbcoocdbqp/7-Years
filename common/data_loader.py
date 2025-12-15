import ast
import pandas as pd
import json
from typing import Dict, Any, List, Tuple


class DataLoader:
    DATASET_MAPPING = {
        "data_table_analysis": "hf://datasets/Cleanlab/data-table-analysis/data_table_analysis.csv",
        "financial_entities": "hf://datasets/Cleanlab/fire-financial-ner-extraction/fire_financial_ner_extraction.csv",
        "insurance_claims": "hf://datasets/Cleanlab/insurance-claims-extraction/insurance_claims_extraction.csv",
        "pii_extraction": "hf://datasets/Cleanlab/pii-extraction/pii_extraction.csv"
    }
    INPUT_COLUMN_MAPPING = {
        "data_table_analysis": "table",
        "financial_entities": "text",
        "insurance_claims": "claim_text",
        "pii_extraction": "text"
    }
    
    @classmethod
    def load_dataset(cls, benchmark_name: str, 
                    sample_size: int = None) -> pd.DataFrame:
        if benchmark_name not in cls.DATASET_MAPPING:
            raise ValueError(f"Undefined benchmark: {benchmark_name}")
        data = pd.read_csv(cls.DATASET_MAPPING[benchmark_name])
        if "ground_truth" in data.columns:
            if benchmark_name == "data_table_analysis":
                data["ground_truth"] = data["ground_truth"].apply(json.loads)
            else:
                data["ground_truth"] = data["ground_truth"].apply(ast.literal_eval)
        if sample_size and len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
        return data
    
    @classmethod
    def prepare_prompts_and_ground_truths(cls, 
                                        data: pd.DataFrame,
                                        prompt_template: str,
                                        input_column: str = "input") -> Tuple[List[str], List[Dict[str, Any]]]:
        prompts = []
        ground_truths = []
        for _, row in data.iterrows():
            prompt = prompt_template.format(input=row[input_column])
            prompts.append(prompt)
            ground_truths.append(row["ground_truth"])
        return prompts, ground_truths
    
    @classmethod
    def get_default_prompt_template(cls, benchmark_name: str) -> str:
        templates = {
            "data_table_analysis": (
                "請分析以下 CSV 資料表並提取結構化資訊：\n\n"
                "{input}\n\n"
                "請提供以下資訊：\n"
                "- 資料表的主題和描述\n"
                "- 欄位數量和列數量\n"
                "- 主要欄位類型\n"
                "- 資料品質評估\n"
                ),
            "financial_entities": (
                "請從以下金融文本中提取相關實體：\n\n"
                "{input}\n\n"
                "請識別並提取所有金融相關的實體，包括公司名稱、金額、日期等。\n"
                ),
            "insurance_claims": (
                "請從以下保險理賠文件中提取結構化資訊：\n\n"
                "{input}\n\n"
                "請提取理賠相關的所有重要資訊，包括理賠編號、日期、金額、事故描述等。\n"
                ),
            "pii_extraction": (
                "請從以下文本中識別並提取個人識別資訊 (PII):\n\n"
                "{input}\n\n"
                "請識別所有 PII 類型，包括姓名、電話、地址、電子郵件等，並標註其類型。\n"
                )
            }
        return templates.get(benchmark_name, "請處理以下輸入：\n\n{input}")

def load_benchmark_data(benchmark_name: str, 
                       sample_size: int = None,
                       prompt_template: str = None) -> Tuple[List[str], List[Dict[str, Any]], pd.DataFrame]:
    data = DataLoader.load_dataset(benchmark_name, sample_size=sample_size)
    
    if prompt_template is None:
        prompt_template = DataLoader.get_default_prompt_template(benchmark_name)
    
    if benchmark_name in DataLoader.INPUT_COLUMN_MAPPING:
        input_column = DataLoader.INPUT_COLUMN_MAPPING.get(benchmark_name, "input")
    else:
        raise AssertionError(f"Undefined banchmark: {benchmark_name}")

    prompts, ground_truths = DataLoader.prepare_prompts_and_ground_truths(
        data, prompt_template, input_column=input_column
    )
    
    return prompts, ground_truths, data