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
                    sample_size: int = -1,
                    random_state: int=64) -> pd.DataFrame:
        if benchmark_name not in cls.DATASET_MAPPING:
            raise ValueError(f"Undefined benchmark: {benchmark_name}")
        data = pd.read_csv(cls.DATASET_MAPPING[benchmark_name])
        if "ground_truth" in data.columns:
            if benchmark_name == "data_table_analysis":
                data["ground_truth"] = data["ground_truth"].apply(json.loads)
            else:
                data["ground_truth"] = data["ground_truth"].apply(ast.literal_eval)
        if sample_size > 0 and len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        return data
    
    @classmethod
    def prepare_tasks_and_ground_truths(cls, 
                                        data: pd.DataFrame,
                                        input_column: str = "input") -> Tuple[List[str], List[Dict[str, Any]]]:
        tasks = []
        ground_truths = []
        for _, row in data.iterrows():
            tasks.append(row[input_column])
            ground_truths.append(row["ground_truth"])
        return tasks, ground_truths
    
    @classmethod
    def get_default_prompt_template(cls, benchmark_name: str) -> str:
        templates = {
            "data_table_analysis": (
                    "You are given a CSV-like string representation of a table (with header row, no index).\n"
                    "Extract a structured JSON object following the provided response format class.\n"
                    "Do not guess: if a value does not exist or is not applicable, return null.\n"
                    "Count rows excluding the header.\n"
                    "Infer each column type as 'str', 'int', or 'float'.\n"
                    "For string columns, set min/max to null.\n"
                    "If the 'Identifier' column is missing, set all Identifier-related fields to null.\n"
                    "For null/None entries, set string columns to '', and numerical to None.\n"
                    "Return only the structured JSON object.\n"
                ),
            "financial_entities": (
                "Identify and extract entities from the following financial news text into the following categories:\n"
                "\n"
                "Entity 1: Company \n"
                "⋆ Definition: Denotes the official or unofficial name of a registered company or a brand.\n"
                "⋆ Example entities: {Apple Inc.; Uber; Bank of America}\n"
                "\n"
                "Entity 2: Date \n"
                "⋆ Definition: Represents a specific time period, whether explicitly mentioned (e.g., 'year ended March 2020') or implicitly referred to (e.g., 'last month'), in the past, present, or future.\n"
                "⋆ Example entities: {June 2nd, 2010; quarter ended 2021; last week; prior year; Wednesday}\n"
                "\n"
                "Entity 3: Location \n"
                "⋆ Definition: Represents geographical locations, such as political regions, countries, states, cities, roads, or any other location, even when used as adjectives.\n"
                "⋆ Example entities: {California; Paris; 1280 W 12th Blvd; Americas; Europe}\n"
                "\n"
                "Entity 4: Money \n"
                "⋆ Definition: Denotes a monetary value expressed in any world currency, including digital currencies.\n"
                "⋆ Example entities: {$76.3 million; $4 Bn; Rs 33.80 crore; 1.2 BTC}\n"
                "\n"
                "Entity 5: Person \n"
                "⋆ Definition: Represents the name of an individual.\n"
                "⋆ Example entities: {Meg Whitman; Mr. Baker; Warren Buffet}\n"
                "\n"
                "Entity 6: Product \n"
                "⋆ Definition: Refers to any physical object or service manufactured or provided by a company to consumers, excluding references to businesses or sectors within the financial context.\n"
                "⋆ Example entities: {iPhone; Tesla model X; cloud services; Microsoft Windows 10; laptops; medical equipment; computer software; online classes; eye surgery}\n"
                "\n"
                "Entity 7: Quantity \n"
                "⋆ Definition: Represents any numeric value that is not categorized as Money, such as percentages, numbers, measurements (e.g., weight, length), or other similar quantities. Note that unit of measurements are also part of the entity.\n"
                "⋆ Example entities: {15%; 25,000 units; 2.75in; 100 tons}\n"
                "\n"
                "For each category:\n"
                "- Extract all relevant entities as a list of strings, preserving the wording from the text\n"
                "- Use None if no entities are found in that category\n"
                "- Only extract entities that are explicitly mentioned in the text itself, do not make inferences or reason about what entities might be implied based on URLs, domain names, or other indirect references\n"
                "- Extract individual items rather than compound or ranged entities (e.g., if a range or compound entity is mentioned, extract each individual item separately)\n"
                "\n"
                "Return the extracted information as a JSON object with all categories included, using None for cases where no entities are found.\n"
                ),
            "insurance_claims": (
                "You are an expert insurance claim processor. Extract structured information from insurance claim descriptions.\n"
                "For dates, use YYYY-MM-DD format.\n"
                "If a piece of information does not exist in the claim description, return null instead of making assumptions.\n"
                "Be thorough in extracting all available information and categorize appropriately.\n"
                ),
            "pii_extraction": (
                "Your task is to extract structured information about PII entities from the text provided by the user.\n"
                "For each field in the response format, extract the corresponding PII entity if it exists in the text.\n"
                "If a particular PII entity is not present in the text, set that field to null.\n"
                "Return the complete structured response with all fields, setting missing entities to null.\n"
                )
            }
        return templates.get(benchmark_name, "Your are a helplful assisant.")

def load_benchmark_data(benchmark_name: str, 
                       sample_size: int = None,
                       system_prompt: str = None) -> Tuple[List[str], List[Dict[str, Any]], pd.DataFrame]:
    data = DataLoader.load_dataset(benchmark_name, sample_size=sample_size)
    
    if system_prompt is None:
        system_prompt = DataLoader.get_default_prompt_template(benchmark_name)
    
    if benchmark_name in DataLoader.INPUT_COLUMN_MAPPING:
        input_column = DataLoader.INPUT_COLUMN_MAPPING.get(benchmark_name, "input")
    else:
        raise AssertionError(f"Undefined banchmark: {benchmark_name}")

    tasks, ground_truths = DataLoader.prepare_tasks_and_ground_truths(
        data, input_column=input_column
    )
    
    return tasks, ground_truths, data, system_prompt