from collections import defaultdict
import json
from deepeval.metrics import (
    BiasMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval


class LLMMetrics:
    def __init__(self, model):
        self.model = model
        self.metrics_data = defaultdict(dict)
        self.metric_keys = {
            "bias": ["score", "reason", "opinions", "verdicts"],
            "faith": ["score", "reason", "truths", "claims", "verdicts"],
            "context_relevancy": ["score", "reason", "verdicts_list"],
            "answer_relevancy": ["score", "reason"],
        }

    def answer_relevancy_metric(self, query: str, answer: str):
        print(f"\n\n{'=' * 25}Answer Relevancy Metric{'=' * 25} Started\n\n")
        metric = AnswerRelevancyMetric(
            threshold=0.7, model=self.model, include_reason=True
        )
        test_case = LLMTestCase(input=query, actual_output=answer)

        metric.measure(test_case)
        self.prepare_data(metric, "answer_relevancy")

    def bias_metric(self, query: str, answer: str):
        print(f"\n\n{'=' * 25}Bias Testing{'=' * 25} Started")
        metric = BiasMetric(threshold=0.5, model=self.model, verbose_mode=True)
        test_case = LLMTestCase(
            input=query,
            # Replace this with the actual output from your LLM application
            actual_output=answer,
        )
        metric.measure(test_case)
        self.prepare_data(metric, "bias")

    def faithfulness_metric(self, query: str, answer: str, context: list):
        print(f"\n\n{'=' * 25}Faithfulness Testing{'=' * 25} Started")
        metric = FaithfulnessMetric(threshold=0.5, model=self.model, verbose_mode=True)
        test_case = LLMTestCase(
            input=query,
            # Replace this with the actual output from your LLM application
            actual_output=answer,
            retrieval_context=context,
        )
        metric.measure(test_case)
        self.prepare_data(metric, "faith")

    def contextual_relevancy_metric(self, query: str, answer: str, context: list):
        print(f"\n\n{'=' * 25}Contextual Relevancy Testing{'=' * 25} Started")
        metric = ContextualRelevancyMetric(
            threshold=0.5, model=self.model, verbose_mode=True
        )
        test_case = LLMTestCase(
            input=query,
            # Replace this with the actual output from your LLM application
            actual_output=answer,
            retrieval_context=context,
        )
        metric.measure(test_case)
        self.prepare_data(metric, "context_relevancy")

    def g_eval_metrics(self):
        # https://docs.confident-ai.com/docs/metrics-llm-evals
        correctness_metric = GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
        )
        test_case = LLMTestCase(
            input="The dog chased the cat up the tree, who ran up the tree?",
            actual_output="It depends, some might consider the cat, while others might argue the dog.",
            expected_output="The cat.",
        )

        correctness_metric.measure(test_case)
        print(correctness_metric.score)
        print(correctness_metric.reason)

    def clean_data(self, data, metric_name):
        """Convert data to JSON-safe format and filter specific keys."""
        if isinstance(data, dict):
            return {
                k: self.clean_data(v, metric_name)
                for k, v in data.items()
                if k in self.metric_keys[metric_name]
            }
        elif isinstance(data, list):
            return [self.clean_data(item, metric_name) for item in data]
        elif hasattr(data, "__dict__"):
            return self.clean_data(vars(data), metric_name)
        return data  # Return as is for strings, numbers, etc

    def prepare_data(self, metric, metric_name: str):
        result = {}
        for key in self.metric_keys[metric_name]:
            value = metric.__dict__[key]
            if key == "verdicts":
                verdicts_data = []
                for verdict in value:
                    flag = verdict.verdict
                    reason = verdict.reason
                    verdicts_data.append({"verdict": flag, "reason": reason})
                result[key] = verdicts_data
            elif key == "verdicts_list":
                result[key] = f"{value}"
            else:
                result[key] = value

        self.metrics_data[metric_name] = result
