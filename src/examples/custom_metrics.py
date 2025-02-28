from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from deepeval.test_case import LLMTestCaseParams
from bs4 import BeautifulSoup
from langchain.chains import llm
from langchain_openai import ChatOpenAI


class HTMLValidation:
    def __init__(self):
        self.name = "HTML Validation"
        self.criteria = "Check if the actual output is a valid HTML document."
        self.evaluation_steps = [
            "Parse the actual output as HTML table tags using BeautifulSoup.",
            "Check if it contains valid HTML table structure (e.g., <table>, <th>, <td> tags).",
            "Ensure the document does not have unclosed or misordered tags."
        ]
        self.metric = GEval(
            name=self.name,
            criteria=self.criteria,
            evaluation_steps=self.evaluation_steps,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        )

    def is_valid_html(self, html_text: str) -> bool:
        """Checks if the provided text is a valid HTML document."""
        try:
            soup = BeautifulSoup(html_text, "html.parser")
            return bool(soup.find())  # Returns True if there are valid HTML elements
        except Exception:
            return False

    def evaluate(self, input: str, expected_output: str, actual_output: str):
        test_case = LLMTestCase(
            input=input, actual_output=actual_output
        )

        # Run GEval metric
        self.metric.measure(test_case)
        return self.metric.score, self.metric.reason


def general_llm(model: str, query: str):

    llm = ChatOpenAI(model=model)

    print(f"\n\n{'=' * 25}LLM RESPONSE{'=' * 25}\n\n")
    answer = llm.invoke(query)
    return answer


html_evaluator = HTMLValidation()
model = "gpt-3.5-turbo"
query = "Give me a comparision between number runs scored by Virat, Rohit and Dhoni and return it as an HTML Table. It should only be as html table do not add any other text before and after"
result = general_llm(model, query)


print(result.content)
score_valid, reason_valid = html_evaluator.evaluate(
    query, result.content
)
print(f"Valid Output -> Score: {score_valid}, Reason: {reason_valid}")
