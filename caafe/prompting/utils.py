import re


# if ```python on response, or ``` on response, or whole response is code, return code
def extract_code(response: str) -> str:
    """Extract code content from text that may contain code blocks.

    Args:
        response: Input text that might contain code blocks
    Returns:
        Extracted code content or original text if no code blocks found
    """
    response = response.strip()
    code_match = re.search(
        r"```(?:\w+)?\s*(.*?)```",
        response,
        re.DOTALL,
    )
    return code_match.group(1).strip() if code_match else response
