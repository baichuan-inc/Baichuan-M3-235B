EXTRACT_CLAIMS_PROMPT = """\
### Introduction
Your task is to list relevant facts in an assistant's response to a given prompt. Your output will be used as the first step in the following fact- checking pipeline used to evaluate an assistant's response for factual correctness.

Fact-Checking Pipeline:
1. Given a prompt and assistant's response, list all relevant factual claims made by the assistant.
2. Separate the list of N claims into M manageable groups.
3. For each group of claims, fact-check each claim in the group by finding evidence supporting or refuting the claim.

### Instructions
- Carefully read the assistant's response to the prompt and identify all factual claims made by the assistant.
- You should isolate your focus to real-world facts (e.g., facts about news, people, places, events, etc.).
- If a statement within an assistant's response concerns something imaginative (e.g., the assistant is writing a fictional story or poem), then you should not consider this a factual claim.
- For each factual claim that you list, another assistant will be tasked with fact-checking it by finding evidence supporting or refuting the claim.
- Each claim that you list should be a single self-contained sentence, and replace pronouns or references with their actual terms.
- You should only consider claims that are relevant for answering the prompt. We consider a claim to be relevant if the subject of the claim is either exactly contained or related to any subject present in the prompt.
- If the same claim is repeated multiple times, you should only list it once.
- Try to list claims in the order that they appear in the assistant's response, so that related claims are grouped together.
- Note that the assistant did not have access to the web to make its response, so you should ignore any claims concerning what information is available on the web. For example, ignore claims such as "no reliable information is available on the [web or other online sources] about [topic]" or "I'm not finding [topic].”

### Formatting
Your response should be a list of claims in the following JSON format:

```json
[
    "fact_1",
    "fact_2",
    ...
]
```

### Example
Below is an example of a prompt and response.

Prompt:
Who is Barack Obama?

Response:
Barack Obama is an American politician and attorney who served as the 44th President of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African American president in U.S. history.

Output:
```json
[
    "Barack Obama is an American politician.",
    "Barack Obama is an attorney.",
    "Barack Obama served as the 44th President of the United States.",
    "Barack Obama served as president from 2009 to 2017.",
    "Barack Obama is a member of the Democratic Party.",
    "Barack Obama was the first African American president in United States history."
]
```

Note that you should expect the assistant's response to potentially be much longer than the one above, and could consist of up to 100 separate claims.

### Task
Prompt:
{prompt}

Response:
{response}
"""


FACT_CHECK_SYSTEM_PROMPT = """\
# Fact-Check System Workflow

## Objective
Evaluate the factual accuracy of claims based on search results using three levels:
- **true**: Completely accurate, search results unanimously support the claim
- **uncertain**: Unclear, contradictory information or insufficient evidence in search results
- **false**: Claim contradicts search results

## Verification Process

### Step 1: Initial Search
1. Use web_search tool with core keywords related to the claim
2. Analyze search result summaries to assess the evidence

### Step 2: Iterative Search (only if necessary)
- **Trigger condition**: Current search results are insufficient to determine truth/falsehood AND additional search is likely to provide critical missing information
- **Action**: Evaluate information gaps, formulate new search queries with different keywords or angles, and analyze the new results
- **Limit**: Maximum {max_search_times} search rounds total

### Step 3: Output Judgment
Use three judgment levels:
- **true**: Completely accurate, search results unanimously support the claim
- **uncertain**: Unclear, contradictory information or insufficient evidence in search results
- **false**: Claim contradicts search results

```json
{
    "reason": "Brief explanation based on search evidence",
    "judgment": "true/uncertain/false"
}
```

## Judgment Criteria
- **true**: Multiple reliable sources consistently confirm the claim is correct
- **false**: Reliable sources clearly contradict or deny the claim
- **uncertain**: Insufficient information, contradictory sources, or inadequate evidence found
"""
