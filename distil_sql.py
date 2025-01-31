import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

train_data = load_dataset('iknow-lab/bird', split='train')
schema_data = load_dataset('iknow-lab/bird-schema-v2', split='train')

schema_dict = {}
for schema in schema_data:
    contents = []
    for table in schema['schema']:
        contents.append(f"## Table Schema of {table['table']}\n{table['create_sql']}\n")
    for description in schema['description']:
        contents.append(f"## Table Description (CSV) of {description['table']}\n{description['description']}\n")
    schema_dict[schema['db_id']] = "\n".join(contents)
    # print(schema_dict[schema['db_id']])

PROMPT_INPUT = """Translate the following question into SQL query under the schema and description below:\n\n"""
PROMPT = """Please reason step by step, and put your final answer within \\boxed{}."""

def build_prompt(question, schema_id, evidence=None):
    prompt = PROMPT_INPUT + schema_dict[schema_id] + "\n\n" + PROMPT + f"\n\nQuestion: {question}" 
    if evidence:
        prompt += f"\n\nHint: {evidence}"
    return prompt

def build_instruction(item):
    return {
        "instruction": build_prompt(item['question'], item['db_id'], item['evidence'])
    }

prompt_template = "{{ instruction }}"

dataset = train_data.map(build_instruction)#.select(range(10))

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Exchange with another smol distilled r1

with Pipeline(
    name="distill-qwen-7b-r1-full",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 16384,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
        cuda_devices=[1],
    )
    prompt_column = "instruction"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=2,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="heegyu/bird-train-deepseek-r1-qwen-7b")