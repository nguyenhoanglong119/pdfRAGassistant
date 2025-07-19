from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

def load_llm():
    model_name = "google/flan-t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
    )
    return HuggingFacePipeline(pipeline=pipe)
