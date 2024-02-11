from typing import List, Dict
from tqdm import tqdm
from guidance import models, system, user, assistant, gen, select

class LLMTopic:
    def __init__(self, 
                 model_path: str, 
                 n_gpu_layers: int = -1, 
                 n_threads: int = 8, 
                 n_threads_batch: int = 8, 
                 n_ctx: int = 2048, 
                 chat_format: str = "chatml", 
                 echo: bool = False):
        self.llm = models.LlamaCppChat(model_path, n_gpu_layers=n_gpu_layers, n_threads=n_threads, n_threads_batch=n_threads_batch, n_ctx=n_ctx, chat_format=chat_format, echo=echo)
        with system():
            self.llm += "You are a helpful assistant."
        self.all_topics = []

    def fit_transform(self, X: List[str], max_topics: int = 3, custom_criteria: str = "Focus on the main topics relevant to customer satisfaction or dissatisfaction, being as specific as possible and using actual terms from the document.") -> List[Dict[int, str]]:
        prompt_template1 = (
            "Please analyze the following document and determine the key topics to extract, "
            "ranging from 0 to {max_topics}. The topics should be directly derived from the document's content. "
            "Each extracted topic should be no more than three words long and must reflect the main points or themes present in the document.\n\n"
            "Criteria for topics: {custom_criteria}\n\n"
            "After analyzing the document, present the topics in a numbered list format. Each topic should be a concise representation of a specific theme or point from the document. "
            "Avoid using generic placeholders. Instead, use actual terms or phrases found in the text.\n\n"
            "Document:\n{document}\n\n"
        )

        for sample in tqdm(X, desc="Processing samples", unit="sample"):

            tmp_llm = self.llm
            with user():
                tmp_llm += prompt_template1.format(max_topics=max_topics, document=sample, custom_criteria=custom_criteria)

            with assistant():
                tmp_llm += "Number of Topics to Extract: " + select(list(range(max_topics+1)), name="num_topics")

            num_topics = int(tmp_llm["num_topics"])
            
            with assistant():
                tmp_llm += "Extracted topics:\n"
                for i in range(num_topics):
                    tmp_llm += f"{i+1}. " + gen(name="topic", list_append=True, max_tokens=10, stop="\n", suffix="\n", regex=r"[a-zA-Z\s]+")
               
            self.all_topics.append(tmp_llm["topic"])

        return self.all_topics
