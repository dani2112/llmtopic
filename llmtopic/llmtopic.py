import base64
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from hnne import HNNE
from sentence_transformers import SentenceTransformer
from guidance import models, system, user, assistant, gen, select
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from pydantic import BaseModel, Field
import guardrails as gd
from guardrails.validators import ValidChoices
import json


def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"


class ImageCategories(BaseModel):
    sex: str = Field(
        description="The sex/gender of the person in the image",
        validators=[ValidChoices(choices=["male", "female"])],
    )
    race: str = Field(
        description="The race of the person in the image",
        validators=[
            ValidChoices(choices=["white", "black", "asian", "indian", "other"])
        ],
    )
    age_group: str = Field(
        description="The age group of the person in the image",
        validators=[ValidChoices(choices=["child", "teen", "adult", "senior"])],
    )
    facial_hair: str = Field(
        description="Describes the facial hair of the person, if any",
        validators=[ValidChoices(choices=["none", "mustache", "beard", "goatee"])],
    )
    glasses: bool = Field(
        description="Indicates if the person in the image is wearing glasses",
    )
    hat: bool = Field(
        description="Indicates if the person in the image is wearing a hat",
    )
    makeup: bool = Field(
        description="Indicates if the person in the image is wearing makeup",
    )
    eye_contact: bool = Field(
        description="Indicates if the person in the image is making eye contact with the camera",
    )
    head_pose: str = Field(
        description="The orientation of the head in the image",
        validators=[
            ValidChoices(choices=["forward", "tilted", "side", "upward", "downward"])
        ],
    )


class LLMTopicImageSupervised:
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_threads: int = 8,
        n_threads_batch: int = 8,
        n_ctx: int = 2048,
        echo: bool = False,
    ):
        self._n_ctx = n_ctx
        chat_handler = Llava15ChatHandler(
            clip_model_path="/home/daniel/private_code/llmtopic/models/mmproj-model-f16.gguf"
        )
        self.llm = Llama(
            model_path,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            n_ctx=n_ctx,
            chat_handler=chat_handler,
            logits_all=True,
            verbose=echo,
        )
        self.all_topics = []

    def fit_transform(self, X: List[str], category_spec: BaseModel):
        
        for image_path in tqdm(X):
            try:
                data_uri = image_to_base64_data_uri(image_path)

                def llava_llm_api(
                    prompt: Optional[str] = None,
                    instruction: Optional[str] = None,
                    msg_history: Optional[list[dict]] = None,
                    **kwargs,
                ) -> str:
                    """Custom LLM API wrapper.

                    At least one of prompt, instruction or msg_history should be provided.

                    Args:
                        prompt (str): The prompt to be passed to the LLM API
                        instruction (str): The instruction to be passed to the LLM API
                        msg_history (list[dict]): The message history to be passed to the LLM API
                        **kwargs: Any additional arguments to be passed to the LLM API

                    Returns:
                        str: The output of the LLM API
                    """

                    schema_properties = {}
                    required_fields = []
                    for field_name, _ in category_spec.model_fields.items():
                        schema_properties[field_name] = {
                            "type": "string"
                        }  # Assuming all fields are of type string for simplicity
                        required_fields.append(field_name)

                    llm_output = self.llm.create_chat_completion(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that extracts properties from an image and return them in json format.",
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": data_uri}},
                                    {
                                        "type": "text",
                                        "text": prompt,
                                    },
                                ],
                            },
                        ],
                        response_format={
                            "type": "json_object",
                            "schema": {
                                "type": "object",
                                "properties": schema_properties,
                                "required": required_fields,
                            },
                        },
                        temperature=0.0,
                    )

                    text_output = llm_output["choices"][0]["message"]["content"]

                    return text_output

                prompt = """Generate JSON containing the properties described in the output format below:
                
                ${gr.complete_json_suffix_v3}
                """

                guard = gd.Guard.from_pydantic(output_class=category_spec, prompt=prompt)

                res = guard(
                    llava_llm_api,
                    max_tokens=4096,
                    num_reasks=0,
                    temperature=0.0,
                )
                self.all_topics.append(res.validated_output)
            except KeyboardInterrupt:
                raise
            except:
                print("Topic extraction failed.")
                self.all_topics.append(None)


class LLMTopic:
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_threads: int = 8,
        n_threads_batch: int = 8,
        n_ctx: int = 2048,
        chat_format: str = "chatml",
        echo: bool = False,
    ):
        self._n_ctx = n_ctx
        self.llm = models.LlamaCppChat(
            model_path,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            n_ctx=n_ctx,
            chat_format=chat_format,
            echo=echo,
        )
        with system():
            self.llm += "You are a helpful assistant."
        self.all_topics = []

    def _compute_topic_matrix(self):
        # Flatten the list and find unique topics
        unique_topics = np.array(
            list(set(topic for sublist in self.all_topics for topic in sublist))
        )

        # Create an empty DataFrame
        df = pd.DataFrame(
            columns=unique_topics, index=range(len(self.all_topics))
        ).fillna(0)

        # Fill in the DataFrame
        for i, topics in enumerate(self.all_topics):
            # total_topics = len(topics)
            for topic in topics:
                df.at[i, topic] = 1  # could also be += 1 / total_topics

        self.topic_matrix = df

    def fit_transform(
        self,
        X: List[str],
        max_topics: int = 3,
        custom_criteria: str = "Focus on the main topics relevant to customer satisfaction or dissatisfaction, being as specific as possible and using actual terms from the document.",
    ) -> List[Dict[int, str]]:
        self.all_topics = []  # Reset the topics
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
            try:
                tmp_llm = self.llm
                with user():
                    max_length = int(0.75 * self._n_ctx)  # 75% of _n_ctx

                    sample_words = sample.split()
                    prompt = ""

                    while len(prompt.split()) <= max_length and sample_words:
                        prompt = prompt_template1.format(
                            max_topics=max_topics,
                            document=" ".join(sample_words),
                            custom_criteria=custom_criteria,
                        )
                        if len(prompt.split()) > max_length:
                            sample_words.pop()
                        else:
                            break

                    if " ".join(sample_words) != sample:
                        print(
                            "Warning: The document was truncated to fit the length limit."
                        )
                    tmp_llm += prompt

                with assistant():
                    tmp_llm += "Number of Topics to Extract: " + select(
                        list(range(max_topics + 1)), name="num_topics"
                    )

                num_topics = int(tmp_llm["num_topics"])

                with assistant():
                    tmp_llm += "Extracted topics:\n"
                    for i in range(num_topics):
                        tmp_llm += f"{i+1}. " + gen(
                            name="topic",
                            list_append=True,
                            max_tokens=10,
                            stop="\n",
                            suffix="\n",
                            regex=r"[a-zA-Z][a-zA-Z][a-zA-Z][a-zA-Z\s]*",
                        )

                self.all_topics.append(tmp_llm["topic"])
            except:
                print("Topic extraction failed.")
                self.all_topics.append([])
        self._compute_topic_matrix()

        return self.all_topics

    def summarize_topics(self, max_cluster_size=None):
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            unique_topics = np.array(
                list(set(topic for sublist in self.all_topics for topic in sublist))
            )
            topic_embeddings = model.encode(unique_topics)

            hnne = HNNE()
            _ = hnne.fit_transform(topic_embeddings)

            partitions = hnne.hierarchy_parameters.partitions[:, ::-1]
            # partition_sizes = hnne.hierarchy_parameters.partition_sizes[::-1]
            number_of_levels = partitions.shape[1]

            if max_cluster_size is None:
                max_cluster_size = round(np.mean(np.bincount(partitions[:, -1])))

            topic_mappings = []

            for partition_idx in range(number_of_levels):
                partitioning = partitions[:, partition_idx]
                clusters, cluster_sizes = np.unique(partitioning, return_counts=True)
                for cluster, cluster_size in tqdm(
                    zip(clusters, cluster_sizes), total=len(clusters)
                ):
                    if cluster == -1:
                        continue
                    if cluster_size > max_cluster_size and not partition_idx == (
                        number_of_levels - 1
                    ):
                        continue
                    tmp_llm = self.llm
                    cluster_indices = np.where(partitioning == cluster)[0]
                    cluster_topics = unique_topics[cluster_indices]
                    original_cluster_topics = cluster_topics
                    if len(cluster_topics) > max_cluster_size:
                        cluster_topics = np.random.choice(
                            cluster_topics, size=max_cluster_size, replace=False
                        )

                    prompt_template = (
                        "Evaluate the following list of topics. Determine if they are semantically similar and not contradicting each other. "
                        "Provide a brief explanation for your decision. If they are similar, identify the most prototypical topic for the cluster. "
                        "Choose a prototypical topic that describes the overall concept well and is not too specific. If there is any nuances regarding sentiment, choose a neutral one, e.g., if there is cheap food or food choose food."
                        "If they are not similar, briefly explain why.\n"
                        "\nExamples:"
                        "\n- Topics: ['Fast Service', 'Quick Response', 'Speedy Delivery'] → Semantically Similar: yes, Explanation: because all topics relate to speed and efficiency. Most Prototypical Topic: Fast Service"
                        "\n- Topics: ['Friendly Staff', 'Poor Hygiene', 'Tasty Food'] → Semantically Similar: no, Explanation: because the topics cover different aspects (staff behavior, cleanliness, and food quality) with no unifying theme."
                        "\n- Topics: ['Organic Ingredients', 'Fresh Produce', 'Local Farming'] → Semantically Similar: yes, Explanation: because all topics are related to natural and local food sourcing. Most Prototypical Topic: Organic Ingredients"
                        "\n- Topics: ['Online Booking', 'Flight Delay', 'Airport Security'] → Semantically Similar: no, Explanation: because the topics cover different stages and aspects of air travel, without a central theme."
                        "\n- Topics: ['Budget Planning', 'Financial Advice', 'Investment Strategies', 'Saving Accounts'] → Semantically Similar: yes, Explanation: because all topics pertain to financial management and planning. Most Prototypical Topic: Financial Advice"
                        "\n- Topics: ['Thriller Genre', 'Romantic Novels', 'Science Fiction'] → Semantically Similar: no, Explanation: because each topic refers to a distinct literary genre without overlap."
                        "\n- Topics: ['Renewable Energy', 'Solar Panels', 'Wind Turbines', 'Sustainable Resources'] → Semantically Similar: yes, Explanation: because all topics relate to sustainable and renewable energy sources. Most Prototypical Topic: Renewable Energy"
                        "\n- Topics: ['Historical Landmarks', 'Museum Exhibits', 'Art Galleries'] → Semantically Similar: yes, Explanation: because all topics are connected to cultural and historical attractions. Most Prototypical Topic: Historical Landmarks"
                        "\n\nOriginal Topic List: {topics}\n"
                    )

                    max_topics = len(
                        cluster_topics
                    )  # Set the maximum number of topics you want to allow

                    with user():
                        tmp_llm += prompt_template.format(
                            max_topics=max_topics, topics=str(cluster_topics)
                        )

                    with assistant():
                        tmp_llm += (
                            "Semantically Similar: "
                            + select(["yes", "no"], name="semantically_similar")
                            + ",  Explanation: "
                            + gen(max_tokens=30, stop=[".", "\n", "!"])
                        )
                        if tmp_llm["semantically_similar"] == "yes":
                            tmp_llm += "\nMost Prototypical Topic: " + select(
                                cluster_topics, name="prototypical_topic"
                            )

                            topic_mappings.append(
                                {
                                    "old": original_cluster_topics.tolist(),
                                    "new": tmp_llm["prototypical_topic"],
                                }
                            )
                            partitions[cluster_indices, partition_idx + 1 :] = -1
        except:
            print(
                "Topic summarization failed. Maybe try to decrease cluster size or increase context length."
            )
            return self.all_topics

        # Creating a dictionary for easy lookup
        topic_lookup = {
            old_topic: mapping["new"]
            for mapping in topic_mappings
            for old_topic in mapping["old"]
        }

        # Applying the mapping
        self.all_topics = [
            [topic_lookup.get(topic, topic) for topic in doc] for doc in self.all_topics
        ]

        self._compute_topic_matrix()

        return self.all_topics


if __name__ == "__main__":
    import pandas as pd
    import os

    model = LLMTopicImageSupervised(
        model_path="/home/daniel/private_code/llmtopic/models/llava-v1.6-mistral-7b.Q4_K_M.gguf"
    )
    df = pd.read_csv(
        "/home/daniel/code/pycon24-presentation/affectnet_hq_png_with_predictions.csv"
    )
    df = df.sample(20)
    prefix_path = "/home/daniel/code/pycon24-presentation"
    df["image"] = df["image"].apply(lambda x: os.path.join(prefix_path, x))
    model.fit_transform(df["image"].tolist(), category_spec=ImageCategories)

    df["attributes"] = model.all_topics
    df.to_json("df_with_attributes.json", orient="records", indent=4)

