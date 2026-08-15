from langsmith import Client, evaluate, traceable
from Inventery_Agent import run
from langchain.tools import tool
from utils import get_sentence_similarity
from dotenv import load_dotenv
load_dotenv()


@traceable
def target(inputs: dict) -> dict:
    question = inputs["question"]
    answer = run(question)
    return {"answer": answer}


client = Client()
dataset_name = "inventorydata"

if not client.has_dataset(dataset_name=dataset_name):
    client.create_dataset(dataset_name=dataset_name)

client.create_examples(
    dataset_name=dataset_name,
    examples=[
        {
            "inputs": {"question": "What is the inventory status of iPhone 15?"},
            "outputs": {"answer": "The iPhone 15 is in stock with 2 units available."},
        },
        {
            "inputs": {"question": "Do you have any Airpods in stock?"},
            "outputs": {"answer": "Airpods are currently out of stock, with 0 units available."},
        },
        {
            "inputs": {"question": "Is MacBook Air M3 available?"},
            "outputs": {"answer": "The MacBook Air M3 is in stock with 5 units available."},
        },
        {
            "inputs": {"question": "How many iPhone 15 units are left?"},
            "outputs": {"answer": "There are 2 units of the iPhone 15 available in stock."},
        },
        {
            "inputs": {"question": "Do you have AirPods Pro in stock?"},
            "outputs": {"answer": "The product is not available in our inventory."},
        },
        {
            "inputs": {"question": "Is the Samsung Galaxy S24 available?"},
            "outputs": {"answer": "The product is not available in our inventory."},
        },
        {
            "inputs": {"question": "I want to travel to the moon, how much will it cost?"},
            "outputs": {"answer": "Sorry, I can't assist with that."},
        },
    ],
)

def semantic_match(example, run):
    expected=example.outputs ["answer"]
    actual = run.outputs ["answer"]
    sim = get_sentence_similarity(expected, actual)
    return {
        "key": "semantic_match",
        "score": float(sim)
    }

evaluate(
    target,
    client=client,
    data=dataset_name,
    evaluators=[semantic_match],
    experiment_prefix="Inventory_Agent_Evaluation_using_Qwen"
    
)