import json

from voxweave.backend import run_conversation

result = run_conversation([0] * 4 + [0.1] * 8 + [0] * 8 + [0.1] * 8, 8000, frame_size=4)
print(
    json.dumps(
        {
            "turns": result["turns"],
            "responses": [
                item["content"][0]["text"]
                for item in result["messages"]
                if item["role"] == "assistant"
            ],
        }
    )
)
