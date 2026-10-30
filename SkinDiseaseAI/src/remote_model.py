"""Call the workspace's OpenAI-compatible remote chat model.

Set OPENAI_API_KEY before running. The API key is deliberately read from the
environment and is never stored in this repository.
"""

import os
import sys

from openai import OpenAI


DEFAULT_BASE_URL = "https://rsjwdt9.abc-tunnel.us/v1"
DEFAULT_MODEL = "Fable 5.1"


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Set it to the API key issued by your "
            "router; do not use the full tunnel URL as the key."
        )

    prompt = " ".join(sys.argv[1:]).strip() or "Hello"
    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.choices[0].message.content or "")


if __name__ == "__main__":
    main()
