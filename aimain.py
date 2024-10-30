from dotenv import load_dotenv
import os
from openai import OpenAI
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize OpenAI client with API key from .env
client = OpenAI()


def make_openai_call(messages):
    # Make API call
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content


def clarity_agent(user_input: str) -> str:
    print("\n1️⃣ Clarity Agent analyzing your goals...")

    # Define messages for the chat
    messages = [
        {
            "role": "system",
            "content": "You are a Clarity Agent helping entrepreneurs understand how to monetize with AI. Be direct and practical.",
        },
        {
            "role": "user",
            "content": f"Help me understand how to make money with AI based on this context: {user_input}",
        },
    ]

    response = make_openai_call(messages)
    print("✅ Clarity analysis complete")
    return response


def niche_agent(clarity_response: str) -> str:
    print("\n2️⃣ Niche Agent identifying target market...")
    messages = [
        {
            "role": "system",
            "content": "You are a Niche Agent helping identify specific market opportunities and ideal customer profiles.",
        },
        {
            "role": "user",
            "content": f"Based on this context, identify the most profitable niche and ideal customer avatar: {clarity_response}",
        },
    ]
    response = make_openai_call(messages)
    print("✅ Niche analysis complete")
    return response


def action_agent(niche_response: str) -> str:
    print("\n3️⃣ Action Agent creating specific steps...")
    messages = [
        {
            "role": "system",
            "content": "You are an Action Agent providing specific, actionable steps for finding and acquiring customers.",
        },
        {
            "role": "user",
            "content": f"Provide specific action steps for this niche and avatar: {niche_response}",
        },
    ]
    response = make_openai_call(messages)
    print("✅ Action plan complete")
    return response


def business_strategist(all_responses: dict, original_input: str) -> str:
    print("\n4️⃣ Business Strategist synthesizing final plan...")
    messages = [
        {
            "role": "system",
            "content": "You are a Business Strategist creating clear, actionable business plans.",
        },
        {
            "role": "user",
            "content": f"""Synthesize this information into a clear business strategy, keeping in mind the original goals: "{original_input}"

                Clarity Analysis: {all_responses['clarity']}
                Niche Focus: {all_responses['niche']}
                Action Steps: {all_responses['action']}""",
        },
    ]
    response = make_openai_call(messages)
    print("✅ Final strategy complete")
    return response


def main():
    print("\n🤖 Welcome to AI Business Builder!\n")
    user_input = input(
        "Tell me about your business goals and how you'd like to make money with AI: "
    )

    clarity_response = clarity_agent(user_input)
    niche_response = niche_agent(clarity_response)
    action_response = action_agent(niche_response)

    all_responses = {
        "clarity": clarity_response,
        "niche": niche_response,
        "action": action_response,
    }

    final_strategy = business_strategist(all_responses)

    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ai_business_strategy_{timestamp}.txt"

    # Save to file
    with open(filename, "w") as f:
        f.write("=== AI Business Strategy ===\n\n")
        f.write(f"Original Input: {user_input}\n\n")
        f.write(f"Final Strategy:\n{final_strategy}\n")

    # Still print to console
    print("\n🎯 === Your AI Business Strategy ===\n")
    print(final_strategy)
    print(f"\n✨ Strategy saved to: {filename} ✨")


if __name__ == "__main__":
    main()
