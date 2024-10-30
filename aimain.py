from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client with API key from .env
client = OpenAI()

def clarity_agent(user_input: str) -> str:
    print("\n1️⃣ Clarity Agent analyzing your goals...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Clarity Agent helping entrepreneurs understand how to monetize with AI. Be direct and practical."},
            {"role": "user", "content": f"Help me understand how to make money with AI based on this context: {user_input}"}
        ]
    )
    print("✅ Clarity analysis complete")
    return response.choices[0].message.content

def niche_agent(clarity_response: str) -> str:
    print("\n2️⃣ Niche Agent identifying target market...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Niche Agent helping identify specific market opportunities and ideal customer profiles."},
            {"role": "user", "content": f"Based on this context, identify the most profitable niche and ideal customer avatar: {clarity_response}"}
        ]
    )
    print("✅ Niche analysis complete")
    return response.choices[0].message.content

def action_agent(niche_response: str) -> str:
    print("\n3️⃣ Action Agent creating specific steps...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an Action Agent providing specific, actionable steps for finding and acquiring customers."},
            {"role": "user", "content": f"Provide specific action steps for this niche and avatar: {niche_response}"}
        ]
    )
    print("✅ Action plan complete")
    return response.choices[0].message.content
def business_strategist(all_responses: dict, original_input: str) -> str:
    print("\n4️⃣ Business Strategist synthesizing final plan...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Business Strategist creating clear, actionable business plans."},
            {"role": "user", "content": f"""Synthesize this information into a clear business strategy, keeping in mind the original goals: "{original_input}"

                Clarity Analysis: {all_responses['clarity']}
                Niche Focus: {all_responses['niche']}
                Action Steps: {all_responses['action']}"""}
        ]
    )
    print("✅ Final strategy complete")
    return response.choices[0].message.content

def main():
    print("\n🤖 Welcome to AI Business Builder!\n")
    user_input = input("Tell me about your business goals and how you'd like to make money with AI: ")
    
    clarity_response = clarity_agent(user_input)
    print("clarity_response ",clarity_response)

    niche_response = niche_agent(clarity_response)
    print("niche_response ",niche_response)

    action_response = action_agent(niche_response)
    print("action_response ",action_response)

    all_responses = {
        "clarity": clarity_response,
        "niche": niche_response,
        "action": action_response
    }
    
    final_strategy = business_strategist(all_responses, user_input)
    
    print("\n🎯 === Your AI Business Strategy ===\n")
    print(final_strategy)
    print("\n✨ Good luck with your AI business journey! ✨")

if __name__ == "__main__":
    main()
