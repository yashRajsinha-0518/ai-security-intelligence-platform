from llm_explainer import generate_explanation

print("Normal traffic explanation:\n")
print(generate_explanation(0))

print("\n\nAttack explanation:\n")
print(generate_explanation(1))