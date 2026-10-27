with open("C:/Finetune-AI-article/AI_domain_text_dataset.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Dataset loaded successfully!")
print()
print("Number of characters:", len(text))
print()
print("First 500 characters:")
print(text[:500])