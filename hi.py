import pandas as pd
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Step 1: Read the AEP CSV data
data = pd.read_csv('aepdata.csv')

# Isolate the 5th column (assuming we want to clean that for now)
comments_column = data.iloc[:, 4]

# Step 2: Define a set of stopwords (you can customize this)
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
    'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
    'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
    'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', 'now'
])

# Step 3: Function to clean comments manually
def clean_comment(comment):
    if not isinstance(comment, str):
        return ""
    
    # Convert comment to lowercase
    comment = comment.lower()

    # Remove punctuation (keeping numbers)
    comment = re.sub(r'[^\w\s]', '', comment)

    # Tokenize
    words = comment.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Reconstruct cleaned comment
    cleaned_comment = ' '.join(words)

    return cleaned_comment

# Apply the cleaning function to the comments column
cleaned_comments = comments_column.apply(clean_comment)

# Load the pre-trained LLM and tokenizer
model_name = "gpt2"  # Specify the model you want to use
tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')

# Set the pad token to the eos token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)

# Define your high-risk scenarios
high_energy_scenarios = """
1. A person fell from a height greater than 4 feet.
2. A vehicle departed from its intended path and was within 6 feet of an employee.
3. Exposure to temperatures greater than 150 degrees Fahrenheit.
4. An explosion occurred or a highly combustible material was involved.
5. There was a release of steam or exposure to unsupported soil in a trench over 5 feet deep.
6. Electrical exposure of 50 volts or greater.
7. Exposure to toxic chemicals or radiation.
8. A moving vehicle caused potential harm to workers on foot.
9. Heavy equipment was in motion near employees.
10. Incidents involving machinery that could cause serious injury.
11. Any mention of burns or scalds.
12. Near misses that could have led to serious injury.
13. Comments indicating a hazardous working condition.
"""

# Function to classify each comment in batches
def classify_comments_in_batches(comments, batch_size=8):
    classifications = []
    
    # Process comments in batches
    for i in range(0, len(comments), batch_size):
        batch_comments = comments[i:i + batch_size]
        
        # Prepare the prompts for each comment in the batch
        prompts = [
            f"Classify the following comment as 'high energy' or 'low energy' based on the scenarios:\n\n"
            f"{high_energy_scenarios}\n\nComment: {comment}\nClassification:"
            for comment in batch_comments
        ]

        # Tokenize the input prompts
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Set max_new_tokens to control only the output length
        max_new_tokens = 20  # Adjust this as needed based on expected output length

        # Generate output from the model, setting pad_token_id
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the output for each comment
        for output in outputs:
            classification = tokenizer.decode(output, skip_special_tokens=True)
            classifications.append(classification.split("Classification:")[-1].strip())

    return classifications

# Apply the classification function to all cleaned comments in batches
data['Classification'] = classify_comments_in_batches(cleaned_comments.tolist(), batch_size=8)

# Save the results to a new CSV file
data.to_csv('classified_aepdata.csv', index=False)

# Print the results
print(data[['Comments', 'Classification']])
