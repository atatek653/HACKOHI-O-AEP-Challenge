import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
import re

# Load data
file_path = '/Users/jaydenvasquez/Downloads/VSCode/hackathon/Backend/aepdata.csv'
data = pd.read_csv(file_path)

# Stop words set
stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
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
    'should', 'now', 'would', 'could', 'might'])  # Your existing stop words

high_energy_setences = ["Suspended load.","Mobile Equipment/Traffic with Workers on Foot", "Heavy rotating equipment",
"Steam", "Explosion", "Electrical contact with source", "Fall from elevation"
"Motor vehicle incident (occupant)", "High temperature", "Fire with sustained fuel source"
"Excavation or trench", "Arc flash", "High dose of toxic chemical or radiation", "I had to advise the contractor to make sure he was utilizing a harness at anytime he was elevated above 4 feet,",
"Employee did not have glasses on while working in control cabinet. Brought to employee's attention and he put them on.,", 
"A worker was working from a scaffold that had a I - beam acting as the top rail of the scaffold and the employee was climbing out on the i - beam and reaching out exposting himself to a fall hazard.", 
"While chipping a limb became jammed. [NAME] disengaged the chipper and was working to remove the limb. [NAME] member using a handsaw to remove limb. To get a better angle the trimmer got on his knees on the chipper tailboard. limb was removed and trimmer stepped away from the chipper. I coached the foreman on this action told him of the risk involved in this practice. It was reported to the GF. Trimmer was reprimanded and given three days of without pay."]
low_energy_setences = ["employee was not wearing overshoes, spoke to the employee and got agreement that overshoes were needed",
"No chipper key available and chipper is required for work. Bucket broke down earlier in the day and key went with truck when towed. No key in spare truck",
"NAME] is going to be working uphill this morning and nothing mentioned about hidden hazards in tall grass.",
"The truck was pulled into location instead of backing in. The employee had to work over the front hood and windshield while aloft.",
"Encouraged foreman to further discuss and document hazards associated with rotating equipment ( task was installing 18 guy anchors at Str. 2",
 "Apprentice had to be told to put their FR shirt on before putting on their gloves and sleeves to set a pole."]

# Function to clean comments
def clean_comment(comment):
    if not isinstance(comment, str):
        return ""
    
    comment = comment.lower()
    comment = re.sub(r'[^\w\s]', '', comment)

    words = comment.split()
    words = [word for word in words if word not in stop_words]

    return ' '.join(words)

# Clean comments column
def clean_comments(data):
    comments_column = data.iloc[:, 4]  # Adjust if necessary
    cleaned_comments = comments_column.apply(clean_comment)
    return cleaned_comments

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Function to get embeddings
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding="longest", truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def calculate_centroid(embeddings):
    return np.mean(embeddings, axis=0)
# Calculate embeddings and centroids
high_energy_embeddings = [get_embedding(sentence) for sentence in high_energy_setences]
low_energy_embeddings = [get_embedding(sentence) for sentence in low_energy_setences]

high_centroid = calculate_centroid(high_energy_embeddings)
low_centroid = calculate_centroid(low_energy_embeddings)

# Prepare for JSON data storage
json_data = []

# Process each cleaned comment and classify
for index, sentence in enumerate(clean_comments(data)):
    new_embedding = get_embedding(sentence)
    print(sentence)

    distance_to_high = np.linalg.norm(new_embedding - high_centroid)
    distance_to_low = np.linalg.norm(new_embedding - low_centroid)
       # Classify based on distance
    if (distance_to_low + distance_to_high) / 2 >= 6:
        classification = "High Energy" 
        print(classification) 
    else:
        classification = "Low Energy"
        print(classification)

    # Create entry for JSON
    entry = {
        "index": index + 1,
        "date": str(data.iloc[index, 1]),
        "comment": sentence,
        "classification": classification
    }
    
    json_data.append(entry)

    if index > 200:
        break

# Count occurrences
counts = pd.Series([d['classification'] for d in json_data]).value_counts()

# Bar Plot for High Energy vs Low Energy Comments
plt.figure(figsize=(8, 5))
plt.bar(counts.index, counts.values, color=['blue', 'orange'])
plt.title('Count of High Energy vs Low Energy Comments')
plt.xlabel('Energy Level')
plt.ylabel('Count')
matplotlib_plot_path = '/Users/jaydenvasquez/Downloads/VSCode/hackathon/Backend/matplotlib_plot.png'
plt.savefig(matplotlib_plot_path)


# Combine embeddings for PCA
high_energy_embeddings = np.vstack([get_embedding(entry['comment']) for entry in json_data if entry['classification'] == "High Energy"])
low_energy_embeddings = np.vstack([get_embedding(entry['comment']) for entry in json_data if entry['classification'] == "Low Energy"])

# Combine all embeddings
all_embeddings = np.vstack([high_energy_embeddings, low_energy_embeddings])
labels = ['High Energy'] * len(high_energy_embeddings) + ['Low Energy'] * len(low_energy_embeddings)

# PCA to reduce dimensions
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(all_embeddings)

# Scatter Plot of Embeddings
plt.figure(figsize=(10, 8))
for label in set(labels):
    plt.scatter(reduced_embeddings[np.array(labels) == label, 0],
                reduced_embeddings[np.array(labels) == label, 1],
                label=label)
embeddings_plot_path = '/Users/jaydenvasquez/Downloads/VSCode/hackathon/Backend/embeddings_plot.png'
plt.title('PCA of Comment Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.savefig(embeddings_plot_path)


# Add plot paths to each JSON entry
for entry in json_data:
    entry["plots"] = {
        "matplotlibPlot": matplotlib_plot_path,
        "embeddingsPlot": embeddings_plot_path
    }

# Write to JSON file
with open('/Users/jaydenvasquez/Downloads/VSCode/hackathon/Backend/data.json', 'w') as json_file:           
    json.dump(json_data, json_file, indent=4)