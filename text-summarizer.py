# %%
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import tkinter as tk
from tkinter import scrolledtext


# This function reads the input text and splits it into sentences.
def read_article(text):
    article = text.split(". ")

    sentences = []
    for sentence in article:
        # Tokenize the sentence by replacing non-alphabetic characters and splitting by spaces
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))

    # Remove empty sentences
    sentences = [sentence for sentence in sentences if sentence]

    return sentences


# This function calculates the similarity between two sentences using cosine distance.
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    # Convert both sentences to lowercase
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    # Create a list of all unique words in both sentences
    all_words = list(set(sent1 + sent2))

    # Initialize vectors to represent the word counts in the sentences
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # Build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        # Increment the count of the word in the vector
        vector1[all_words.index(w)] += 1

    # Build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        # Increment the count of the word in the vector
        vector2[all_words.index(w)] += 1

    # Calculate the similarity as 1 minus the cosine distance
    return 1 - cosine_distance(vector1, vector2)


# This function builds a similarity matrix for sentences based on their pairwise similarity.
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix with dimensions equal to the number of sentences
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    # Iterate through all sentence pairs
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # Skip if both sentences are the same
                continue

            # Calculate the similarity between the two sentences using the sentence_similarity function
            similarity_matrix[idx1][idx2] = sentence_similarity(
                sentences[idx1], sentences[idx2], stop_words
            )

    return similarity_matrix


# This function generates a text summary based on the input file and the top 'n' sentences.
def generate_summary(sentences, top_n=5):

    nltk.download("stopwords")
    stop_words = stopwords.words("english")

    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)

    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True
    )

    # Initialize a list to store the summarized sentences
    summarize_text = []
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentences[i][1]))

    return ". ".join(summarize_text) + "."


# Driver code
def main():
    # Create the main tkinter window
    window = tk.Tk()
    window.title("Text Summarization Tool")
    window.geometry("800x600")  # Set the window size

    def generate_button_click():
        text_input = text_input_area.get(
            "1.0", "end-1c"
        )  # Get the text from the input area
        # Call the modified read_article function with the user's input text
        sentences = read_article(text_input)
        # Generate the summary
        summary = generate_summary(sentences, top_n=5)
        # Display the summary in the summary area
        summary_area.config(state="normal")  # Enable writing to the summary area
        summary_area.delete(1.0, tk.END)  # Clear previous content
        summary_area.insert(tk.END, summary)
        summary_area.config(state="disabled")  # Set the summary area back to read-only

    # Create a label for the input area
    input_label = tk.Label(window, text="Paste your article below:")
    input_label.pack(pady=5, padx=10)

    # Create a text input area
    text_input_area = scrolledtext.ScrolledText(window, width=70, height=15)
    text_input_area.pack(pady=5, padx=10)

    # Create a button to generate the summary
    generate_button = tk.Button(
        window, text="Generate Summary", command=generate_button_click
    )
    generate_button.pack(pady=5, padx=10)

    # Create a label for the summary area
    summary_label = tk.Label(window, text="Summary:")
    summary_label.pack(pady=5, padx=10)

    # Create a summary area
    summary_area = scrolledtext.ScrolledText(
        window, width=70, height=10, state="disabled"
    )
    summary_area.pack(pady=5, padx=10)

    # Start the tkinter main loop
    window.mainloop()


if __name__ == "__main__":
    main()
