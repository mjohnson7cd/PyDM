from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os
from io import StringIO
import tensorflow as tf
import numpy as np


def extract_text_from_pdf(pdf_path):
    resource_manager = PDFResourceManager()
    string_io = StringIO()
    laparams = LAParams()

    with open(pdf_path, 'rb') as file:
        interpreter = PDFPageInterpreter(resource_manager,
                                         TextConverter(resource_manager, string_io, laparams=laparams))
        for page in PDFPage.get_pages(file):
            interpreter.process_page(page)

    text = string_io.getvalue()
    string_io.close()

    return text


def save_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)


# Usage:
print('Running....\n')
# pdf_path = './pdfs/modules/DDEX11_Defiance_in_Phlan.pdf'
# text = extract_text_from_pdf(pdf_path)
# cleaned_text = text.strip()
#
# file_path = './text/modules/DDEX11_Defiance_in_Phlan.txt'
# save_text_to_file(text, file_path)
#
# pdf_path = './pdfs/handbooks/DnD_5e_Players_Handbook.pdf'
# text = extract_text_from_pdf(pdf_path)
# cleaned_text = text.strip()
#
# file_path = './text/handbooks/DnD_5e_Players_Handbook.text'
# save_text_to_file(text, file_path)
#
# pdf_path = './pdfs/handbooks/Dungeon_Masters_Guide.pdf'
# text = extract_text_from_pdf(pdf_path)
# cleaned_text = text.strip()
#
# file_path = './text/handbooks/Dungeon_Masters_Guide.txt'
# save_text_to_file(text, file_path)
#
#
# # Directory containing the text files
# input_directory = './text/handbooks/'
#
# # New directory for the consolidated file
# output_directory = './text/consolidated'
#
# # Create the output directory if it doesn't exist
# os.makedirs(output_directory, exist_ok=True)
#
# # Consolidated file path
# output_file = os.path.join(output_directory, 'consolidated.txt')
#
# # Iterate over each file in the input directory
# file_contents = []
# for filename in os.listdir(input_directory):
#     file_path = os.path.join(input_directory, filename)
#     if os.path.isfile(file_path):
#         with open(file_path, 'r', encoding='utf-8') as file:
#             file_contents.append(file.read())
#
# # Consolidate the file contents into a single string
# consolidated_text = '\n'.join(file_contents)
#
# # Write the consolidated text to the output file
# with open(output_file, 'w', encoding='utf-8') as file:
#     file.write(consolidated_text)
#
# print("Consolidation complete. Consolidated file saved at:", output_file)

chunk_size = 1000
sequence_length = 50

file_path = './text/consolidated/consolidated.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()

chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

# Create input sequences
sequences = []
for chunk in chunks:
    for i in range(0, len(chunk) - sequence_length, 1):
        sequence = chunk[i:i + sequence_length]
        sequences.append(sequence)

# Tokenize the sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)

# Convert sequences to numpy array
sequences = np.array(sequences)

# Pad sequences to a uniform length
sequences = tf.keras.utils.pad_sequences(sequences, maxlen=sequence_length)

# Determine the number of unique tokens
num_unique_tokens = len(tokenizer.word_index) + 1

# Prepare the input and target sequences
input_sequences = sequences[:, :-1]
target_sequences = sequences[:, -1]

# Convert target_sequences to one-hot encoding
target_sequences = tf.keras.utils.to_categorical(target_sequences, num_classes=num_unique_tokens)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_unique_tokens, 128, input_length=sequence_length - 1),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(num_unique_tokens, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(input_sequences, target_sequences, epochs=50)

# Save the trained model
model.save('dnd_language_model')