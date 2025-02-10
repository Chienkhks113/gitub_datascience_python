
'''
Read the entire file story.txt and write a program to print out top 100 words occur most
 frequently and their corresponding appearance. You could ignore all
 punction characters such as comma , dot , semicolon

'''


import string
from collections import Counter

def find_top_words(filename, num_words=100):
  """Finds the top `num_words` most frequent words in a given file.

  Args:
    filename: The path to the text file.
    num_words: The number of top words to return.

  Returns:
    A list of tuples, where each tuple contains a word and its frequency.
  """

  with open(filename, 'r') as f:
    text = f.read()

  # Remove punctuation and convert to lowercase
  text = text.translate(str.maketrans('', '', string.punctuation)).lower()

  # Split the text into words
  words = text.split()

  # Count word frequencies
  word_counts = Counter(words)

  # Get the top `num_words` most frequent words
  top_words = word_counts.most_common(num_words)

  return top_words

if __name__ == '__main__':
  filename = 'story.txt'  # Replace with your file path
  top_words = find_top_words(filename)

  for word, count in top_words:
    print(f"{word}: {count}")