import heapq
import nltk
nltk.download('words')
from nltk.corpus import words

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,      
                dp[i][j - 1] + 1,      
                dp[i - 1][j - 1] + cost 
            )

    return dp[m][n]


def get_top_k_suggestions(input_word, word_list, k=5):
    distances = []
    filtered_words = [w for w in word_list if abs(len(w) - len(input_word)) <= 2]

    for word in filtered_words:
        dist = levenshtein_distance(input_word.lower(), word.lower())
        distances.append((dist, word))

    top_k = heapq.nsmallest(k, distances)

    return [(word, dist) for dist, word in top_k]


if __name__ == '__main__':
    word_list = words.words()
    print("Dictionary size:", len(word_list))

    input_word = input("Enter a word: ")

    suggestions = get_top_k_suggestions(input_word, word_list)

    print("\nTop 5 suggestions:")
    for word, dist in suggestions:
        print(f"{word} (distance: {dist})")