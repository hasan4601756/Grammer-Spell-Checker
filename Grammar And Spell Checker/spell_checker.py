import heapq
import nltk
nltk.download('words')
from nltk.corpus import words


class SpellChecker:
    def __init__(self):
        self.word_list = words.words()
        self.word_set = set(w.lower() for w in self.word_list)
        print("Dictionary size:", len(self.word_list))

    def levenshtein_distance(self, s1, s2):
        m, n = len(s1), len(s2)

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1

                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )

        return dp[m][n]

    def is_correct(self, input_word):
        return input_word.lower() in self.word_set

    def get_suggestions(self, input_word, k=5):
        filtered_words = [
            w for w in self.word_list
            if abs(len(w) - len(input_word)) <= 2
        ]

        distances = []
        for word in filtered_words:
            dist = self.levenshtein_distance(input_word.lower(), word.lower())
            distances.append((dist, word))

        top_k = heapq.nsmallest(k, distances)
        return [(word, dist) for dist, word in top_k]


if __name__ == '__main__':
    checker = SpellChecker()

    input_word = input("Enter a word: ")

    if checker.is_correct(input_word):
        print("\nThe spelling is correct.")
    else:
        print("\nThe spelling is incorrect. Suggestions:")

        suggestions = checker.get_suggestions(input_word)

        print("\nTop 5 suggestions:")
        for word, dist in suggestions:
            print(f"{word} (distance: {dist})")