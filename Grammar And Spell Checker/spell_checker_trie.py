import nltk
nltk.download('words')
from nltk.corpus import words


# ---------------- TRIE NODE ----------------
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


# ---------------- TRIE ----------------
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root

        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]

        node.is_end = True

    def _dfs(self, node, prefix, result):
        if len(result) >= 5:
            return

        if node.is_end:
            result.append(prefix)

        for ch, child in node.children.items():
            self._dfs(child, prefix + ch, result)

    def get_suggestions(self, prefix):
        node = self.root

        # go to prefix node
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]

        result = []
        self._dfs(node, prefix, result)
        return result


# ---------------- SPELL CHECKER ----------------
class SpellChecker:
    def __init__(self):
        self.word_list = words.words()
        self.word_set = set(w.lower() for w in self.word_list)

        self.trie = Trie()

        print("Building Trie... please wait")

        for w in self.word_list:
            self.trie.insert(w.lower())

        print("Dictionary loaded:", len(self.word_list))

    def is_correct(self, word):
        return word.lower() in self.word_set

    def get_suggestions(self, word):
        return self.trie.get_suggestions(word.lower())


# ---------------- MAIN ----------------
if __name__ == "__main__":
    checker = SpellChecker()

    while True:
        word = input("\nEnter word (or 'exit'): ")

        if word == "exit":
            break

        if checker.is_correct(word):
            print("✅ Correct spelling")
        else:
            print("❌ Incorrect spelling")
            print("Suggestions:")

            suggestions = checker.get_suggestions(word)

            if not suggestions:
                print("No prefix-based suggestions found")
            else:
                for s in suggestions:
                    print(" -", s)