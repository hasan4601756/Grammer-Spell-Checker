import nltk
nltk.download('words')
from nltk.corpus import words

# ---------------- TRIE NODE ----------------
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = None           # store full word at terminal nodes

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
        node.word = word           # save full word here

    def search(self, word):
        """Exact lookup — O(L)."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    # ------------------------------------------------------------------ #
    #  FUZZY SEARCH  (this replaces your old prefix-only get_suggestions) #
    # ------------------------------------------------------------------ #
    def get_suggestions(self, word, max_dist=2, k=5):
        results = []
        initial_row = list(range(len(word) + 1))   # [0, 1, 2, ..., len(word)]
        self._fuzzy_dfs(self.root, word, initial_row, max_dist, results)
        results.sort(key=lambda x: (x[1], x[0]))   # sort by distance, then alpha
        return results[:k]

    def _fuzzy_dfs(self, node, target, prev_row, max_dist, results):
        for ch, child in node.children.items():

            # ── compute next DP row for this character ──────────────────────
            curr_row = [prev_row[0] + 1]

            for col in range(1, len(target) + 1):
                insert_cost  = curr_row[col - 1] + 1
                delete_cost  = prev_row[col]     + 1
                replace_cost = prev_row[col - 1] + (0 if target[col - 1] == ch else 1)
                curr_row.append(min(insert_cost, delete_cost, replace_cost))

            # ── terminal node within threshold → record it ──────────────────
            if child.is_end and curr_row[-1] <= max_dist:
                results.append((child.word, curr_row[-1]))

            # ── pruning: only recurse if improvement is still possible ──────
            if min(curr_row) <= max_dist:
                self._fuzzy_dfs(child, target, curr_row, max_dist, results)


# ---------------- SPELL CHECKER ----------------
class SpellChecker:
    def __init__(self):
        self.word_list = words.words()

        self.trie = Trie()

        print("Building Trie... please wait")
        for w in self.word_list:
            self.trie.insert(w.lower())

        print("Dictionary loaded:", len(self.word_list))

    def is_correct(self, word):
        return self.trie.search(word.lower())   # trie handles it — no set needed

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
            print("Correct spelling")
        else:
            print("Incorrect spelling")
            suggestions = checker.get_suggestions(word)

            if not suggestions:
                print("No suggestions found (edit distance > 2)")
            else:
                print("Suggestions:")
                for w, dist in suggestions:
                    print(f"  - {w}  (distance: {dist})")
