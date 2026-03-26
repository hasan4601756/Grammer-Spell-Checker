import nltk
import numpy as np
from nltk.corpus import treebank
from collections import defaultdict, Counter


class HMMPOSTagger:
    def __init__(self):
        self.tags = []
        self.vocab = set()

        self.initial_prob = {}
        self.transition_prob = defaultdict(dict)
        self.emission_prob = defaultdict(dict)

    def train(self, train_data):
        tag_counts = Counter()
        word_tag_counts = defaultdict(Counter)
        transition_counts = defaultdict(Counter)
        initial_counts = Counter()

        for sentence in train_data:
            prev_tag = '<s>'

            for i, (word, tag) in enumerate(sentence):
                self.vocab.add(word)

                tag_counts[tag] += 1
                word_tag_counts[tag][word] += 1

                if i == 0:
                    initial_counts[tag] += 1

                transition_counts[prev_tag][tag] += 1
                prev_tag = tag

        self.tags = list(tag_counts.keys())
        num_tags = len(self.tags)
        vocab_size = len(self.vocab)
        total_sentences = len(train_data)

        # -----------------------------
        # Laplace Smoothed Initial Prob
        # -----------------------------
        for tag in self.tags:
            self.initial_prob[tag] = (
                initial_counts[tag] + 1
            ) / (total_sentences + num_tags)

        # -----------------------------
        # Laplace Smoothed Transition
        # -----------------------------
        for prev_tag in ['<s>'] + self.tags:
            total = sum(transition_counts[prev_tag].values())

            for tag in self.tags:
                self.transition_prob[prev_tag][tag] = (
                    transition_counts[prev_tag][tag] + 1
                ) / (total + num_tags)

        # -----------------------------
        # Laplace Smoothed Emission
        # -----------------------------
        for tag in self.tags:
            total = tag_counts[tag]

            for word in self.vocab:
                self.emission_prob[tag][word] = (
                    word_tag_counts[tag][word] + 1
                ) / (total + vocab_size)

    def get_emission(self, tag, word):
        # Handle unseen words (important)
        vocab_size = len(self.vocab)
        return 1 / (sum(self.emission_prob[tag].values()) * vocab_size)

    def viterbi(self, sentence):
        n = len(sentence)
        m = len(self.tags)

        viterbi = np.zeros((m, n))
        backpointer = np.zeros((m, n), dtype=int)

        # Initialization
        for i, tag in enumerate(self.tags):
            emission = self.emission_prob[tag].get(
                sentence[0], self.get_emission(tag, sentence[0])
            )
            viterbi[i, 0] = self.initial_prob[tag] * emission

        # Recursion
        for t in range(1, n):
            for i, tag in enumerate(self.tags):
                max_prob = -1
                max_state = 0

                emission = self.emission_prob[tag].get(
                    sentence[t], self.get_emission(tag, sentence[t])
                )

                for j, prev_tag in enumerate(self.tags):
                    prob = (
                        viterbi[j, t - 1] *
                        self.transition_prob[prev_tag][tag] *
                        emission
                    )

                    if prob > max_prob:
                        max_prob = prob
                        max_state = j

                viterbi[i, t] = max_prob
                backpointer[i, t] = max_state

        # Backtracking
        best_path = []
        last_state = np.argmax(viterbi[:, n - 1])
        best_path.append(self.tags[last_state])

        for t in range(n - 1, 0, -1):
            last_state = backpointer[last_state, t]
            best_path.insert(0, self.tags[last_state])

        return best_path

    def tag_sentence(self, sentence):
        words = sentence.split()
        tags = self.viterbi(words)
        return list(zip(words, tags))

    def evaluate(self, test_data):
        correct = 0
        total = 0

        for sentence in test_data:
            words = [w for w, t in sentence]
            true_tags = [t for w, t in sentence]

            pred_tags = self.viterbi(words)

            correct += sum(p == t for p, t in zip(pred_tags, true_tags))
            total += len(true_tags)

        return correct / total


if __name__ == "__main__":
    try:
        nltk.data.find('treebank')
    except LookupError:
        nltk.download('treebank')
    try:
        nltk.data.find('universal_tagset')
    except LookupError:
        nltk.download('universal_tagset')
        
    data = treebank.tagged_sents(tagset='universal')

    train_data = data[:3000]
    test_data = data[3000:]

    tagger = HMMPOSTagger()
    tagger.train(train_data)

    acc = tagger.evaluate(test_data)
    print(f"Accuracy: {acc:.4f}")

    sentence = "let us go to the college"
    print(tagger.tag_sentence(sentence))