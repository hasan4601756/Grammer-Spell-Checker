from tokenizer import Preprocessor
from spell_checker import SpellChecker as DistanceSpellChecker
from spell_checker_trie import SpellChecker as TrieSpellChecker
from pos_tagger import HMMPOSTagger

import nltk
from nltk.corpus import treebank


def run_preprocessing():
    text = input("\nEnter text: ")
    pre = Preprocessor(text)

    clean = pre.preprocess()
    words = pre.tokenize()
    sentences = pre.sentence_tokenize()

    print("\n--- Preprocessing Output ---")
    print("Clean Text:", clean)
    print("Word Tokens:", words)
    print("Sentence Tokens:", sentences)


def run_trie_spellchecker():
    checker = TrieSpellChecker()

    while True:
        word = input("\nEnter word (or 'exit'): ")
        if word == "exit":
            break

        if checker.is_correct(word):
            print("✅ Correct spelling")
        else:
            print("❌ Incorrect spelling")
            suggestions = checker.get_suggestions(word)

            if suggestions:
                print("Suggestions:", suggestions)
            else:
                print("No suggestions found")


def run_distance_spellchecker():
    checker = DistanceSpellChecker()

    while True:
        word = input("\nEnter word (or 'exit'): ")
        if word == "exit":
            break

        if checker.is_correct(word):
            print("✅ Correct spelling")
        else:
            print("❌ Incorrect spelling")
            suggestions = checker.get_suggestions(word)
            
            print("\nTop suggestions:")
            for w, d in suggestions:
                print(f"{w} (distance: {d})")


def run_pos_tagger():
    print("\nTraining HMM POS Tagger...")

    try:
        nltk.data.find('treebank')
    except:
        nltk.download('treebank')

    try:
        nltk.data.find('universal_tagset')
    except:
        nltk.download('universal_tagset')

    data = treebank.tagged_sents(tagset='universal')

    train_data = data[:3000]
    test_data = data[3000:]

    tagger = HMMPOSTagger()
    tagger.train(train_data)

    acc = tagger.evaluate(test_data)
    print(f"\nModel Accuracy: {acc:.4f}")

    while True:
        sentence = input("\nEnter sentence (or 'exit'): ")
        if sentence == "exit":
            break

        tagged = tagger.tag_sentence(sentence)
        print("Tagged Sentence:")
        print(tagged)


def main():
    while True:
        print("\n=========== NLP TOOLKIT ===========")
        print("1. Text Preprocessing & Tokenization")
        print("2. Spell Checker (Trie - Prefix Based)")
        print("3. Spell Checker (Edit Distance)")
        print("4. POS Tagger (HMM + Viterbi)")
        print("5. Exit")

        choice = input("Select option: ")

        if choice == "1":
            run_preprocessing()
        elif choice == "2":
            run_trie_spellchecker()
        elif choice == "3":
            run_distance_spellchecker()
        elif choice == "4":
            run_pos_tagger()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()