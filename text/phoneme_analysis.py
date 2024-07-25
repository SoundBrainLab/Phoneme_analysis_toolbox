import os
import re
import copy
import tqdm
import pickle
import cmudict

class Corpus:
    def __init__(self, directory, ignore_stress = True, cache = True, load_from_cache = True):
        """
        Initialize the Corpus instance. The Corpus is a collection of text passages.

        Args:
            directory (str): Directory containing text files.
            ignore_stress (bool, optional): Whether the phoneme transcriptions should ignore stress level (0, 1, 2) for vowels. Defaults to True.
            cache (bool, optional): Whether to cache the passage data. Defaults to True.
            load_from_cache (bool, optional): Whether to load passage data from cache if available. Defaults to True.
        """
        self.directory = directory
        self.ignore_stress = ignore_stress
        self.cache = cache
        self.load_from_cache = load_from_cache
        self.cache_dir = os.path.join(directory, 'cached')
        self.passages = self.__load_passages()
        self.oovs = self.__find_oovs()

    def __load_passages(self):
        """
        Load passages from the specified directory and create Passage instances.

        Returns:
            list: List of Passage instances.
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        passages = []
        txt_files = [f for f in os.listdir(self.directory) if f.endswith('.txt')]
        new_files = []

        for filename in txt_files:
            filepath = os.path.join(self.directory, filename)
            cache_filepath = os.path.join(self.cache_dir, filename.replace('.txt', '.pkl'))

            if self.load_from_cache and os.path.exists(cache_filepath):
                with open(cache_filepath, 'rb') as cache_file:
                    passage = pickle.load(cache_file)
                print(f'Loading passage data for {filename} from cached...')
                passages.append(passage)
            else:
                new_files.append(filepath)

        if new_files:
            for filepath in tqdm.tqdm(new_files, desc = 'Loading and analyzing passage'):
                passage = Passage(filepath, ignore_stress = self.ignore_stress)
                passages.append(passage)
                if self.cache:
                    cache_filepath = os.path.join(self.cache_dir, os.path.basename(filepath).replace('.txt', '.pkl'))
                    with open(cache_filepath, 'wb') as cache_file:
                        pickle.dump(passage, cache_file)

        return passages

    def __find_oovs(self):
        """
        Find out-of-vocabulary words in all passages.
        """
        all_oovs = [p.get_oovs() for p in self.passages]
        all_oovs = list(set([oov for oovlist in all_oovs for oov in oovlist]))
        return all_oovs

    def get_oovs(self):
        """
        Return out-of-vocabulary words.

        Returns:
            list: List of OOV words.
        """
        return self.oovs[:]

    def export_oovs(self, outpath = None):
        """
        Export the OOV word list to a txt file.

        Args:
            outpath (str, optional): Path to which the list will be saved. If None, save to the passage folder with the name oovs.txt. Default to None.
        """
        if len(self.oovs) > 0:
            if outpath is None:
                outpath = self.directory + '/oovs.txt'
            with open(outpath, 'w') as f:
                f.write('\n'.join(self.oovs[:]))
        else:
            print('No OOV words in the corpus.')
        
# Cache for phoneme lookups
phoneme_cache = {}

class Passage:
    def __init__(self, text, name = None, ignore_stress = True):
        """
        Initialize the Passage instance.

        Args:
            text (str): A string of text or a path to a text file.
            name (str, optional): Name of the passage. Defaults to None. If text argument is a path, set to the text document filename.
            ignore_stress (bool, optional): Whether the phoneme transcriptions should ignore stress level (0, 1, 2) for vowels. Deafults to True.
        """
        self.name = name
        self.oovs = []
        self.ignore_stress = ignore_stress
        
        self.__read_input_text(text)
        self.__clean_text()
        self.__to_phonemes()
        
        self.phoneme_count, self.phoneme_count_by_pos = self.__get_phoneme_stats()

        self.__check_counts()

        self.n_phonemes = sum(self.phoneme_count.values())
        self.n_unique_phonemes = len(self.phoneme_count.keys())
        self.n_words = len(self.phon_trans)

    def __read_input_text(self, input_data):
        """
        Read the input text data or data path.

        Args:
            input_data (str): A string of text or a path to a text file.
        """
        if os.path.isfile(input_data):
            with open(input_data, 'r', encoding='utf-8') as file:
                text = file.read()
            if self.name is None:
                self.name = os.path.basename(input_data)
        else:
            text = input_data
        self.text = text
        
    def __clean_text(self):
        """
        Clean the input text by converting to lowercase and removing punctuations except for apostrophes, newline characters, trailing spaces, and extra spaces.
        """
        self.text = self.text.lower().replace('\n', ' ')
        self.text = self.text.replace('’', "'")
        self.text = re.sub(r"[^\w\s']", '', self.text)
        self.text = re.sub(r'\s+', ' ', self.text).strip()

    def __to_phonemes(self):
        """
        Convert the cleaned text into phoneme transcriptions using the CMU Pronunciation Dictionary.
        https://pypi.org/project/cmudict/
        """
        words = self.text.split(' ')
        phon_transcript = []
        for word in words:
            if word in phoneme_cache:
                phonemes = phoneme_cache[word]
            else:
                phonemes = cmudict.dict()[word]
                if phonemes:
                    phonemes = phonemes[0]  # Take the first pronunciation variant
                    phoneme_cache[word] = phonemes
                else:
                    self.oovs.append(word)
                    phoneme_cache[word] = []  # Cache the OOV word with an empty list
            phon_transcript.append((word, phonemes))
        self.phon_trans = phon_transcript
        
    def __get_phoneme_stats(self):
        """
        Calculate the frequency of each unique phoneme in the phoneme transcriptions and its frequency in word-initial, -medial, and -final positions.

        Returns:
            dict: Dictionary with total count for each phoneme and dictionary with count by word position
        """
        phoneme_count = {}
        phoneme_count_by_pos = {} # Record counts for word-initial, medial, final positions
        for _, phonemes in self.phon_trans:
            n_ph = len(phonemes)
            for i, p in enumerate(phonemes):
                if self.ignore_stress:
                    p = re.sub(r'\d$', '', p) # Remove stress marker
                if p in phoneme_count:
                    phoneme_count[p] += 1
                else:
                    phoneme_count[p] = 1

                if p not in phoneme_count_by_pos:
                    phoneme_count_by_pos[p] = [0, 0, 0]

                if n_ph >= 2:
                    if i > 0:
                        if i == (n_ph-1):
                            pos_idx = 2 # Word-final
                        else:
                            pos_idx = 1 # Word-medial
                    else:
                        pos_idx = 0 # Word-initial
                        
                    phoneme_count_by_pos[p][pos_idx] += 1
        phoneme_count = dict(sorted(phoneme_count.items()))
        phoneme_count_by_pos = dict(sorted(phoneme_count_by_pos.items()))
        return phoneme_count, phoneme_count_by_pos

    def __check_counts(self):
        """
        Check if the phoneme counts are valid.
        """
        checks = [self.phoneme_count[k] >=  sum(self.phoneme_count_by_pos[k]) for k in self.phoneme_count]
        if not all(checks):
            raise ValueError('Invalid phoneme counts. For each phoneme, the count summed over all positions should not be greater than its total count.')
        
    def rename(self, new_name):
        """
        Rename the current instance of Passage.

        Args:
            new_name (str): New name.
        """
        self.name = new_name

    # Getter functions
    def get_name(self):
        """
        Returns name of the passage instance.

        Returns:
            str: Name of passage.
        """
        return self.name

    def get_oovs(self):
        """
        Returns out-of-vocabulary words.

        Returns:
            list: List of OOV words.
        """
        return self.oovs[:]

    def get_n_phonemes(self):
        """
        Returns total phoneme instance count.

        Returns:
            int: Total phoneme instance count.
        """
        return self.n_phonemes

    def get_n_unique_phonemes(self):
        """
        Returns no. unique phonemes.

        Returns:
            int: Unique phoneme count.
        """
        return self.n_unique_phonemes

    def get_n_words(self):
        """
        Returns total word token count.

        Returns:
            int: Total word token count.
        """
        return self.n_words
    
    def get_phoneme_freqs(self, by_position = False):
        """
        Returns the total frequency/count for each phoneme or frequency by position.

        Args:
            by_position (bool): Whether to return frequency by position or not

        Returns:
            dict: Frequency for each phoneme 
        """
        if by_position:
            return copy.deepcopy(self.phoneme_count_by_pos)
        else:
            return copy.deepcopy(self.phoneme_count)
