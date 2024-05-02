import torch
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class Tokenizer:
    """
    Tokenizer class for converting protein sequences into numerical representations.

    This tokenizer handles a vocabulary of special tokens for classification (`<cls>`, `<eos>`),
    padding (`<pad>`), and unknown amino acids (`<unk>`), along with the 20 standard amino acids.

    Attributes:
        token_to_index (Dict[str, int]): Mapping from token to its index in the vocabulary.
        index_to_token (Dict[int, str]): Mapping from index to its corresponding token.
        vocab_size (int): Size of the vocabulary (number of tokens).
        pad_token_id (int): Index of the padding token (`<pad>`).
    """

    def __init__(self):
        """
        Initializes the tokenizer with a vocabulary of special tokens and amino acids.
        """

        # special tokens
        vocab = ["<cls>", "<pad>", "<eos>", "<unk>"]
        # 20 anonical amino acids
        vocab += list("ACDEFGHIKLMNPQRSTVWY")
        # mapping
        self.token_to_index = {tok: i for i, tok in enumerate(vocab)}
        self.index_to_token = {i: tok for i, tok in enumerate(vocab)}

    @property
    def vocab_size(self):
        """
        Returns the size of the vocabulary (number of tokens).
        """
        return len(self.token_to_index)

    @property
    def pad_token_id(self):
        """
        Returns the index of the padding token (`<pad>`).
        """
        return self.token_to_index["<pad>"]

    def __call__(
        self, seqs: list[str], padding: bool = True
    ) -> dict[str, list[list[int]]]:
        """
        Tokenizes a list of protein sequences and creates input representations with attention masks.

        Args:
            seqs (List[str]): List of protein sequences to tokenize.
            padding (bool, optional): Whether to pad sequences to a maximum length. Defaults to True.

        Returns:
            Dict[str, List[List[int]]]: A dictionary containing:
                - input_ids (List[List[int]]): List of token IDs for each sequence.
                - attention_mask (List[List[int]]): List of attention masks for each sequence.
        """

        input_ids = []
        attention_mask = []

        if padding:
            max_len = max(len(seq) for seq in seqs)

        for seq in seqs:
            # Preprocessing: strip whitespace, convert to uppercase
            seq = seq.strip().upper()

            # Add special tokens
            toks = ["<cls>"] + list(seq) + ["<eos>"]

            if padding:
                # Pad with '<pad>' tokens to reach max_len
                toks += ["<pad>"] * (max_len - len(seq))

            # Convert tokens to IDs (handling unknown amino acids)
            unk_id = self.token_to_index["<unk>"]
            input_ids.append([self.token_to_index.get(tok, unk_id) for tok in toks])

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask.append([1 if tok != "<pad>" else 0 for tok in toks])

        return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_data(data_loc: str) -> tuple[pd.DataFrame, dict]:
    """
    Loads and preprocesses data from a Parquet file.

    This function reads a Pandas DataFrame from the specified Parquet file location (`data_loc`),
    encodes categorical target labels using LabelEncoder, and returns the preprocessed DataFrame.

    Args:
        data_loc (str): Path to the Parquet file containing BCR data.

    Returns:
        tuple: A tuple containing:
            * pd.DataFrame: Preprocessed Pandas DataFrame containing:
                - Existing features from the original data.
                - "label" (int): Encoded representation of the target variable.
            * dict: A dictionary mapping encoded labels (integers) to their original class values.
    """
    df = pd.read_parquet(data_loc)

    # Create LabelEncoder for target variable
    le = LabelEncoder()

    # Encode target labels
    df["label"] = le.fit_transform(df["target"])

    # class mappinge
    classes = {i: c for i, c in enumerate(le.classes_)}

    return df, classes


class BCRDataset(Dataset):
    """
    BCRDataset class for loading and preparing B-cell receptor (BCR) dataset.

    This class inherits from `torch.utils.data.Dataset` and is used to load and prepare
    BCR data from a Pandas DataFrame for training or evaluation with a model.

    Attributes:
        df (pd.DataFrame): Pandas DataFrame containing BCR data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the BCRDataset object.

        Args:
            df (pd.DataFrame): Pandas DataFrame containing BCR data.
                - sequence (str): Amino acid sequence of the heavy chain.
                - label (int): Label associated with the BCR sample.
        """
        super().__init__()
        self.df = df

    def __len__(self) -> int:
        """
        Returns the length of the dataset (number of samples).

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, i) -> tuple[str, int]:
        """
        Retrieves a data point (sequence and label) at a specific index.

        Args:
            i (int): Index of the data point to retrieve.

        Returns:
            Tuple[str, int]: A tuple containing:
                - x (str): Amino acid sequence of the heavy chain.
                - y (int): Label associated with the BCR sample.
        """

        x = self.df.loc[i, "sequence"]
        y = self.df.loc[i, "label"]

        return x, y


def collate_fn(
    batch: list[tuple[str, int]],  # Tuples of (sequence, label)
    tokenizer: Tokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Collate function to prepare a batch of data for training or evaluation.

    This function takes a batch of data points, each containing a sequence (str) and its
    corresponding label (int), and processes them into a dictionary suitable for
    training or evaluation with a model. It performs the following steps:

    Args:
        batch (List[Tuple[str, int]]): Batch of data points, where each data point
            is a tuple containing a sequence (str) and its corresponding label (int ).
        tokenizer (Tokenizer): Tokenizer object used to convert sequences into numerical representations.
        device (torch.device): Device (CPU or GPU) where the tensors should be placed.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing the processed data:
            - input_ids (torch.Tensor): Tokenized sequence IDs (shape: batch_size, max_len).
            - attention_mask (torch.Tensor): Attention masks (shape: batch_size, max_len).
            - label (torch.Tensor): Labels (shape: batch_size).
    """

    # Unpack sequences and labels from the batch
    seqs, labels = zip(*batch)

    # Tokenize sequences and create attention masks with padding
    batch = tokenizer(seqs, padding=True)

    # convert to tensor
    for k in batch.keys():
        batch[k] = torch.tensor(batch[k], dtype=torch.long, device=device)

    # labels
    batch["label"] = torch.tensor(labels, dtype=torch.long, device=device)

    return batch
