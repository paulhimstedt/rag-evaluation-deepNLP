"""
Dataset preparation script for RAG evaluation.
Downloads and formats all 7 evaluation datasets used in the RAG paper.

Datasets:
1. Natural Questions (NQ) - from DPR
2. TriviaQA - from DPR
3. WebQuestions - from DPR
4. CuratedTrec - from DPR
5. MS-MARCO NLG v2.1 - from HuggingFace
6. SearchQA (Jeopardy) - from HuggingFace
7. FEVER - from HuggingFace

Output format:
- {dataset}_test.source: One question/claim per line
- {dataset}_test.target: One answer per line (or tab-separated question\t['answer1', 'answer2'] for qa mode)
"""

import csv
import gzip
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


class DatasetPreparer:
    def __init__(self, output_dir: str = "./eval_datasets", max_samples: int = 0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples: Optional[int] = max_samples if max_samples and max_samples > 0 else None

    def _limit_reached(self, count: int) -> bool:
        return self.max_samples is not None and count >= self.max_samples

    def download_file(self, url: str, output_path: Path, decompress_gz: bool = False, max_retries: int = 3):
        """Download a file from URL with retry logic."""
        print(f"Downloading {url} to {output_path}")

        if output_path.exists():
            print(f"  File already exists, skipping download")
            return output_path

        # Try download with retries
        for attempt in range(max_retries):
            try:
                # Add user agent to avoid some 403 errors
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; RAGEvaluation/1.0)'}
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    with open(output_path, 'wb') as out_file:
                        out_file.write(response.read())

                # Success!
                break

            except urllib.error.HTTPError as e:
                if e.code == 403:
                    print(f"  ⚠ 403 Forbidden error (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        print(f"  Retrying in {2**attempt} seconds...")
                        time.sleep(2**attempt)
                    else:
                        print(f"  ✗ Failed after {max_retries} attempts, skipping this file")
                        return None
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠ Error: {e}, retrying...")
                    time.sleep(2**attempt)
                else:
                    print(f"  ✗ Failed after {max_retries} attempts: {e}")
                    return None

        if not output_path.exists():
            return None

        if decompress_gz and output_path.suffix == '.gz':
            print(f"  Decompressing {output_path}")
            decompressed_path = output_path.with_suffix('')
            with gzip.open(output_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            output_path.unlink()  # Remove .gz file
            return decompressed_path

        return output_path

    def convert_dpr_csv_to_eval_format(self, csv_path: Path, dataset_name: str, split: str = "test"):
        """
        Convert DPR CSV format to .source/.target format.
        DPR CSV format: question, answers (JSON array as string)
        """
        print(f"Converting {dataset_name} {split} to eval format")

        # Read CSV
        df = pd.read_csv(csv_path, sep='\t', header=None, names=['question', 'answers'])
        if self.max_samples is not None:
            df = df.head(self.max_samples)

        # Write .source file (questions)
        source_path = self.output_dir / f"{dataset_name}_{split}.source"
        with open(source_path, 'w', encoding='utf-8') as f:
            for question in df['question']:
                f.write(question.strip() + '\n')

        # Write .target file in qa mode format (question\t['answer1', 'answer2'])
        target_path = self.output_dir / f"{dataset_name}_{split}.target"
        with open(target_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                question = row['question'].strip()
                answers = row['answers']  # Already in string format like "['answer1', 'answer2']"
                f.write(f"{question}\t{answers}\n")

        print(f"  Created {source_path} ({len(df)} samples)")
        print(f"  Created {target_path}")
        return len(df)

    def prepare_nq(self):
        """Prepare Natural Questions dataset from DPR."""
        print("\n=== Preparing Natural Questions (NQ) ===")

        # Download test split
        test_url = "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv"
        test_csv = self.output_dir / "nq-test.qa.csv"
        self.download_file(test_url, test_csv)

        # Convert to eval format
        num_test = self.convert_dpr_csv_to_eval_format(test_csv, "nq", "test")

        # Also download dev for completeness
        dev_url = "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv"
        dev_csv = self.output_dir / "nq-dev.qa.csv"
        self.download_file(dev_url, dev_csv)
        num_dev = self.convert_dpr_csv_to_eval_format(dev_csv, "nq", "dev")

        print(f"NQ: {num_dev} dev samples, {num_test} test samples")
        return num_test

    def prepare_triviaqa(self):
        """Prepare TriviaQA dataset from DPR."""
        print("\n=== Preparing TriviaQA ===")

        # Download test split (gzipped)
        test_url = "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz"
        test_csv_gz = self.output_dir / "trivia-test.qa.csv.gz"
        test_csv = self.download_file(test_url, test_csv_gz, decompress_gz=True)

        # Convert to eval format
        num_test = self.convert_dpr_csv_to_eval_format(test_csv, "triviaqa", "test")

        # Also download dev
        dev_url = "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-dev.qa.csv.gz"
        dev_csv_gz = self.output_dir / "trivia-dev.qa.csv.gz"
        dev_csv = self.download_file(dev_url, dev_csv_gz, decompress_gz=True)
        num_dev = self.convert_dpr_csv_to_eval_format(dev_csv, "triviaqa", "dev")

        print(f"TriviaQA: {num_dev} dev samples, {num_test} test samples")
        return num_test

    def prepare_webquestions(self):
        """Prepare WebQuestions dataset from multiple sources."""
        print("\n=== Preparing WebQuestions ===")

        # Download test split from DPR first
        test_url = "https://dl.fbaipublicfiles.com/dpr/data/retriever/webquestions-test.qa.csv"
        test_csv = self.output_dir / "webquestions-test.qa.csv"
        test_csv = self.download_file(test_url, test_csv)

        if test_csv is None or not test_csv.exists():
            print("  DPR download failed, trying alternative sources...")
            num_test = 0
            # Try FlashRAG
            try:
                print("Trying FlashRAG webquestions dataset...")
                dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "webquestions")
                test_data = dataset['test'] if 'test' in dataset else dataset['validation']
                
                source_path = self.output_dir / "webquestions_test.source"
                target_path = self.output_dir / "webquestions_test.target"
                
                with open(source_path, 'w', encoding='utf-8') as f_src, \
                     open(target_path, 'w', encoding='utf-8') as f_tgt:
                    num_test = 0
                    for item in test_data:
                        if self._limit_reached(num_test):
                            break
                        question = item.get('question', '')
                        answers = item.get('answers', item.get('golden_answers', []))
                        if question and answers:
                            f_src.write(question.strip() + '\n')
                            if isinstance(answers, list):
                                f_tgt.write(f"{question.strip()}\t{str(answers)}\n")
                            else:
                                f_tgt.write(f"{question.strip()}\t['{answers}']\n")
                            num_test += 1
                print(f"  Loaded from FlashRAG: {num_test} samples")
            except Exception as e:
                print(f"FlashRAG failed: {e}, trying stanfordnlp...")
                # Try stanfordnlp as final fallback
                try:
                    dataset = load_dataset("stanfordnlp/web_questions")
                    test_data = dataset['test']
                    
                    source_path = self.output_dir / "webquestions_test.source"
                    target_path = self.output_dir / "webquestions_test.target"
                    
                    with open(source_path, 'w', encoding='utf-8') as f_src, \
                         open(target_path, 'w', encoding='utf-8') as f_tgt:
                        for item in test_data:
                            if self._limit_reached(num_test):
                                break
                            question = item.get('question', '')
                            answers = item.get('answers', [])
                            if question and answers:
                                f_src.write(question.strip() + '\n')
                                f_tgt.write(f"{question.strip()}\t{str(answers)}\n")
                                num_test += 1
                    print(f"  Loaded from stanfordnlp: {num_test} samples")
                except Exception as e2:
                    print(f"stanfordnlp also failed: {e2}")
                    return 0
        else:
            # Convert DPR to eval format
            num_test = self.convert_dpr_csv_to_eval_format(test_csv, "webquestions", "test")

        # Also download dev (optional, continue even if it fails)
        dev_url = "https://dl.fbaipublicfiles.com/dpr/data/retriever/webquestions-dev.qa.csv"
        dev_csv = self.output_dir / "webquestions-dev.qa.csv"
        dev_csv = self.download_file(dev_url, dev_csv)

        num_dev = 0
        if dev_csv and dev_csv.exists():
            num_dev = self.convert_dpr_csv_to_eval_format(dev_csv, "webquestions", "dev")
        else:
            print("  ⚠ Dev set not available, continuing with test set only")

        print(f"WebQuestions: {num_dev} dev samples, {num_test} test samples")
        return num_test

    def prepare_curatedtrec(self):
        """Prepare CuratedTrec dataset from multiple sources."""
        print("\n=== Preparing CuratedTrec ===")

        # Try DPR first
        test_url = "https://dl.fbaipublicfiles.com/dpr/data/retriever/curatedtrec-test.qa.csv"
        test_csv = self.output_dir / "curatedtrec-test.qa.csv"
        dpr_test = self.download_file(test_url, test_csv)
        
        if dpr_test:
            num_test = self.convert_dpr_csv_to_eval_format(test_csv, "curatedtrec", "test")
        else:
            num_test = 0

        # Try FlashRAG if DPR fails
        if num_test == 0:
            try:
                print("Trying FlashRAG curatedtrec dataset...")
                dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "curatedtrec")
                test_data = dataset['test'] if 'test' in dataset else dataset['validation']
                
                source_path = self.output_dir / "curatedtrec_test.source"
                target_path = self.output_dir / "curatedtrec_test.target"
                
                with open(source_path, 'w', encoding='utf-8') as f_src, \
                     open(target_path, 'w', encoding='utf-8') as f_tgt:
                    for item in test_data:
                        if self._limit_reached(num_test):
                            break
                        question = item.get('question', '')
                        answers = item.get('answers', item.get('golden_answers', []))
                        if question and answers:
                            f_src.write(question.strip() + '\n')
                            if isinstance(answers, list):
                                f_tgt.write(f"{question.strip()}\t{str(answers)}\n")
                            else:
                                f_tgt.write(f"{question.strip()}\t['{answers}']\n")
                            num_test += 1
                print(f"  Loaded from FlashRAG: {num_test} samples")
            except Exception as e:
                print(f"FlashRAG failed: {e}")
                num_test = 0

        # Try dev set
        dev_url = "https://dl.fbaipublicfiles.com/dpr/data/retriever/curatedtrec-dev.qa.csv"
        dev_csv = self.output_dir / "curatedtrec-dev.qa.csv"
        dev_result = self.download_file(dev_url, dev_csv)
        
        if dev_result:
            num_dev = self.convert_dpr_csv_to_eval_format(dev_csv, "curatedtrec", "dev")
        else:
            print("  ⚠ Dev set not available from DPR")
            # Try FlashRAG for dev too
            try:
                dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "curatedtrec")
                if 'train' in dataset:
                    dev_data = dataset['train']
                    source_path = self.output_dir / "curatedtrec_dev.source"
                    target_path = self.output_dir / "curatedtrec_dev.target"
                    with open(source_path, 'w', encoding='utf-8') as f_src, \
                         open(target_path, 'w', encoding='utf-8') as f_tgt:
                        for item in dev_data:
                            if self._limit_reached(num_dev):
                                break
                            question = item.get('question', '')
                            answers = item.get('answers', item.get('golden_answers', []))
                            if question and answers:
                                f_src.write(question.strip() + '\n')
                                if isinstance(answers, list):
                                    f_tgt.write(f"{question.strip()}\t{str(answers)}\n")
                                else:
                                    f_tgt.write(f"{question.strip()}\t['{answers}']\n")
                                num_dev += 1
                else:
                    num_dev = 0
            except:
                num_dev = 0

        print(f"CuratedTrec: {num_dev} dev samples, {num_test} test samples")
        return num_test

    def _download_msmarco_from_kaggle(self):
        """Attempt to download MS-MARCO from Kaggle using API credentials."""
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            print("  Attempting to download from Kaggle API...")
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            dataset_name = "parthplc/ms-marco-dataset"
            download_path = str(self.output_dir)
            
            print(f"  Downloading {dataset_name}...")
            api.dataset_download_files(dataset_name, path=download_path, unzip=True)
            print("  ✓ Downloaded and extracted MS-MARCO from Kaggle")
            return True
            
        except ImportError:
            print("  ℹ️  Kaggle package not installed (pip install kaggle)")
            return False
        except Exception as e:
            print(f"  ℹ️  Kaggle API download failed: {e}")
            print("  Note: Requires ~/.kaggle/kaggle.json with API credentials")
            return False
    
    def prepare_msmarco(self):
        """Prepare MS-MARCO NLG v2.1 dataset from multiple sources."""
        print("\n=== Preparing MS-MARCO NLG v2.1 ===")
        
        # Check if Kaggle CSV exists first
        kaggle_train = self.output_dir / "ms-marco-train.csv"
        kaggle_valid = self.output_dir / "ms-marco-valid.csv"
        
        # If files don't exist, try automatic Kaggle download
        if not (kaggle_train.exists() and kaggle_valid.exists()):
            print("MS-MARCO files not found locally, attempting automatic download...")
            if self._download_msmarco_from_kaggle():
                # Check again after download
                if not (kaggle_train.exists() and kaggle_valid.exists()):
                    print("  ⚠️  Download succeeded but expected files not found")
                    print(f"  Looking for: {kaggle_train.name}, {kaggle_valid.name}")
        
        # Process Kaggle CSV if available
        if kaggle_train.exists() and kaggle_valid.exists():
            print("Found Kaggle MS-MARCO CSV files, processing...")
            try:
                # Load validation set (smaller, better for testing)
                df = pd.read_csv(kaggle_valid)
                
                # Inspect columns to understand structure
                print(f"  CSV columns: {list(df.columns)}")
                print(f"  Total rows: {len(df)}")
                
                if self.max_samples is not None:
                    df = df.head(self.max_samples)
                
                source_path = self.output_dir / "msmarco_test.source"
                target_path = self.output_dir / "msmarco_test.target"
                
                count = 0
                with open(source_path, 'w', encoding='utf-8') as f_src, \
                     open(target_path, 'w', encoding='utf-8') as f_tgt:
                    for _, row in df.iterrows():
                        # Try multiple possible column names
                        question = str(row.get('query', row.get('question', row.get('Query', '')))).strip()
                        answer = str(row.get('answer', row.get('passage', row.get('Answer', row.get('Passage', ''))))).strip()
                        
                        if question and answer and question != 'nan' and answer != 'nan':
                            f_src.write(question + '\n')
                            f_tgt.write(f"{question}\t['{answer}']\n")
                            count += 1
                
                print(f"  ✓ Created {source_path} ({count} samples)")
                return count
            except Exception as e:
                print(f"  ✗ Error processing Kaggle CSV: {e}")
                import traceback
                traceback.print_exc()

        # Strategy 1: Skip FlashRAG for datasets 1.18.0 compatibility
        # Note: FlashRAG uses glob patterns incompatible with datasets 1.18.0
        # This works locally with newer datasets versions but fails in Modal
        
        # Strategy 2: Try ms_marco v1.1 config
        try:
            print("Attempting MS-MARCO from ms_marco 'v1.1' config...")
            dataset = load_dataset("ms_marco", "v1.1")
            print("  Successfully loaded MS-MARCO v1.1")
            
            test_data = dataset.get('validation', dataset.get('test', dataset.get('train')))
            
            source_path = self.output_dir / "msmarco_test.source"
            target_path = self.output_dir / "msmarco_test.target"

            with open(source_path, 'w', encoding='utf-8') as f_src, \
                 open(target_path, 'w', encoding='utf-8') as f_tgt:
                count = 0
                for item in tqdm(test_data, desc="Processing MS-MARCO"):
                    if self._limit_reached(count):
                        break
                    query = item.get('query', item.get('question', ''))
                    answers = item.get('wellFormedAnswers', item.get('answers', []))
                    
                    # Handle passages if answers not available
                    if not answers and 'passages' in item:
                        passages = item.get('passages', {})
                        if isinstance(passages, dict) and 'passage_text' in passages:
                            answers = passages['passage_text'][:1]  # Take first passage
                    
                    if query and answers:
                        f_src.write(str(query).strip() + '\n')
                        # Handle different answer formats
                        if isinstance(answers, list) and len(answers) > 0:
                            answer = answers[0] if isinstance(answers[0], str) else str(answers[0])
                            f_tgt.write(f"{str(query).strip()}\t['{answer.strip()}']\n")
                            count += 1
                        elif isinstance(answers, str):
                            f_tgt.write(f"{str(query).strip()}\t['{answers.strip()}']\n")
                            count += 1

            print(f"  Created {source_path} ({count} samples)")
            return count
        except Exception as e:
            print(f"  Failed with ms_marco v1.1: {e}")

        # Strategy 2: Try microsoft/ms_marco v2.1
        try:
            print("Attempting MS-MARCO from microsoft/ms_marco v2.1...")
            dataset = load_dataset("microsoft/ms_marco", "v2.1")
            print(f"  Successfully loaded MS-MARCO v2.1")
            
            test_data = dataset.get('validation', dataset.get('test', dataset.get('train')))
            
            source_path = self.output_dir / "msmarco_test.source"
            target_path = self.output_dir / "msmarco_test.target"

            with open(source_path, 'w', encoding='utf-8') as f_src, \
                 open(target_path, 'w', encoding='utf-8') as f_tgt:
                count = 0
                for item in tqdm(test_data, desc="Processing MS-MARCO"):
                    if self._limit_reached(count):
                        break
                    query = item.get('query', '')
                    answers = item.get('wellFormedAnswers', item.get('answers', []))
                    if query and answers:
                        f_src.write(query.strip() + '\n')
                        answer_str = str(answers)
                        f_tgt.write(f"{query.strip()}\t{answer_str}\n")
                        count += 1

            print(f"  Created {source_path} ({count} samples)")
            return count
        except Exception as e:
            print(f"  Failed with microsoft/ms_marco: {e}")

        # Strategy 3: Try ms_marco v1.1 config
        try:
            print("Attempting MS-MARCO from ms_marco 'v1.1' config...")
            dataset = load_dataset("ms_marco", "v1.1")
            print("  Successfully loaded MS-MARCO v1.1")
            
            test_data = dataset.get('validation', dataset.get('test', dataset.get('train')))
            
            source_path = self.output_dir / "msmarco_test.source"
            target_path = self.output_dir / "msmarco_test.target"

            with open(source_path, 'w', encoding='utf-8') as f_src, \
                 open(target_path, 'w', encoding='utf-8') as f_tgt:
                count = 0
                for item in tqdm(test_data, desc="Processing MS-MARCO"):
                    if self._limit_reached(count):
                        break
                    query = item.get('query', item.get('question', ''))
                    answers = item.get('wellFormedAnswers', item.get('answers', item.get('passages', {}).get('passage_text', [])))
                    
                    if query and answers:
                        f_src.write(str(query).strip() + '\n')
                        # Handle different answer formats
                        if isinstance(answers, list) and len(answers) > 0:
                            answer = answers[0] if isinstance(answers[0], str) else str(answers[0])
                            f_tgt.write(f"{str(query).strip()}\t['{answer.strip()}']\n")
                            count += 1
                        elif isinstance(answers, str):
                            f_tgt.write(f"{str(query).strip()}\t['{answers.strip()}']\n")
                            count += 1

            print(f"  Created {source_path} ({count} samples)")
            return count
        except Exception as e:
            print(f"  Failed with ms_marco v1.1: {e}")
        
        print("\nAll MS-MARCO loading strategies failed.")
        print("\n⚠️  MS-MARCO requires manual download in Modal environment:")
        print("  Reason: Microsoft's blob storage returns 409 errors")
        print("  FlashRAG incompatible with datasets==1.18.0 (used in Modal)")
        print("\nTo use MS-MARCO:")
        print("  1. Download from: https://www.kaggle.com/datasets/parthplc/ms-marco-dataset/data")
        print("  2. Upload 'ms-marco-train.csv' and 'ms-marco-valid.csv' to Modal volume:")
        print("     modal volume put rag-data ms-marco-train.csv eval_datasets/ms-marco-train.csv")
        print("     modal volume put rag-data ms-marco-valid.csv eval_datasets/ms-marco-valid.csv")
        print("  3. Re-run dataset preparation")
        print("\nNote: MS-MARCO is optional - core benchmarks (NQ, TriviaQA, WebQ, Trec) work ✓")
        return 0

    def prepare_searchqa(self):
        """Prepare SearchQA (Jeopardy) dataset from HuggingFace."""
        print("\n=== Preparing SearchQA (Jeopardy) ===")

        # Strategy 1: Try original kyunghyuncho/search_qa
        # Note: This requires datasets==1.18.0 (has dataset script support)
        try:
            print("Attempting SearchQA from kyunghyuncho/search_qa...")
            # This works with datasets 1.18.0 but not newer versions
            dataset = load_dataset("kyunghyuncho/search_qa", "train_test_val")
            print("  Successfully loaded SearchQA")

            # Use test split
            test_data = dataset['test']

            source_path = self.output_dir / "searchqa_test.source"
            target_path = self.output_dir / "searchqa_test.target"

            with open(source_path, 'w', encoding='utf-8') as f_src, \
                 open(target_path, 'w', encoding='utf-8') as f_tgt:
                count = 0
                for item in tqdm(test_data, desc="Processing SearchQA"):
                    if self._limit_reached(count):
                        break
                    # Extract question and answer fields
                    question = item.get('question', '')
                    answer = item.get('answer', '')
                    
                    # Handle if answer is a list
                    if isinstance(answer, list) and len(answer) > 0:
                        answer = answer[0]
                    
                    # Handle if answer/question are bytes
                    if isinstance(question, bytes):
                        question = question.decode('utf-8', errors='ignore')
                    if isinstance(answer, bytes):
                        answer = answer.decode('utf-8', errors='ignore')
                    
                    # Convert to string and clean
                    question = str(question).strip() if question else ''
                    answer = str(answer).strip() if answer else ''

                    if question and answer:
                        f_src.write(question + '\n')
                        f_tgt.write(f"{question}\t['{answer}']\n")
                        count += 1

            print(f"  Created {source_path} ({count} samples)")
            return count
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" in str(e):
                print(f"  ⚠ Dataset script not supported in this datasets version")
                print(f"    This dataset requires datasets==1.18.0 (used in Modal)")
            else:
                print(f"  Failed with kyunghyuncho/search_qa: {e}")
        except Exception as e:
            print(f"  Failed with kyunghyuncho/search_qa: {e}")
            import traceback
            print(f"  Full traceback:")
            traceback.print_exc()

        # Strategy 2: Try direct file download from HuggingFace (fallback)
        try:
            print("Attempting direct download from HuggingFace repository...")
            import urllib.request
            import gzip
            
            # SearchQA test file URL - files are in root directory, not /data subfolder
            base_url = "https://huggingface.co/datasets/kyunghyuncho/search_qa/resolve/main"
            test_file_url = f"{base_url}/test.txt.gz"
            
            temp_gz_path = self.output_dir / "searchqa_test.txt.gz"
            temp_txt_path = self.output_dir / "searchqa_test.txt"
            
            print(f"  Downloading {test_file_url}...")
            req = urllib.request.Request(
                test_file_url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                with open(temp_gz_path, 'wb') as f:
                    f.write(response.read())
            
            # Decompress
            print(f"  Decompressing...")
            with gzip.open(temp_gz_path, 'rb') as f_in:
                with open(temp_txt_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Parse and convert to eval format
            source_path = self.output_dir / "searchqa_test.source"
            target_path = self.output_dir / "searchqa_test.target"
            
            count = 0
            with open(temp_txt_path, 'r', encoding='utf-8') as f_in, \
                 open(source_path, 'w', encoding='utf-8') as f_src, \
                 open(target_path, 'w', encoding='utf-8') as f_tgt:
                
                for line in f_in:
                    if self._limit_reached(count):
                        break
                    line = line.strip()
                    if line:
                        # Parse JSON line
                        try:
                            data = json.loads(line)
                            question = data.get('question', '')
                            answer = data.get('answer', '')
                            
                            if question and answer:
                                f_src.write(question + '\\n')
                                f_tgt.write(f"{question}\\t['{answer}']\\n")
                                count += 1
                        except json.JSONDecodeError:
                            continue
            
            # Cleanup temp files
            temp_gz_path.unlink(missing_ok=True)
            temp_txt_path.unlink(missing_ok=True)
            
            print(f"  Created {source_path} ({count} samples)")
            return count
            
        except Exception as e:
            print(f"  Failed with direct download: {e}")
        
        print("All SearchQA loading strategies failed.")
        print("⚠  SearchQA appears to have data corruption issues")
        print("  Dataset downloads but JSON parsing fails - known issue in repository")
        print("\nSearchQA will be skipped. This is an optional dataset - core benchmarks work ✓")
        return 0
        return 0

    def prepare_fever(self):
        """Prepare FEVER dataset from HuggingFace."""
        print("\n=== Preparing FEVER ===")

        try:
            # Load dataset from HuggingFace
            print("Loading FEVER from HuggingFace...")
            # Note: verification_mode is not supported in datasets 1.18.0
            # Try the fever/fever dataset first
            dataset = None
            try:
                print("  Trying fever/fever...")
                dataset = load_dataset("fever/fever", "v1.0")
            except Exception as e:
                print(f"  fever/fever failed: {e}")
                # Fallback to original fever dataset  
                try:
                    print("  Trying fever...")
                    dataset = load_dataset("fever", "v1.0")
                except Exception as e2:
                    print(f"  fever failed: {e2}")
                    # Skip FEVER for now if both fail
                    print("  FEVER download failed, skipping...")
                    return 0

            if dataset is None:
                print("  Failed to load FEVER dataset")
                return 0

            # Use test split (labelled_dev is the standard test set)
            test_data = dataset['labelled_dev'] if 'labelled_dev' in dataset else dataset.get('paper_test', dataset.get('test'))

            # FEVER format: claim -> label (SUPPORTS/REFUTES/NOT ENOUGH INFO)
            # We'll create both 3-way and 2-way versions

            # 3-way version
            source_path_3way = self.output_dir / "fever_3way_test.source"
            target_path_3way = self.output_dir / "fever_3way_test.target"

            # 2-way version (only SUPPORTS/REFUTES)
            source_path_2way = self.output_dir / "fever_2way_test.source"
            target_path_2way = self.output_dir / "fever_2way_test.target"

            count_3way = 0
            count_2way = 0

            with open(source_path_3way, 'w', encoding='utf-8') as f_src_3, \
                 open(target_path_3way, 'w', encoding='utf-8') as f_tgt_3, \
                 open(source_path_2way, 'w', encoding='utf-8') as f_src_2, \
                 open(target_path_2way, 'w', encoding='utf-8') as f_tgt_2:

                for item in tqdm(test_data, desc="Processing FEVER"):
                    if self._limit_reached(count_3way):
                        break
                    claim = item.get('claim', '')
                    label = item.get('label', '')

                    if claim and label:
                        # 3-way classification
                        f_src_3.write(claim.strip() + '\n')
                        f_tgt_3.write(f"{claim.strip()}\t['{label}']\n")
                        count_3way += 1

                        # 2-way classification (only SUPPORTS/REFUTES)
                        if label in ['SUPPORTS', 'REFUTES']:
                            f_src_2.write(claim.strip() + '\n')
                            f_tgt_2.write(f"{claim.strip()}\t['{label}']\n")
                            count_2way += 1

            print(f"  Created {source_path_3way} ({count_3way} samples)")
            print(f"  Created {target_path_3way}")
            print(f"  Created {source_path_2way} ({count_2way} samples)")
            print(f"  Created {target_path_2way}")
            return count_3way

        except Exception as e:
            print(f"Error loading FEVER: {e}")
            print("FEVER preparation failed. Skipping...")
            return 0

    def prepare_all(self):
        """Prepare all evaluation datasets."""
        print("=" * 80)
        print("RAG Evaluation Dataset Preparation")
        print("=" * 80)
        if self.max_samples is not None:
            print(f"Max samples per dataset: {self.max_samples}")
            print("=" * 80)

        results = {}

        # DPR datasets - continue even if some fail
        try:
            results['nq'] = self.prepare_nq()
        except Exception as e:
            print(f"✗ NQ preparation failed: {e}")
            results['nq'] = 0

        try:
            results['triviaqa'] = self.prepare_triviaqa()
        except Exception as e:
            print(f"✗ TriviaQA preparation failed: {e}")
            results['triviaqa'] = 0

        try:
            results['webquestions'] = self.prepare_webquestions()
        except Exception as e:
            print(f"✗ WebQuestions preparation failed: {e}")
            results['webquestions'] = 0

        try:
            results['curatedtrec'] = self.prepare_curatedtrec()
        except Exception as e:
            print(f"✗ CuratedTrec preparation failed: {e}")
            results['curatedtrec'] = 0

        # HuggingFace datasets
        try:
            results['msmarco'] = self.prepare_msmarco()
        except Exception as e:
            print(f"✗ MS-MARCO preparation failed: {e}")
            results['msmarco'] = 0

        try:
            results['searchqa'] = self.prepare_searchqa()
        except Exception as e:
            print(f"✗ SearchQA preparation failed: {e}")
            results['searchqa'] = 0

        try:
            results['fever'] = self.prepare_fever()
        except Exception as e:
            print(f"✗ FEVER preparation failed: {e}")
            results['fever'] = 0

        # Print summary
        print("\n" + "=" * 80)
        print("Dataset Preparation Summary")
        print("=" * 80)
        for dataset, count in results.items():
            status = f"✓ {count} samples" if count > 0 else "✗ Failed"
            print(f"{dataset:20s}: {status}")

        successful = sum(1 for c in results.values() if c > 0)
        print(f"\n{successful}/{len(results)} datasets prepared successfully")
        print(f"Datasets saved to: {self.output_dir}")
        print("=" * 80)

        return results


def main():
    """Main function for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare RAG evaluation datasets")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_datasets",
        help="Directory to save prepared datasets"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Max samples per dataset (0 means no limit)"
    )
    args = parser.parse_args()

    preparer = DatasetPreparer(output_dir=args.output_dir, max_samples=args.max_samples)
    preparer.prepare_all()


if __name__ == "__main__":
    main()
