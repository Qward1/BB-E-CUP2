
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
from typing import List, Optional
from tqdm import tqdm
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if not torch.cuda.is_available():
    print("⚠️ ВНИМАНИЕ: GPU не доступен! Проверьте установку CUDA и PyTorch с поддержкой GPU")
    print("Для установки PyTorch с CUDA выполните:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
else:
    print(f"✓ GPU доступен: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA версия: {torch.version.cuda}")

# NLP библиотеки
import nltk
from nltk.corpus import stopwords
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Загрузка необходимых ресурсов NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Опциональные библиотеки для лемматизации
try:
    import pymorphy3

    PYMORPHY_AVAILABLE = True
except ImportError:
    PYMORPHY_AVAILABLE = False
    print("pymorphy3 не установлен. Используется упрощенная нормализация.")
    print("Для лемматизации установите: pip install pymorphy3 pymorphy3-dicts-ru")


class TextPreprocessor:


    def __init__(self):
        # Инициализация лемматизатора если доступен
        if PYMORPHY_AVAILABLE:
            try:
                self.morph = pymorphy3.MorphAnalyzer()
                self.use_lemmatization = True
            except Exception as e:
                print(f"Ошибка инициализации pymorphy3: {e}")
                self.use_lemmatization = False
        else:
            self.use_lemmatization = False

        # Стоп-слова
        try:
            self.russian_stopwords = set(stopwords.words('russian'))
            self.english_stopwords = set(stopwords.words('english'))
        except:
            self.russian_stopwords = set()
            self.english_stopwords = set()

        # Добавляем частые слова электронной коммерции
        self.custom_stopwords = {
            'товар', 'продукт', 'купить', 'заказать', 'цена', 'руб', 'рубль',
            'product', 'buy', 'order', 'price', 'item', 'sale', 'new',
            'шт', 'штук', 'штука', 'piece', 'pcs', 'артикул', 'код'
        }
        self.all_stopwords = self.russian_stopwords | self.english_stopwords | self.custom_stopwords

    def remove_html(self, text: str) -> str:

        if pd.isna(text):
            return ""
        try:
            soup = BeautifulSoup(str(text), 'html.parser')
            return soup.get_text(separator=' ')
        except:
            text = re.sub(r'<[^>]+>', ' ', str(text))
            return text

    def clean_text(self, text: str) -> str:
        """Очистка текста от специальных символов"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'#\S+|@\S+', '', text)
        text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize_word(self, word: str) -> str:

        if word.isdigit():
            return word

        if self.use_lemmatization and PYMORPHY_AVAILABLE:
            try:
                parsed = self.morph.parse(word)[0]
                return parsed.normal_form
            except:
                return word
        else:
            if re.match(r'[а-яё]+', word):
                word = re.sub(
                    r'(ами|ями|ом|ем|ой|ей|ая|яя|ое|ее|ые|ие|ого|его|ому|ему|ым|им|ых|их|ую|юю|ая|яя|ое|ее|ые|ие)$', '',
                    word)
                word = re.sub(r'(а|я|о|е|ы|и|у|ю|ой|ей|ом|ем|ах|ях)$', '', word)
            elif re.match(r'[a-z]+', word):
                if word.endswith('ies'):
                    word = word[:-3] + 'y'
                elif word.endswith('es'):
                    word = word[:-2]
                elif word.endswith('s') and not word.endswith('ss'):
                    word = word[:-1]
                elif word.endswith('ed'):
                    word = word[:-2]
                elif word.endswith('ing'):
                    word = word[:-3]

            return word

    def process_text(self, text: str, remove_stopwords: bool = True) -> str:

        text = self.remove_html(text)
        text = self.clean_text(text)

        if not text:
            return ""

        tokens = text.split()
        tokens = [self.normalize_word(token) for token in tokens]

        if remove_stopwords:
            tokens = [token for token in tokens
                      if token not in self.all_stopwords
                      and len(token) > 2
                      and not token.isdigit()]

        return ' '.join(tokens)

    def save(self, path: str):

        save_dict = {
            'use_lemmatization': self.use_lemmatization,
            'russian_stopwords': self.russian_stopwords,
            'english_stopwords': self.english_stopwords,
            'custom_stopwords': self.custom_stopwords,
            'all_stopwords': self.all_stopwords
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: str):

        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        preprocessor = cls()
        preprocessor.use_lemmatization = save_dict['use_lemmatization']
        preprocessor.russian_stopwords = save_dict['russian_stopwords']
        preprocessor.english_stopwords = save_dict['english_stopwords']
        preprocessor.custom_stopwords = save_dict['custom_stopwords']
        preprocessor.all_stopwords = save_dict['all_stopwords']

        return preprocessor


class FineTuningDataset(Dataset):


    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 64):
        self.labels = labels
        self.max_length = max_length

        print("Токенизация всех текстов...")
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class EmbeddingExtractor:


    def __init__(self, model_name: str = 'xlm-roberta-base', device: Optional[str] = None):

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
                torch.cuda.set_device(0)
                print(f"✓ Используется GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print("⚠️ GPU не доступен, используется CPU")
        else:
            self.device = torch.device(device)
            if device.startswith('cuda'):
                torch.cuda.set_device(self.device)

        print(f"Используется устройство: {self.device}")

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        self.model_name = model_name

        print(f"Загрузка модели {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.classification_model = None
        self.is_finetuned = False

        if self.device.type == 'cuda':
            self.batch_size = 64
            self.fine_tune_batch_size = 32
            self.accumulation_steps = 2
        else:
            self.batch_size = 16
            self.fine_tune_batch_size = 8
            self.accumulation_steps = 4

        self.max_length = 64

    def save_model(self, save_dir: str):

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\nСохранение модели в {save_dir}...")

        # Сохраняем токенизатор
        self.tokenizer.save_pretrained(save_path / "tokenizer")

        # Сохраняем основную модель (encoder)
        self.model.save_pretrained(save_path / "encoder")

        # Сохраняем модель классификации если она обучена
        if self.classification_model is not None:
            self.classification_model.save_pretrained(save_path / "classifier")

        # Сохраняем метаданные
        metadata = {
            'model_name': self.model_name,
            'is_finetuned': self.is_finetuned,
            'max_length': self.max_length,
            'batch_size': self.batch_size
        }

        with open(save_path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"✓ Модель успешно сохранена в {save_dir}")

    @classmethod
    def load_model(cls, load_dir: str, device: Optional[str] = None):

        load_path = Path(load_dir)

        if not load_path.exists():
            raise ValueError(f"Директория {load_dir} не существует")

        print(f"\nЗагрузка модели из {load_dir}...")

        # Загружаем метаданные
        with open(load_path / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Создаем экстрактор с базовыми настройками
        extractor = cls(model_name=metadata['model_name'], device=device)

        # Загружаем токенизатор
        extractor.tokenizer = AutoTokenizer.from_pretrained(load_path / "tokenizer")

        # Загружаем encoder
        extractor.model = AutoModel.from_pretrained(load_path / "encoder").to(extractor.device)
        extractor.model.eval()

        # Загружаем модель классификации если она есть
        classifier_path = load_path / "classifier"
        if classifier_path.exists():
            extractor.classification_model = AutoModelForSequenceClassification.from_pretrained(
                classifier_path
            ).to(extractor.device)
            extractor.classification_model.eval()

        # Восстанавливаем метаданные
        extractor.is_finetuned = metadata['is_finetuned']
        extractor.max_length = metadata['max_length']
        extractor.batch_size = metadata['batch_size']

        print(f"✓ Модель успешно загружена из {load_dir}")
        if extractor.is_finetuned:
            print("✓ Загружена FINE-TUNED модель")
        else:
            print("✓ Загружена PRE-TRAINED модель")

        return extractor

    def fine_tune(
            self,
            texts: List[str],
            labels: List[int],
            num_epochs: int = 1,
            learning_rate: float = 2e-5,
            validation_split: float = 0.15,
            sample_size: Optional[int] = 10000,
            freeze_layers: bool = True
    ):

        print("\n" + "=" * 60)
        print("FINE-TUNING МОДЕЛИ")
        print("=" * 60)

        if sample_size and len(texts) > sample_size:
            print(f"Используем подвыборку: {sample_size} из {len(texts)} примеров")
            _, texts, _, labels = train_test_split(
                texts, labels,
                test_size=sample_size / len(texts),
                stratify=labels,
                random_state=42
            )

        self.classification_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        ).to(self.device)

        if freeze_layers:
            if hasattr(self.classification_model, 'roberta'):
                base_model = self.classification_model.roberta
            elif hasattr(self.classification_model, 'bert'):
                base_model = self.classification_model.bert
            elif hasattr(self.classification_model, 'base_model'):
                base_model = self.classification_model.base_model
            else:
                base_model = None

            if base_model:
                for param in base_model.embeddings.parameters():
                    param.requires_grad = False

                if hasattr(base_model, 'encoder'):
                    encoder_layers = base_model.encoder.layer
                    layers_to_freeze = max(0, len(encoder_layers) - 2)
                    for i in range(layers_to_freeze):
                        for param in encoder_layers[i].parameters():
                            param.requires_grad = False
                    print(f"Заморожено {layers_to_freeze} из {len(encoder_layers)} слоев энкодера")

        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=validation_split, random_state=42, stratify=labels
        )

        print(f"Размер обучающей выборки: {len(X_train)}")
        print(f"Размер валидационной выборки: {len(X_val)}")

        train_dataset = FineTuningDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = FineTuningDataset(X_val, y_val, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = AdamW(
            [p for p in self.classification_model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        best_val_accuracy = 0
        best_model_state = None

        for epoch in range(num_epochs):
            print(f"\nЭпоха {epoch + 1}/{num_epochs}")

            self.classification_model.train()
            train_loss = 0

            for batch in tqdm(train_loader, desc="Обучение"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.classification_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.classification_model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            self.classification_model.eval()
            val_predictions = []
            val_true = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Валидация"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.classification_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    predictions = torch.argmax(outputs.logits, dim=-1)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())

            val_accuracy = accuracy_score(val_true, val_predictions)
            val_f1 = f1_score(val_true, val_predictions, average='weighted')

            print(f"Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = self.classification_model.state_dict()
                print(f"✓ Сохранена лучшая модель (Accuracy: {best_val_accuracy:.4f})")

        self.classification_model.load_state_dict(best_model_state)

        if hasattr(self.classification_model, 'roberta'):
            self.model = self.classification_model.roberta
        elif hasattr(self.classification_model, 'bert'):
            self.model = self.classification_model.bert
        elif hasattr(self.classification_model, 'base_model'):
            self.model = self.classification_model.base_model
        else:
            for name, module in self.classification_model.named_children():
                if 'classifier' not in name.lower() and 'dropout' not in name.lower():
                    self.model = module
                    break

        self.model.eval()
        self.is_finetuned = True

        print(f"\nFine-tuning завершен! Лучшая точность: {best_val_accuracy:.4f}")
        print("=" * 60)

    def get_transformer_embeddings(self, texts: List[str], prefix: str) -> pd.DataFrame:

        embeddings = []

        model_desc = "fine-tuned" if self.is_finetuned else "pre-trained"

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size),
                          desc=f"Извлечение {prefix} эмбеддингов ({model_desc})"):
                batch_texts = texts[i:i + self.batch_size]
                batch_texts = [text if text else " " for text in batch_texts]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)

                outputs = self.model(**encoded)

                attention_mask = encoded['attention_mask']
                token_embeddings = outputs.last_hidden_state

                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask

                embeddings.append(mean_pooled.cpu().numpy())

        embeddings = np.vstack(embeddings)

        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f"{prefix}_emb_{i}" for i in range(embeddings.shape[1])]
        )

        return embedding_df


def train_and_save_model(
        df_train: pd.DataFrame,
        save_dir: str = "./saved_model",
        description_col: str = 'description',
        name_col: str = 'name_rus',
        target_col: str = 'resolution',
        model_name: str = 'xlm-roberta-base',
        device: Optional[str] = None,
        fine_tune_epochs: int = 1,
        fine_tune_lr: float = 5e-5,
        fine_tune_sample_size: Optional[int] = 10000,
        freeze_layers: bool = True
) -> None:

    print("=" * 60)
    print("РЕЖИМ ОБУЧЕНИЯ")
    print(f"Размер train: {len(df_train)} строк")
    print("=" * 60)

    # Инициализация препроцессора и экстрактора
    preprocessor = TextPreprocessor()
    extractor = EmbeddingExtractor(model_name=model_name, device=device)

    # Копируем DataFrame
    df_train_copy = df_train.copy()

    # Заполнение пропущенных значений
    for col in [description_col, name_col]:
        if col in df_train_copy.columns:
            df_train_copy[col] = df_train_copy[col].fillna('')
        else:
            print(f"Предупреждение: колонка '{col}' не найдена в train данных")
            df_train_copy[col] = ''

    # Предобработка текстов
    print("\n1. Предобработка текстов...")
    train_descriptions = [
        preprocessor.process_text(text, remove_stopwords=True)
        for text in tqdm(df_train_copy[description_col], desc="Train: описания")
    ]

    train_names = [
        preprocessor.process_text(text, remove_stopwords=True)
        for text in tqdm(df_train_copy[name_col], desc="Train: названия")
    ]

    train_combined_texts = [
        f"{name} [SEP] {desc}" for name, desc in zip(train_names, train_descriptions)
    ]

    # Fine-tuning
    if target_col in df_train_copy.columns:
        print("\n2. Fine-tuning модели на задаче классификации...")

        labels = df_train_copy[target_col].values.astype(int)

        unique, counts = np.unique(labels, return_counts=True)
        print(f"Распределение классов в train:")
        for cls, cnt in zip(unique, counts):
            print(f"  Класс {cls}: {cnt} ({cnt / len(labels) * 100:.2f}%)")

        extractor.fine_tune(
            texts=train_combined_texts,
            labels=labels,
            num_epochs=fine_tune_epochs,
            learning_rate=fine_tune_lr,
            sample_size=fine_tune_sample_size,
            freeze_layers=freeze_layers
        )
    else:
        print(f"\n⚠️ Предупреждение: колонка '{target_col}' не найдена. Fine-tuning пропущен.")

    # Сохранение модели и препроцессора
    print("\n3. Сохранение модели...")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    extractor.save_model(save_dir)
    preprocessor.save(save_path / "preprocessor.pkl")

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"Модель сохранена в: {save_dir}")
    print("=" * 60)


def inference_embeddings(
        df_test: pd.DataFrame,
        model_dir: str = "./saved_model",
        description_col: str = 'description',
        name_col: str = 'name_rus',
        device: Optional[str] = None
) -> pd.DataFrame:

    print("=" * 60)
    print("РЕЖИМ ИНФЕРЕНСА")
    print(f"Размер test: {len(df_test)} строк")
    print("=" * 60)

    # Загрузка модели и препроцессора
    print("\n1. Загрузка модели и препроцессора...")
    model_path = Path(model_dir)

    if not model_path.exists():
        raise ValueError(f"Модель не найдена в {model_dir}. Сначала обучите модель с помощью train_and_save_model()")

    extractor = EmbeddingExtractor.load_model(model_dir, device=device)
    preprocessor = TextPreprocessor.load(model_path / "preprocessor.pkl")

    # Копируем DataFrame
    df_test_copy = df_test.copy()

    # Заполнение пропущенных значений
    for col in [description_col, name_col]:
        if col in df_test_copy.columns:
            df_test_copy[col] = df_test_copy[col].fillna('')
        else:
            print(f"Предупреждение: колонка '{col}' не найдена в test данных")
            df_test_copy[col] = ''

    # Предобработка текстов
    print("\n2. Предобработка текстов...")
    test_descriptions = [
        preprocessor.process_text(text, remove_stopwords=True)
        for text in tqdm(df_test_copy[description_col], desc="Test: описания")
    ]

    test_names = [
        preprocessor.process_text(text, remove_stopwords=True)
        for text in tqdm(df_test_copy[name_col], desc="Test: названия")
    ]

    # Извлечение эмбеддингов
    print("\n3. Извлечение эмбеддингов...")
    desc_embeddings = extractor.get_transformer_embeddings(test_descriptions, prefix='desc')
    name_embeddings = extractor.get_transformer_embeddings(test_names, prefix='name')

    # Объединение эмбеддингов
    test_embeddings = pd.concat([desc_embeddings, name_embeddings], axis=1)
    test_embeddings.index = df_test.index

    print("\n" + "=" * 60)
    print("ИНФЕРЕНС ЗАВЕРШЕН!")
    print(f"Test эмбеддинги: {test_embeddings.shape}")
    print(f"Колонки эмбеддингов:")
    print(f"  - Описание: {desc_embeddings.shape[1]} признаков")
    print(f"  - Название: {name_embeddings.shape[1]} признаков")
    if extractor.is_finetuned:
        print("✓ Использована FINE-TUNED модель")
    else:
        print("✓ Использована PRE-TRAINED модель")
    print("=" * 60)

    return test_embeddings




import joblib
import pandas as pd




def apply_pretrained_reducer(X_test, embedding_blocks, reducer_path=".\\models\\PCA.pkl"):


    # Загружаем предобученные объекты
    fitted_objects = joblib.load(reducer_path)

    reduced_features = []

    for i, block in enumerate(embedding_blocks):
        # масштабируем тестовый блок
        X_scaled = fitted_objects['scalers'][i].transform(X_test[block])

        # применяем обученный редьюсер
        reducer = fitted_objects['reducers'][i]
        X_reduced = reducer.transform(X_scaled)

        # формируем колонки
        if hasattr(reducer, "components_"):  # PCA или SVD
            col_names = [f"block{i}_svd_{j}" for j in range(X_reduced.shape[1])]
        else:  # вдруг кортеж (SVD+PCA)
            col_names = [f"block{i}_part_{j}" for j in range(X_reduced.shape[1])]

        df_reduced = pd.DataFrame(X_reduced, index=X_test.index, columns=col_names)
        reduced_features.append(df_reduced)

    return pd.concat(reduced_features, axis=1)



