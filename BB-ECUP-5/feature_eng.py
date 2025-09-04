
import pandas as pd
import numpy as np
import pickle
import os
from category_encoders import CatBoostEncoder
import joblib

class MetaDataProcessor:


    def __init__(self, encoder_path='catboost_encoder.pkl'):

        self.encoder_path = encoder_path
        self.encoder = None
        self.threshold_cat = 40
        self.threshold_brand = 50

        # Загружаем предобученный энкодер
        self._load_encoder()

    def _load_encoder(self):
        """Загрузка предобученного CatBoost энкодера из файла"""
        if not os.path.exists(self.encoder_path):
            print(f"ВНИМАНИЕ: файл {self.encoder_path} не существует!")
            self.encoder = None
            return

        try:
            with open(self.encoder_path, 'rb') as f:
                self.encoder = joblib.load(f)
            print(f"✓ Энкодер успешно загружен из {self.encoder_path}")

            # Проверяем, что загруженный объект - это действительно CatBoostEncoder
            if hasattr(self.encoder, 'transform'):
                print(f"  Тип энкодера: {type(self.encoder).__name__}")
                if hasattr(self.encoder, 'cols'):
                    print(f"  Колонки для кодирования: {self.encoder.cols}")
            else:
                print("ВНИМАНИЕ: Загруженный объект не является валидным энкодером!")
                self.encoder = None

        except Exception as e:
            print(f"ОШИБКА при загрузке энкодера: {e}")
            print(f"Тип ошибки: {type(e).__name__}")
            self.encoder = None

    def save_encoder(self):
        """Сохранение текущего энкодера в файл"""
        if self.encoder is not None:
            with open(self.encoder_path, 'wb') as f:
                pickle.dump(self.encoder, f)
            print(f"Энкодер сохранен в {self.encoder_path}")

    def create_and_save_encoder(self, train_df, target_values):

        cat_cols = ["CommercialTypeName4", 'brand_name', 'SellerID']

        # Проверяем наличие нужных колонок
        missing_cols = [col for col in cat_cols if col not in train_df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки: {missing_cols}")

        print(f"Обучение CatBoost энкодера на {len(train_df)} записях...")
        self.encoder = CatBoostEncoder(cols=cat_cols, random_state=42, handle_unknown="ignore")
        self.encoder.fit(train_df[cat_cols], target_values)

        # Сохраняем энкодер
        self.save_encoder()
        print(f"✓ Энкодер создан и сохранен в {self.encoder_path}")

    def process(self, meta_train, meta_test, train_encoder=False, pred_df=None, skip_encoding=False):

        # Создаем копии для безопасности
        meta_train = meta_train.copy()
        meta_test = meta_test.copy()

        print("Шаг 1: Feature Engineering...")
        # Применяем feature engineering
        meta_train, meta_test = self._preprocess_features(meta_train, meta_test)

        print("Шаг 2: Оптимизация малых брендов и категорий...")
        # Оптимизация малых брендов и категорий
        meta_train, meta_test = self._optimize_small_brands(meta_train, meta_test)

        # Применяем CatBoost энкодинг
        if skip_encoding:
            print("Шаг 3: CatBoost энкодирование пропущено (skip_encoding=True)")
        elif train_encoder:
            if pred_df is None:
                raise ValueError("Для обучения нового энкодера необходим pred_df с предсказаниями!")
            print("Шаг 3: Обучение нового CatBoost энкодера...")
            meta_train, meta_test = self._train_and_apply_catboost_encoding(meta_train, meta_test, pred_df)
        else:
            if self.encoder is None:
                print("ВНИМАНИЕ: Энкодер не загружен!")
                print("Варианты действий:")
                print("1. Проверьте путь к файлу энкодера")
                print("2. Используйте skip_encoding=True для пропуска энкодирования")
                print("3. Используйте train_encoder=True и передайте pred_df для обучения нового энкодера")
                raise ValueError("Энкодер не загружен. См. варианты действий выше.")

            print("Шаг 3: Применение загруженного CatBoost энкодера...")
            meta_train, meta_test = self._apply_catboost_encoding(meta_train, meta_test)

        print("✓ Обработка завершена!")
        return meta_train, meta_test

    def _optimize_small_brands(self, meta_train, meta_test):

        meta_train_opt = meta_train.copy()
        meta_test_opt = meta_test.copy()

        # Обработка брендов
        brand_counts = meta_train_opt['brand_name'].value_counts()
        brand_map = {brand: (brand if count >= self.threshold_brand else "Other")
                     for brand, count in brand_counts.items()}
        meta_train_opt['brand_name'] = meta_train_opt['brand_name'].map(brand_map).fillna("Other")
        meta_test_opt['brand_name'] = meta_test_opt['brand_name'].map(brand_map).fillna("Other")

        # Обработка категорий
        cat_counts = meta_train_opt['CommercialTypeName4'].value_counts()
        cat_map = {cat: (cat if count >= self.threshold_cat else "Other")
                   for cat, count in cat_counts.items()}
        meta_train_opt['CommercialTypeName4'] = meta_train_opt['CommercialTypeName4'].map(cat_map).fillna("Other")
        meta_test_opt['CommercialTypeName4'] = meta_test_opt['CommercialTypeName4'].map(cat_map).fillna("Other")

        return meta_train_opt, meta_test_opt

    def _train_and_apply_catboost_encoding(self, train_df, test_df, pred_df):

        target_col = "resolution"
        cat_cols = ["CommercialTypeName4", 'brand_name', 'SellerID']

        # Добавляем resolution в test_df по порядку
        test_with_res = test_df.copy()
        test_with_res[target_col] = pred_df["prediction"].values

        # Объединяем train и test для обучения энкодера
        combined = pd.concat(
            [train_df[cat_cols + [target_col]], test_with_res[cat_cols + [target_col]]],
            axis=0
        )

        # Обучаем CatBoostEncoder
        self.encoder = CatBoostEncoder(cols=cat_cols, random_state=42, handle_unknown="ignore")
        self.encoder.fit(combined[cat_cols], combined[target_col])

        # Сохраняем обученный энкодер
        self.save_encoder()

        # Кодируем train и test
        train_enc = train_df.copy()
        test_enc = test_df.copy()

        train_enc[cat_cols] = self.encoder.transform(train_df[cat_cols])
        test_enc[cat_cols] = self.encoder.transform(test_df[cat_cols])

        return train_enc, test_enc

    def _apply_catboost_encoding(self, train_df, test_df):

        cat_cols = ["CommercialTypeName4", 'brand_name', 'SellerID']

        # Кодируем train и test
        train_enc = train_df.copy()
        test_enc = test_df.copy()

        train_enc[cat_cols] = self.encoder.transform(train_df[cat_cols])
        test_enc[cat_cols] = self.encoder.transform(test_df[cat_cols])

        return train_enc, test_enc

    def _preprocess_features(self, meta_train, meta_test):

        full_meta = [meta_train, meta_test]

        for df in full_meta:
            # Общее количество рейтингов
            df["rating_count_total"] = (
                df[["rating_1_count", "rating_2_count", "rating_3_count",
                    "rating_4_count", "rating_5_count"]].sum(axis=1))

            # Доля единичных рейтингов
            df["rating_share_1"] = df["rating_1_count"] / df["rating_count_total"].replace(0, np.nan)
            df['rating_share_1'] = df['rating_share_1'].fillna(0)

            # Баланс рейтингов
            pos = df["rating_4_count"] + df["rating_5_count"]
            neg = df["rating_1_count"] + df["rating_2_count"]
            df["rating_balance"] = pos / (neg + 1)
            df['rating_balance'] = df['rating_balance'].fillna(0)

            # Общие продажи и возвраты
            df["sales_total"] = df["item_count_sales90"]
            df["returns_total"] = df["item_count_returns90"]

            # Метрики по периодам
            for period in [7, 30, 90]:
                df[f"return_rate_{period}"] = df[f"item_count_returns{period}"] / (df[f"item_count_sales{period}"] + 1)
                if f"item_count_fake_returns{period}" in df.columns:
                    df[f"fake_return_rate_{period}"] = df[f"item_count_fake_returns{period}"] / (
                            df[f"item_count_sales{period}"] + 1)
                df[f"avg_check_{period}"] = df[f"GmvTotal{period}"] / (df[f"item_count_sales{period}"] + 1)

            # Тренд возвратов
            df["return_trend"] = df["item_count_returns90"] / (
                    df["item_count_returns7"] + df["item_count_returns30"] + 1)

            # Рост GMV
            df["gmv_growth_rate"] = df["GmvTotal30"] / (df["GmvTotal90"] + 1)

            # Комментарии на продажу
            df["comments_per_sale"] = df["comments_published_count"] / (df["item_count_sales90"] + 1)

            # Медиа на товар
            df["media_per_item"] = (df["photos_published_count"] + df["videos_published_count"]) / (
                    df["ItemVarietyCount"] + 1)

            # Флаги
            df["fast_growth_flag"] = (df["item_count_sales7"] > df["item_count_sales30"] * 0.8).astype(int)
            df["is_new_item"] = (df["item_time_alive"] < 30).astype(int)
            df["is_new_seller"] = (df["seller_time_alive"] < 90).astype(int)

            # Дополнительные признаки
            df['count_bad_ratings'] = df['rating_1_count'] + df['rating_2_count']
            df['count_good_ratings'] = df['rating_4_count'] + df['rating_5_count']
            df['returns_per_sale'] = df['returns_total'] / (df['sales_total'] + 1)
            df['is_very_new_item'] = (df['item_time_alive'] < 14).astype(int)
            df['new_new'] = df['is_new_item'] * df['is_new_seller']

            # Отношение стоимости возвратов
            df[f"return_value_ratio_money{30}"] = df[f"ExemplarReturnedValueTotal{30}"] / (
                    df[f"GmvTotal{30}"] + 1)
            df[f"return_value_ratio_money{90}"] = df[f"ExemplarReturnedValueTotal{90}"] / (
                    df[f"GmvTotal{90}"] + 1)

        # Статистика по продавцам
        seller_stats = meta_train.groupby("SellerID").agg({
            "sales_total": "mean",
            "returns_total": "mean",
            "rating_balance": "mean",
            "item_time_alive": "mean",
            "ItemID": "nunique",
            "PriceDiscounted": "mean"
        }).rename(columns={
            "sales_total": "seller_avg_sales",
            "returns_total": "seller_avg_returns",
            "rating_balance": "seller_avg_rating_balance",
            "item_time_alive": "seller_avg_item_lifetime",
            "ItemID": "seller_item_diversity",
            "PriceDiscounted": "seller_avg_price"
        })

        # Дополнительная статистика по продавцам для train
        seller_stats_train = meta_train.groupby("SellerID").agg(
            seller_total_items_seller=("ItemVarietyCount", "sum"),
            seller_mean_time=("item_time_alive", "mean"),
        ).reset_index()
        meta_train = meta_train.merge(seller_stats_train, on="SellerID", how="left")
        meta_train["items_per_seller"] = meta_train["ItemVarietyCount"] / (meta_train["seller_total_items_seller"] + 1)
        meta_train["seller_activity_ratio"] = meta_train["item_time_alive"] / (meta_train["seller_mean_time"] + 1)

        # Дополнительная статистика по продавцам для test
        seller_stats_test = meta_test.groupby("SellerID").agg(
            seller_total_items_seller=("ItemVarietyCount", "sum"),
            seller_mean_time=("item_time_alive", "mean"),
        ).reset_index()
        meta_test = meta_test.merge(seller_stats_test, on="SellerID", how="left")
        meta_test["items_per_seller"] = meta_test["ItemVarietyCount"] / (meta_test["seller_total_items_seller"] + 1)
        meta_test["seller_activity_ratio"] = meta_test["item_time_alive"] / (meta_test["seller_mean_time"] + 1)

        # Объединяем со статистикой продавцов
        meta_train = meta_train.merge(seller_stats, on="SellerID", how="left")
        meta_test = meta_test.merge(seller_stats, on="SellerID", how="left")

        # Удаляем ненужные колонки
        cols_to_drop = ['ItemID', 'item_count_fake_returns7', 'item_count_fake_returns30',
                        'item_count_sales7', 'item_count_sales30', 'item_count_sales90',
                        'item_count_returns7', 'item_count_returns30', 'item_count_returns90',
                        'GmvTotal7', 'GmvTotal30', 'GmvTotal90',
                        'ExemplarReturnedValueTotal7', 'ExemplarReturnedValueTotal30',
                        'ExemplarReturnedValueTotal90', 'OrderAcceptedCountTotal7',
                        'OrderAcceptedCountTotal30', 'OrderAcceptedCountTotal90',
                        'ExemplarReturnedCountTotal7', 'ExemplarReturnedCountTotal30',
                        'ExemplarReturnedCountTotal90', 'ExemplarReturnedValueTotal7',
                        'ExemplarReturnedValueTotal30', 'ExemplarReturnedValueTotal90',
                        'ExemplarAcceptedCountTotal7', 'ExemplarAcceptedCountTotal30',
                        'ExemplarAcceptedCountTotal90', 'comments_published_count',
                        'photos_published_count', 'videos_published_count', 'sales_total',
                        'return_rate_30', 'return_rate_90', 'gmv_growth_rate', 'media_per_item',
                        'seller_avg_price', 'avg_check_7']

        meta_test = meta_test.drop(columns=cols_to_drop, axis=1)
        meta_train = meta_train.drop(columns=cols_to_drop, axis=1)

        return meta_train, meta_test




