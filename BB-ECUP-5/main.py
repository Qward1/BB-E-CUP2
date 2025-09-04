
import pandas as pd
import pred
from extract_text_feature import prepare_text_features
import feature_eng as fe
import importlib
import extract_emb as ee
importlib.reload(ee)
importlib.reload(fe)



def final(df_path):
    df_test = pd.read_csv(df_path)
    df_train = pd.read_csv('.\\ml_ozon_сounterfeit_train.csv')

    df_test[['description', 'brand_name']] = df_test[['description', 'brand_name']].fillna('Unknown')
    df_test = df_test.fillna(0)

    meta_train = df_train.drop(['id', 'description', 'name_rus'], axis=1)
    meta_test = df_test.drop(['id', 'description', 'name_rus'], axis=1)
    meta_train['SellerID'] = meta_train['SellerID'].astype(object)
    meta_test['SellerID'] = meta_test['SellerID'].astype(object)
    text_test = df_test[['description', 'name_rus']]

    processor = fe.MetaDataProcessor(
        encoder_path='.\\models\\catboost_encoder.pkl'
    )


    meta_train, meta_test = processor.process(meta_train, meta_test)

    test_embeddings = ee.inference_embeddings(
        df_test=text_test,
        model_dir=".\\resources",
        description_col='description',
        name_col='name_rus'
    )

    features_text_test = prepare_text_features(text_test)

    desc_columns = test_embeddings.filter(like='desc_').columns.tolist()
    name_columns = test_embeddings.filter(like='name_').columns.tolist()
    desc_name_columns = [desc_columns, name_columns]

    pca_test = ee.apply_pretrained_reducer(test_embeddings, embedding_blocks=desc_name_columns)

    pca_test['name_embed_mean'] = pca_test.mean(axis=1)
    pca_test['name_embed_std'] = pca_test.std(axis=1)
    pca_test['name_embed_max'] = pca_test.max(axis=1)
    pca_test['name_embed_min'] = pca_test.min(axis=1)

    con_test = pd.concat([pca_test, meta_test, features_text_test], axis=1)

    preds = pred.ensemble_predict_proba(con_test)

    submission = pd.DataFrame({
        'id': df_test['id'],
        'prediction': preds > 0.4
    })
    submission.to_csv('.\\output\\submission.csv', index=False)

    return '.\\output\\submission.csv'


