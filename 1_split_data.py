import pandas as pd
import numpy as np

def main():
    SIZE = 'small'
#     SIZE = 'large'

    df = pd.read_csv('ml-latest-small/ratings.csv')
#     df = pd.read_csv('ml-latest/ratings.csv')

    df = df.drop(columns='timestamp')
    userid_cutoff = int(np.ceil(np.mean(df.groupby('userId').count()['rating']) * 2)) * 0.1
    userid_series = df.groupby('userId').count()['rating'] < userid_cutoff
    rare_userid_list = list(userid_series[userid_series.values==True].index)
    
    movieid_cutoff = int(np.ceil(np.mean(df.groupby('userId').count()['rating']) * 2)) * 0.1
    movieid_series = df.groupby('movieId').count()['rating'] < movieid_cutoff
    rare_movieid_list = list(movieid_series[movieid_series.values==True].index)

    rare_user_df = df[df['userId'].isin(rare_userid_list)]
    rare_movie_df = df[df['movieId'].isin(rare_movieid_list)]
    train_df = pd.concat([rare_user_df, rare_movie_df])
    assert(train_df.shape[0] == rare_movie_df.shape[0] + rare_user_df.shape[0])
    train_df = train_df.drop_duplicates()
    selected_idx = list(train_df.index)
    full_idx_list = list(range(len(df)))
    choice_idx_list = list(set(full_idx_list) - set(selected_idx))
    selected_num  = int(np.ceil(len(df)*0.6 - len(train_df) + 1))
    assert(((len(choice_idx_list) + len(train_df)) / len(df)) == 1)

    np.random.seed(2022)

    selected_choices = np.random.choice(choice_idx_list, selected_num, replace=False)
    new_train_df = df.iloc[selected_choices]
    assert(len(selected_choices) == len(new_train_df))
    final_train_df = pd.concat([train_df, new_train_df])
    final_train_df = final_train_df.drop_duplicates()
    assert(len(np.unique(df['userId'])) == len(np.unique(final_train_df['userId'])))
    assert(len(np.unique(df['movieId'])) == len(np.unique(final_train_df['movieId'])))
    raw_test_idx = list(set(full_idx_list) - set(final_train_df.index))
    assert(len(full_idx_list) == len(raw_test_idx) + len(final_train_df.index))
    raw_test = df.iloc[raw_test_idx]

    validation = raw_test[raw_test['userId']%2 == 0]
    test = raw_test[raw_test['userId']%2 == 1]
    assert(len(validation) + len(test) == len(raw_test))
    
    final_train_df.to_csv(f'ratings_{SIZE}_training.csv', index=None)
    validation.to_csv(f'ratings_{SIZE}_validation.csv', index=None)
    test.to_csv(f'ratings_{SIZE}_test.csv', index=None)

if __name__ == "__main__":
    main()
