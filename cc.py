#%%
import pandas as pd

# 讀取上傳的CSV文件
file_path = 'Suicide_Detection.csv'
data = pd.read_csv(file_path)

# 檢查數據的前幾行和基本信息
data_info = data.info()
data_head = data.head()
#%%
print(data_head)

# %%
# 分析數據集中的類別分佈
class_distribution = data['class'].value_counts()

class_distribution

# %%
from sklearn.feature_extraction.text import CountVectorizer

# 將數據集按類別進行分割
suicide_texts = data[data['class'] == 'suicide']['text']
non_suicide_texts = data[data['class'] == 'non-suicide']['text']

# 進行詞頻分析
vectorizer = CountVectorizer(stop_words='english', max_features=20)

# 對suicide類別進行詞頻分析
suicide_word_counts = vectorizer.fit_transform(suicide_texts)
suicide_top_words = pd.DataFrame(suicide_word_counts.sum(axis=0), 
                                 columns=vectorizer.get_feature_names_out()).T
suicide_top_words.columns = ['frequency']
suicide_top_words = suicide_top_words.sort_values(by='frequency', ascending=False)

# 對non-suicide類別進行詞頻分析
non_suicide_word_counts = vectorizer.fit_transform(non_suicide_texts)
non_suicide_top_words = pd.DataFrame(non_suicide_word_counts.sum(axis=0), 
                                     columns=vectorizer.get_feature_names_out()).T
non_suicide_top_words.columns = ['frequency']
non_suicide_top_words = non_suicide_top_words.sort_values(by='frequency', ascending=False)

suicide_top_words, non_suicide_top_words

# %%
from textblob import TextBlob

# 定義一個函數來計算情感傾向
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# 計算suicide和non-suicide文本的情感傾向
suicide_sentiment = suicide_texts.apply(get_sentiment)
non_suicide_sentiment = non_suicide_texts.apply(get_sentiment)
#%%
# 計算情感傾向的平均值
suicide_sentiment_mean = suicide_sentiment.mean()
non_suicide_sentiment_mean = non_suicide_sentiment.mean()
#%%
print(suicide_sentiment_mean, non_suicide_sentiment_mean)

# %%
